/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Verified lowering facts for the narrow packed-AS3 local-storage lane.
//!
//! The physical carrier representation is a storage property, not Rust pointer
//! provenance. This module therefore does not extend `MirPointerKind`. Instead,
//! immediately before dialect conversion it performs a closed-world walk over
//! MIR, proves every use of an eligible compiler-owned local address, and only
//! after the complete proof succeeds stamps the exact physical LLVM type on the
//! MIR operations that consume the address.
//!
//! No lowering converter reconstructs carrier identity from `OperandsInfo` or
//! from an LLVM defining-op chain. If a carrier address crosses a call, cast,
//! block-argument edge, return, pointer offset, nested projection, or any other
//! unmodelled operation, preparation fails before any MIR operation is lowered.

use crate::convert::target_stable_storage::{StorageRewriteOptions, target_stable_storage_type};
use crate::convert::types::{
    PackedSharedInternalAbiInfo, StructLayoutInfo, build_struct_slot_map, convert_type,
    is_zero_sized_type, packed_shared_internal_abi_info,
};
use dialect_mir::ops::{
    MirAllocaOp, MirArrayElementAddrOp, MirAssertOp, MirCallOp, MirCastOp, MirCondBranchOp,
    MirDbgValueListOp, MirDbgValueOp, MirFieldAddrOp, MirGotoOp, MirLoadOp, MirPtrOffsetOp,
    MirReturnOp, MirStoreOp,
};
use dialect_mir::types::{MirPtrType, MirStructType};
use llvm_export::types as llvm_types;
use pliron::builtin::attributes::TypeAttr;
use pliron::builtin::types::{FP32Type, FP64Type, IntegerType};
use pliron::context::{Context, Ptr};
use pliron::identifier::Identifier;
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::{TypeHandle, Typed};
use pliron::value::Value;
use rustc_hash::FxHashMap;

const CARRIER_STORAGE_TYPE_KEY: &str = "cuda_oxide_packed_shared_carrier_storage_type";
const CARRIER_GEP_SOURCE_TYPE_KEY: &str = "cuda_oxide_packed_shared_carrier_gep_source_type";

#[derive(Clone, Copy, Debug)]
struct CarrierAddress {
    physical_pointee: TypeHandle,
    projection_depth: u8,
}

#[derive(Default)]
struct CarrierFactPlan {
    storage_types: Vec<(Ptr<Operation>, TypeHandle)>,
    gep_source_types: Vec<(Ptr<Operation>, TypeHandle)>,
}

impl CarrierFactPlan {
    fn plan_storage_type(&mut self, operation: Ptr<Operation>, ty: TypeHandle) {
        self.storage_types.push((operation, ty));
    }

    fn plan_gep_source_type(&mut self, operation: Ptr<Operation>, ty: TypeHandle) {
        self.gep_source_types.push((operation, ty));
    }

    fn apply(self, ctx: &mut Context) {
        for (operation, ty) in self.storage_types {
            set_type_attr(ctx, operation, CARRIER_STORAGE_TYPE_KEY, ty);
        }
        for (operation, ty) in self.gep_source_types {
            set_type_attr(ctx, operation, CARRIER_GEP_SOURCE_TYPE_KEY, ty);
        }
    }
}

fn attr_key(name: &str) -> Identifier {
    Identifier::try_new(name.to_string()).expect("static carrier attribute key must be valid")
}

fn get_type_attr(ctx: &Context, op: Ptr<Operation>, name: &str) -> Option<TypeHandle> {
    op.deref(ctx)
        .attributes
        .get::<TypeAttr>(&attr_key(name))
        .map(|attr| attr.get_type(ctx))
}

fn set_type_attr(ctx: &mut Context, op: Ptr<Operation>, name: &str, ty: TypeHandle) {
    op.deref_mut(ctx)
        .attributes
        .set(attr_key(name), TypeAttr::new(ty));
}

/// Physical storage type proven for `mir.alloca`, `mir.load`, or `mir.store`.
///
/// The attribute is created only by [`prepare_packed_shared_local_storage`]
/// after the complete whole-tree carrier plan validates, then is consumed
/// mechanically during lowering.
pub(crate) fn carrier_storage_type(ctx: &Context, op: Ptr<Operation>) -> Option<TypeHandle> {
    get_type_attr(ctx, op, CARRIER_STORAGE_TYPE_KEY)
}

/// Physical aggregate type that a carrier-backed `mir.field_addr` must index.
pub(crate) fn carrier_gep_source_type(ctx: &Context, op: Ptr<Operation>) -> Option<TypeHandle> {
    get_type_attr(ctx, op, CARRIER_GEP_SOURCE_TYPE_KEY)
}

fn collect_operations(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut result = Vec::new();
    let mut pending = vec![root];
    while let Some(operation) = pending.pop() {
        let nested = {
            let op = operation.deref(ctx);
            op.regions()
                .flat_map(|region| region.deref(ctx).iter(ctx))
                .flat_map(|block| block.deref(ctx).iter(ctx))
                .collect::<Vec<_>>()
        };
        pending.extend(nested);
        result.push(operation);
    }
    result
}

fn reject_preexisting_carrier_facts(ctx: &Context, operations: &[Ptr<Operation>]) -> Result<()> {
    for &operation in operations {
        if carrier_storage_type(ctx, operation).is_some()
            || carrier_gep_source_type(ctx, operation).is_some()
        {
            return pliron::input_err_noloc!(
                "packed-AS3 carrier lowering facts must be created by mir-lower preparation, not supplied by input MIR"
            );
        }
    }
    Ok(())
}

/// Recognize exactly the narrow local-storage shape required by #1036.
///
/// The internal device ABI is intentionally broader today. Local storage does
/// not inherit that widening: the root must be a packed struct with exactly one
/// direct AS3 pointer field and all other non-ZST fields scalar. Nested
/// aggregates, arrays, vectors, and multiple shared-pointer leaves remain
/// outside this lane even when the internal ABI can carry them.
fn packed_shared_local_storage_info(
    ctx: &mut Context,
    mir_ty: TypeHandle,
) -> std::result::Result<Option<PackedSharedInternalAbiInfo>, anyhow::Error> {
    let layout = {
        let ty_ref = mir_ty.deref(ctx);
        let Some(struct_ty) = ty_ref.downcast_ref::<MirStructType>() else {
            return Ok(None);
        };
        StructLayoutInfo::of_struct(struct_ty)
    };

    let map = build_struct_slot_map(ctx, &layout)?;
    let mut direct_shared_pointers = 0_u64;
    for field_ty in &map.field_llvm_types {
        if is_zero_sized_type(ctx, *field_ty) {
            continue;
        }
        let field_ref = field_ty.deref(ctx);
        if let Some(pointer) = field_ref.downcast_ref::<llvm_types::PointerType>() {
            if pointer.address_space() == llvm_types::address_space::SHARED {
                direct_shared_pointers += 1;
            }
            continue;
        }
        if field_ref.is::<IntegerType>()
            || field_ref.is::<llvm_types::HalfType>()
            || field_ref.is::<FP32Type>()
            || field_ref.is::<FP64Type>()
        {
            continue;
        }
        return Ok(None);
    }
    if direct_shared_pointers != 1 {
        return Ok(None);
    }

    packed_shared_internal_abi_info(ctx, mir_ty)
}

fn target_stable_local_value_type(
    ctx: &mut Context,
    semantic_mir_ty: TypeHandle,
    role: &str,
) -> std::result::Result<TypeHandle, anyhow::Error> {
    let semantic_llvm_ty = convert_type(ctx, semantic_mir_ty)?;
    Ok(target_stable_storage_type(
        ctx,
        semantic_llvm_ty,
        StorageRewriteOptions {
            canonicalize_bool: false,
        },
        role,
    )?
    .ty)
}

fn seed_carrier_allocas(
    ctx: &mut Context,
    operations: &[Ptr<Operation>],
    addresses: &mut FxHashMap<Value, CarrierAddress>,
    plan: &mut CarrierFactPlan,
) -> Result<()> {
    for &operation in operations {
        if Operation::get_op::<MirAllocaOp>(operation, ctx).is_none() {
            continue;
        }
        let result = operation.deref(ctx).get_result(0);
        let pointee = {
            let result_ty = result.get_type(ctx);
            let result_ref = result_ty.deref(ctx);
            let Some(pointer) = result_ref.downcast_ref::<MirPtrType>() else {
                continue;
            };
            pointer.pointee
        };
        let Some(info) = packed_shared_local_storage_info(ctx, pointee)
            .map_err(|error| pliron::input_error_noloc!("{error}"))?
        else {
            continue;
        };

        plan.plan_storage_type(operation, info.storage_ty);
        addresses.insert(
            result,
            CarrierAddress {
                physical_pointee: info.storage_ty,
                projection_depth: 0,
            },
        );
    }
    Ok(())
}

fn derive_direct_field_projections(
    ctx: &mut Context,
    operations: &[Ptr<Operation>],
    addresses: &mut FxHashMap<Value, CarrierAddress>,
    plan: &mut CarrierFactPlan,
) -> Result<()> {
    // A direct projection depends only on its root alloca. Iterate so malformed
    // nested projections are diagnosed deterministically regardless of block
    // walk order, rather than becoming an unrecognised-use fallback.
    let mut changed = true;
    while changed {
        changed = false;
        for &operation in operations {
            if Operation::get_op::<MirFieldAddrOp>(operation, ctx).is_none() {
                continue;
            }
            let (base, result) = {
                let op = operation.deref(ctx);
                (op.get_operand(0), op.get_result(0))
            };
            if addresses.contains_key(&result) {
                continue;
            }
            let Some(base_state) = addresses.get(&base).copied() else {
                continue;
            };
            if base_state.projection_depth != 0 {
                return pliron::input_err_noloc!(
                    "packed-AS3 carrier locals currently support only one direct field projection; nested carrier projections remain out of scope"
                );
            }

            let semantic_field = {
                let result_ty = result.get_type(ctx);
                let result_ref = result_ty.deref(ctx);
                let pointer = result_ref.downcast_ref::<MirPtrType>().ok_or_else(|| {
                    pliron::input_error_noloc!(
                        "mir.field_addr result must be a MIR pointer before packed-AS3 carrier preparation"
                    )
                })?;
                pointer.pointee
            };
            let physical_field = target_stable_local_value_type(
                ctx,
                semantic_field,
                "packed shared local field projection",
            )
            .map_err(|error| pliron::input_error_noloc!("{error}"))?;

            plan.plan_gep_source_type(operation, base_state.physical_pointee);
            addresses.insert(
                result,
                CarrierAddress {
                    physical_pointee: physical_field,
                    projection_depth: 1,
                },
            );
            changed = true;
        }
    }
    Ok(())
}

fn reject_boundary_use(kind: &str) -> Result<()> {
    pliron::input_err_noloc!(
        "packed-AS3 carrier-local address cannot cross a {}; the physical storage contract must never be dropped and reconstructed later",
        kind
    )
}

fn validate_and_plan_uses(
    ctx: &Context,
    operations: &[Ptr<Operation>],
    addresses: &FxHashMap<Value, CarrierAddress>,
    plan: &mut CarrierFactPlan,
) -> Result<()> {
    for &operation in operations {
        let operands = operation.deref(ctx).operands().collect::<Vec<_>>();
        for (index, operand) in operands.into_iter().enumerate() {
            let Some(state) = addresses.get(&operand).copied() else {
                continue;
            };

            if Operation::get_op::<MirLoadOp>(operation, ctx).is_some() {
                if index != 0 {
                    return pliron::input_err_noloc!(
                        "packed-AS3 carrier-local load used its address in an unexpected operand position"
                    );
                }
                plan.plan_storage_type(operation, state.physical_pointee);
                continue;
            }

            if Operation::get_op::<MirStoreOp>(operation, ctx).is_some() {
                if index != 0 {
                    return pliron::input_err_noloc!(
                        "packed-AS3 carrier-local address cannot itself be stored as a value"
                    );
                }
                plan.plan_storage_type(operation, state.physical_pointee);
                continue;
            }

            if Operation::get_op::<MirFieldAddrOp>(operation, ctx).is_some() {
                if index == 0 && state.projection_depth == 0 {
                    continue;
                }
                return pliron::input_err_noloc!(
                    "packed-AS3 carrier locals currently support only direct root field projections"
                );
            }

            if Operation::get_op::<MirDbgValueOp>(operation, ctx).is_some()
                || Operation::get_op::<MirDbgValueListOp>(operation, ctx).is_some()
            {
                continue;
            }

            if Operation::get_op::<MirCallOp>(operation, ctx).is_some() {
                return reject_boundary_use("call boundary");
            }
            if Operation::get_op::<MirCastOp>(operation, ctx).is_some() {
                return reject_boundary_use("cast");
            }
            if Operation::get_op::<MirGotoOp>(operation, ctx).is_some()
                || Operation::get_op::<MirCondBranchOp>(operation, ctx).is_some()
                || Operation::get_op::<MirAssertOp>(operation, ctx).is_some()
            {
                return reject_boundary_use("block-argument edge");
            }
            if Operation::get_op::<MirReturnOp>(operation, ctx).is_some() {
                return reject_boundary_use("return boundary");
            }
            if Operation::get_op::<MirPtrOffsetOp>(operation, ctx).is_some()
                || Operation::get_op::<MirArrayElementAddrOp>(operation, ctx).is_some()
            {
                return pliron::input_err_noloc!(
                    "packed-AS3 carrier locals do not support pointer arithmetic or array-element projections"
                );
            }

            return pliron::input_err_noloc!(
                "packed-AS3 carrier-local address has an unsupported use by {}; carrier identity would otherwise be lost",
                Operation::get_opid(operation, ctx)
            );
        }
    }
    Ok(())
}

/// Prove and stamp every use of the narrow packed-AS3 local-storage lane.
///
/// This must run after the final MIR-producing transform and immediately before
/// dialect conversion. Its attributes are lowering-private capabilities: input
/// MIR is rejected if it tries to supply one, and every carrier address escape
/// not explicitly modelled here is rejected before LLVM pointer opacity can
/// erase the distinction between semantic and physical pointees. Planning is
/// transactional: no carrier capability is attached until every use validates.
pub(crate) fn prepare_packed_shared_local_storage(
    ctx: &mut Context,
    root: Ptr<Operation>,
) -> Result<()> {
    let operations = collect_operations(ctx, root);
    reject_preexisting_carrier_facts(ctx, &operations)?;

    let mut addresses = FxHashMap::default();
    let mut plan = CarrierFactPlan::default();
    seed_carrier_allocas(ctx, &operations, &mut addresses, &mut plan)?;
    if addresses.is_empty() {
        return Ok(());
    }

    derive_direct_field_projections(ctx, &operations, &mut addresses, &mut plan)?;
    validate_and_plan_uses(ctx, &operations, &addresses, &mut plan)?;
    plan.apply(ctx);
    Ok(())
}

#[cfg(test)]
// Tests build kinded fixture types directly; production minting lives in mir-importer/facts.rs.
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::convert::ops::test_util::*;
    use dialect_mir::attributes::MirCastKindAttr;
    use dialect_mir::ops as mir;
    use llvm_export::op_interfaces::PointerTypeResult;
    use llvm_export::ops as llvm;
    use llvm_export::types::{PointerType, StructType, address_space as llvm_addr};
    use pliron::basic_block::BasicBlock;
    use pliron::builtin::attributes::StringAttr;
    use pliron::builtin::types::{FunctionType, Signedness};
    use pliron::op::Op;

    fn packed_shared_fixture(ctx: &mut Context) -> (TypeHandle, TypeHandle, TypeHandle) {
        let tag: TypeHandle = IntegerType::get(ctx, 8, Signedness::Unsigned).into();
        let pointee: TypeHandle = IntegerType::get(ctx, 32, Signedness::Unsigned).into();
        let shared: TypeHandle = MirPtrType::get(
            ctx,
            pointee,
            true,
            dialect_mir::types::address_space::SHARED,
        )
        .into();
        let packed: TypeHandle = MirStructType::get_with_full_layout(
            ctx,
            "PackedShared".into(),
            vec!["tag".into(), "ptr".into()],
            vec![tag, shared],
            vec![0, 1],
            vec![0, 1],
            9,
            1,
        )
        .into();
        (packed, tag, shared)
    }

    fn append_alloca(ctx: &mut Context, block: Ptr<BasicBlock>, pointee: TypeHandle) -> Value {
        let pointer: TypeHandle = MirPtrType::get_generic(ctx, pointee, true).into();
        let op = Operation::new(
            ctx,
            mir::MirAllocaOp::get_concrete_op_info(),
            vec![pointer],
            vec![],
            vec![],
            0,
        );
        op.insert_at_back(block, ctx);
        op.deref(ctx).get_result(0)
    }

    #[test]
    fn direct_projection_uses_explicit_carrier_facts() {
        let mut ctx = make_ctx();
        let (packed, _tag, shared) = packed_shared_fixture(&mut ctx);
        let (module, block) = build_kernel(&mut ctx, vec![], vec![]);
        let slot = append_alloca(&mut ctx, block, packed);

        let undef = mir::MirUndefOp::new(&mut ctx, packed);
        undef.get_operation().insert_at_back(block, &ctx);
        let whole_value = undef.get_operation().deref(&ctx).get_result(0);
        let whole_store = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, whole_value],
            vec![],
            0,
        );
        whole_store.insert_at_back(block, &ctx);

        let field_ptr_ty: TypeHandle = MirPtrType::get_generic(&mut ctx, shared, true).into();
        let field_addr = mir::MirFieldAddrOp::build(&mut ctx, slot, field_ptr_ty, 1)
            .expect("field address build");
        field_addr.insert_at_back(block, &ctx);
        let field_ptr = field_addr.deref(&ctx).get_result(0);

        let load = Operation::new(
            &mut ctx,
            mir::MirLoadOp::get_concrete_op_info(),
            vec![shared],
            vec![field_ptr],
            vec![],
            0,
        );
        load.insert_at_back(block, &ctx);
        let loaded_shared = load.deref(&ctx).get_result(0);
        let projected_store = Operation::new(
            &mut ctx,
            mir::MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![field_ptr, loaded_shared],
            vec![],
            0,
        );
        projected_store.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        crate::lower_mir_to_llvm(&mut ctx, module).expect("carrier lowering failed");

        let body = kernel_blocks(&ctx, module);
        let alloca = find_first::<llvm::AllocaOp>(&ctx, &body).expect("expected alloca");
        let storage = alloca.result_pointee_type(&ctx);
        let storage_ref = storage.deref(&ctx);
        let storage_struct = storage_ref
            .downcast_ref::<StructType>()
            .expect("carrier local must use struct storage");
        let pointer_ty = storage_struct.field_type(1);
        let pointer_ref = pointer_ty.deref(&ctx);
        let pointer = pointer_ref
            .downcast_ref::<PointerType>()
            .expect("carrier pointer field must lower to LLVM pointer");
        assert_eq!(pointer.address_space(), llvm_addr::GENERIC);
        assert_eq!(
            count_ops::<llvm::AddrSpaceCastOp>(&ctx, &body),
            3,
            "whole-value store, projected load, and projected store must each cross p3/p0 exactly once"
        );
    }

    #[test]
    fn carrier_address_rejects_cast_escape() {
        let mut ctx = make_ctx();
        let (packed, _, _) = packed_shared_fixture(&mut ctx);
        let (module, block) = build_kernel(&mut ctx, vec![], vec![]);
        let slot = append_alloca(&mut ctx, block, packed);
        let pointer_ty = slot.get_type(&ctx);
        let cast = Operation::new(
            &mut ctx,
            mir::MirCastOp::get_concrete_op_info(),
            vec![pointer_ty],
            vec![slot],
            vec![],
            0,
        );
        mir::MirCastOp::new(cast).set_attr_cast_kind(&ctx, MirCastKindAttr::PtrToPtr);
        cast.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        let alloca = slot
            .defining_op()
            .expect("alloca result must have a defining op");
        let error = crate::lower_mir_to_llvm(&mut ctx, module)
            .expect_err("carrier address cast must fail closed in the full lowering pipeline");
        assert!(error.to_string().contains("cast"));
        assert!(
            carrier_storage_type(&ctx, alloca).is_none(),
            "a rejected carrier plan must not leave partial lowering capabilities behind"
        );
    }

    #[test]
    fn carrier_address_rejects_block_argument_escape() {
        let mut ctx = make_ctx();
        let (packed, _, _) = packed_shared_fixture(&mut ctx);
        let (module, block) = build_kernel(&mut ctx, vec![], vec![]);
        let slot = append_alloca(&mut ctx, block, packed);
        let slot_ty = slot.get_type(&ctx);
        let successor = append_block(&mut ctx, block, vec![slot_ty]);
        let goto = Operation::new(
            &mut ctx,
            mir::MirGotoOp::get_concrete_op_info(),
            vec![],
            vec![slot],
            vec![successor],
            0,
        );
        goto.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, successor, vec![]);

        let error = crate::lower_mir_to_llvm(&mut ctx, module).expect_err(
            "carrier address block argument must fail closed in the full lowering pipeline",
        );
        assert!(error.to_string().contains("block-argument"));
    }

    #[test]
    fn carrier_address_rejects_call_escape() {
        let mut ctx = make_ctx();
        let (packed, _, _) = packed_shared_fixture(&mut ctx);
        let (module, block) = build_kernel(&mut ctx, vec![], vec![]);
        let slot = append_alloca(&mut ctx, block, packed);
        let slot_ty = slot.get_type(&ctx);
        let call = Operation::new(
            &mut ctx,
            mir::MirCallOp::get_concrete_op_info(),
            vec![],
            vec![slot],
            vec![],
            0,
        );
        let call_op = mir::MirCallOp::new(call);
        call_op.set_attr_callee(&ctx, StringAttr::new("sink".to_string()));
        let signature = FunctionType::get(&ctx, vec![slot_ty], vec![]);
        call_op.set_external_callee_signature(&mut ctx, signature.into());
        call.insert_at_back(block, &ctx);
        append_mir_return(&mut ctx, block, vec![]);

        let error = crate::lower_mir_to_llvm(&mut ctx, module)
            .expect_err("carrier address call must fail closed in the full lowering pipeline");
        assert!(error.to_string().contains("call boundary"));
    }
}
