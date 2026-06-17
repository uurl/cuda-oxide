/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLVM dialect for cuda-oxide.
//!
//! The dialect *modeling* (types, ops, attributes, op-interfaces) now lives
//! upstream in [`pliron_llvm`]; this crate is a thin shim that re-exports it so
//! existing `llvm_export::{ops,types,attributes,op_interfaces}` paths keep
//! resolving, plus the small set of GPU-specific extensions pliron-llvm does
//! not carry (named address spaces, syncscope enum, fp16 bit helpers). The
//! pure-Rust textual `.ll` exporter ([`export`]) stays here: pliron-llvm only
//! emits real `.ll` via an `llvm-sys` bridge, which is exactly what cuda-oxide
//! is avoiding.
//!
//! Registration is automatic: every dialect/op/type/attribute linked into the
//! binary registers itself when a [`pliron::context::Context`] is created
//! (`Context::default` runs all link-time `CONTEXT_REGISTRATIONS`), so no
//! explicit `register()` entry point is needed.

pub mod export;

/// LLVM types: re-exported from pliron-llvm, plus GPU address-space helpers.
pub mod types {
    pub use pliron_llvm::types::*;

    /// `f16` maps to pliron core's builtin `FP16Type`.
    pub use pliron::builtin::types::FP16Type as HalfType;

    /// NVVM address spaces (generic=0, global=1, shared=3, constant=4,
    /// local=5, tmem=6). pliron-llvm's `PointerType` stores a raw `u32`
    /// address space with no named constants, so we keep these here.
    pub mod address_space {
        /// Generic / flat address space.
        pub const GENERIC: u32 = 0;
        /// Global memory.
        pub const GLOBAL: u32 = 1;
        /// Shared (CTA) memory.
        pub const SHARED: u32 = 3;
        /// Constant memory.
        pub const CONSTANT: u32 = 4;
        /// Thread-local memory.
        pub const LOCAL: u32 = 5;
        /// Tensor memory (Blackwell tcgen05).
        pub const TMEM: u32 = 6;
    }

    use pliron::{context::Context, r#type::TypePtr};
    pub use pliron_llvm::types::PointerType;

    /// Address-space convenience constructors/predicates re-homed from the
    /// pre-migration local `PointerType`. Upstream ships only
    /// `PointerType::get(ctx, address_space)` + `address_space()`.
    pub trait PointerTypeExt {
        /// Pointer into the generic address space.
        fn get_generic(ctx: &mut Context) -> TypePtr<PointerType>;
        /// Pointer into the shared address space.
        fn get_shared(ctx: &mut Context) -> TypePtr<PointerType>;
        /// Pointer into the global address space.
        fn get_global(ctx: &mut Context) -> TypePtr<PointerType>;
        /// Pointer into tensor memory.
        fn get_tmem(ctx: &mut Context) -> TypePtr<PointerType>;
        /// True if this pointer is in the shared address space.
        fn is_shared(&self) -> bool;
        /// True if this pointer is in tensor memory.
        fn is_tmem(&self) -> bool;
    }

    impl PointerTypeExt for PointerType {
        fn get_generic(ctx: &mut Context) -> TypePtr<PointerType> {
            PointerType::get(ctx, address_space::GENERIC)
        }
        fn get_shared(ctx: &mut Context) -> TypePtr<PointerType> {
            PointerType::get(ctx, address_space::SHARED)
        }
        fn get_global(ctx: &mut Context) -> TypePtr<PointerType> {
            PointerType::get(ctx, address_space::GLOBAL)
        }
        fn get_tmem(ctx: &mut Context) -> TypePtr<PointerType> {
            PointerType::get(ctx, address_space::TMEM)
        }
        fn is_shared(&self) -> bool {
            self.address_space() == address_space::SHARED
        }
        fn is_tmem(&self) -> bool {
            self.address_space() == address_space::TMEM
        }
    }
}

/// LLVM attributes: re-exported from pliron-llvm, plus the syncscope enum and
/// the cuda-oxide names for atomic ordering / rmw-kind.
pub mod attributes {
    pub use pliron_llvm::attributes::*;

    /// `f16` constants use pliron core's builtin `FPHalfAttr`.
    pub use pliron::builtin::attributes::FPHalfAttr;

    /// Atomic ordering / rmw-kind were named `Llvm*` locally; upstream calls
    /// them `Atomic*Attr`. Keep the local names resolving.
    pub use pliron_llvm::attributes::{
        AtomicOrderingAttr as LlvmAtomicOrdering, AtomicRmwKindAttr as LlvmAtomicRmwKind,
    };

    /// Synchronization scope for atomics. pliron-llvm models syncscope as a
    /// free-form `Option<String>` (None = system); cuda-oxide only emits these
    /// three scopes, so we keep the enum at the lowering boundary and translate
    /// to pliron's representation via [`LlvmSyncScope::to_pliron`].
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum LlvmSyncScope {
        /// System-wide scope (`syncscope("")` / default).
        System,
        /// Device (GPU) scope.
        Device,
        /// Block / CTA scope.
        Block,
    }

    impl LlvmSyncScope {
        /// Map to pliron's free-form syncscope string (`None` = system).
        pub fn to_pliron(self) -> Option<String> {
            match self {
                LlvmSyncScope::System => None,
                LlvmSyncScope::Device => Some("device".to_string()),
                LlvmSyncScope::Block => Some("block".to_string()),
            }
        }
    }
}

/// LLVM ops: re-exported from pliron-llvm, plus the builtin `ConstantOp` and a
/// convergent inline-asm constructor.
pub mod ops {
    pub use pliron_llvm::ops::*;

    use std::path::PathBuf;

    use combine::stream::position::SourcePosition;

    /// `ConstantOp` moved from the LLVM dialect to pliron core `builtin`.
    pub use pliron::builtin::ops::ConstantOp;

    use pliron::{
        builtin::{
            attributes::{BoolAttr, StringAttr},
            op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface},
        },
        common_traits::Verify,
        context::{Context, Ptr},
        identifier::Identifier,
        op::Op,
        operation::Operation,
        result::Error,
        r#type::TypeObj,
        value::Value,
    };
    use pliron_derive::pliron_op;
    use pliron_llvm::attributes::AlignmentAttr;
    pub use pliron_llvm::ops::{GlobalOp, InlineAsmOp};

    /// Convergent inline-asm constructor re-homed from the pre-migration local
    /// op. Upstream `InlineAsmOp::new` takes a trailing `convergent: bool`;
    /// this keeps the `new_convergent(...)` call shape used across mir-lower.
    pub trait InlineAsmOpExt {
        /// Build a convergent `InlineAsmOp` (use a void result type for asm
        /// with no result value).
        fn new_convergent(
            ctx: &mut Context,
            result_ty: Ptr<TypeObj>,
            inputs: Vec<Value>,
            asm_template: &str,
            constraints: &str,
        ) -> Self;
    }

    impl InlineAsmOpExt for InlineAsmOp {
        fn new_convergent(
            ctx: &mut Context,
            result_ty: Ptr<TypeObj>,
            inputs: Vec<Value>,
            asm_template: &str,
            constraints: &str,
        ) -> Self {
            InlineAsmOp::new(ctx, result_ty, inputs, asm_template, constraints, true)
        }
    }

    /// Op-attribute key for a `GlobalOp`'s explicit alignment.
    const GLOBAL_ALIGNMENT_KEY: &str = "cuda_oxide_global_alignment";

    /// Op-attribute key under which a memory op's (`load` / `store` / `alloca`)
    /// explicit ABI alignment is stashed. Stamped by the mir-lower alignment
    /// pre-pass (while types are still MIR, so `repr(align(N))` is visible)
    /// and emitted as `align N` during export.
    const OP_ALIGNMENT_KEY: &str = "cuda_oxide_op_alignment";

    /// Op-attribute key controlling whether an inline asm op is emitted with
    /// LLVM's `sideeffect` marker. Absent means true, matching the conservative
    /// default for user-authored inline PTX.
    const INLINE_ASM_SIDEEFFECT_KEY: &str = "cuda_oxide_inline_asm_sideeffect";

    /// Debug type metadata for a local variable described by `llvm.dbg.declare`.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub enum DebugLocalTypeKind {
        /// A scalar `DIBasicType`.
        Basic {
            name: String,
            size_bits: u64,
            encoding: &'static str,
        },
        /// A pointer/reference `DIDerivedType`.
        Pointer { name: String, size_bits: u64 },
    }

    /// Debug metadata attached to the alloca that stores a source local.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugLocalVariableInfo {
        pub name: String,
        pub argument_index: Option<u16>,
        pub ty: DebugLocalTypeKind,
    }

    /// A source position small enough to carry through cuda-oxide attrs.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugSourcePosition {
        pub file: PathBuf,
        pub line: i32,
        pub column: i32,
    }

    /// Extra scope information rustc records for MIR inlining.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugInlinedScope {
        pub callee_name: String,
        pub callsite: Option<DebugSourcePosition>,
    }

    /// One rustc MIR `SourceScope`, flattened into stable data.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugSourceScope {
        pub id: u32,
        pub parent: Option<u32>,
        pub span: Option<DebugSourcePosition>,
        pub inlined: Option<DebugInlinedScope>,
    }

    /// The original rustc MIR source scope for a statement or terminator span.
    ///
    /// stable MIR currently exposes the span, but not the `SourceScope`, on
    /// statements and terminators. The rustc-codegen bridge records that
    /// pairing before the stable-MIR conversion so instruction `!dbg` scopes
    /// can match the lexical scopes used by local variables.
    #[derive(Clone, Debug, Eq, Hash, PartialEq)]
    pub struct DebugSourceScopeLocation {
        pub pos: DebugSourcePosition,
        pub scope: u32,
    }

    /// The source-scope table for one function body.
    #[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
    pub struct DebugSourceScopeMap {
        pub scopes: Vec<DebugSourceScope>,
        pub locations: Vec<DebugSourceScopeLocation>,
    }

    const DEBUG_LOCAL_NAME_KEY: &str = "cuda_oxide_debug_local_name";
    const DEBUG_LOCAL_ARG_KEY: &str = "cuda_oxide_debug_local_arg";
    const DEBUG_LOCAL_TYPE_KIND_KEY: &str = "cuda_oxide_debug_local_type_kind";
    const DEBUG_LOCAL_TYPE_NAME_KEY: &str = "cuda_oxide_debug_local_type_name";
    const DEBUG_LOCAL_TYPE_SIZE_KEY: &str = "cuda_oxide_debug_local_type_size_bits";
    const DEBUG_LOCAL_TYPE_ENCODING_KEY: &str = "cuda_oxide_debug_local_type_encoding";
    const DEBUG_LOCAL_DECL_FILE_KEY: &str = "cuda_oxide_debug_local_decl_file";
    const DEBUG_LOCAL_DECL_LINE_KEY: &str = "cuda_oxide_debug_local_decl_line";
    const DEBUG_LOCAL_DECL_COLUMN_KEY: &str = "cuda_oxide_debug_local_decl_column";
    const DEBUG_LOCAL_SCOPE_KEY: &str = "cuda_oxide_debug_local_scope";
    const DEBUG_SOURCE_SCOPE_COUNT_KEY: &str = "cuda_oxide_debug_scope_count";
    const DEBUG_SOURCE_SCOPE_LOCATION_COUNT_KEY: &str = "cuda_oxide_debug_scope_location_count";

    /// Stamp the ABI alignment (bytes) onto a memory op.
    pub fn set_op_alignment(ctx: &mut Context, op: Ptr<Operation>, align: u32) {
        let key = Identifier::try_new(OP_ALIGNMENT_KEY.to_string()).expect("valid identifier");
        op.deref_mut(ctx).attributes.set(key, AlignmentAttr(align));
    }

    /// Read the ABI alignment (bytes) stamped on a memory op, if any.
    pub fn op_alignment(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
        let key = Identifier::try_new(OP_ALIGNMENT_KEY.to_string()).expect("valid identifier");
        op.deref(ctx)
            .attributes
            .get::<AlignmentAttr>(&key)
            .map(|a| a.0)
    }

    /// Stamp whether an inline asm op has side effects beyond its operands.
    pub fn set_inline_asm_sideeffect(ctx: &mut Context, op: Ptr<Operation>, sideeffect: bool) {
        let key =
            Identifier::try_new(INLINE_ASM_SIDEEFFECT_KEY.to_string()).expect("valid identifier");
        op.deref_mut(ctx)
            .attributes
            .set(key, BoolAttr::new(sideeffect));
    }

    /// Read whether an inline asm op should be emitted with `sideeffect`.
    pub fn inline_asm_sideeffect(ctx: &Context, op: Ptr<Operation>) -> bool {
        let key =
            Identifier::try_new(INLINE_ASM_SIDEEFFECT_KEY.to_string()).expect("valid identifier");
        op.deref(ctx)
            .attributes
            .get::<BoolAttr>(&key)
            .map(|a| bool::from((*a).clone()))
            .unwrap_or(true)
    }

    /// Attach source-local debug metadata to a memory slot op.
    pub fn set_debug_local_variable(
        ctx: &mut Context,
        op: Ptr<Operation>,
        info: DebugLocalVariableInfo,
    ) {
        set_string_attr(ctx, op, DEBUG_LOCAL_NAME_KEY, info.name);
        if let Some(arg) = info.argument_index {
            set_string_attr(ctx, op, DEBUG_LOCAL_ARG_KEY, arg.to_string());
        }

        match info.ty {
            DebugLocalTypeKind::Basic {
                name,
                size_bits,
                encoding,
            } => {
                set_string_attr(ctx, op, DEBUG_LOCAL_TYPE_KIND_KEY, "basic".to_string());
                set_string_attr(ctx, op, DEBUG_LOCAL_TYPE_NAME_KEY, name);
                set_string_attr(ctx, op, DEBUG_LOCAL_TYPE_SIZE_KEY, size_bits.to_string());
                set_string_attr(ctx, op, DEBUG_LOCAL_TYPE_ENCODING_KEY, encoding.to_string());
            }
            DebugLocalTypeKind::Pointer { name, size_bits } => {
                set_string_attr(ctx, op, DEBUG_LOCAL_TYPE_KIND_KEY, "pointer".to_string());
                set_string_attr(ctx, op, DEBUG_LOCAL_TYPE_NAME_KEY, name);
                set_string_attr(ctx, op, DEBUG_LOCAL_TYPE_SIZE_KEY, size_bits.to_string());
            }
        }
    }

    /// Read source-local debug metadata from a memory slot op, if present.
    pub fn debug_local_variable(
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Option<DebugLocalVariableInfo> {
        let name = get_string_attr(ctx, op, DEBUG_LOCAL_NAME_KEY)?;
        let argument_index =
            get_string_attr(ctx, op, DEBUG_LOCAL_ARG_KEY).and_then(|arg| arg.parse::<u16>().ok());
        let kind = get_string_attr(ctx, op, DEBUG_LOCAL_TYPE_KIND_KEY)?;
        let type_name = get_string_attr(ctx, op, DEBUG_LOCAL_TYPE_NAME_KEY)?;
        let size_bits = get_string_attr(ctx, op, DEBUG_LOCAL_TYPE_SIZE_KEY)?
            .parse()
            .ok()?;

        let ty = match kind.as_str() {
            "basic" => DebugLocalTypeKind::Basic {
                name: type_name,
                size_bits,
                encoding: debug_type_encoding(ctx, op)?,
            },
            "pointer" => DebugLocalTypeKind::Pointer {
                name: type_name,
                size_bits,
            },
            _ => return None,
        };

        Some(DebugLocalVariableInfo {
            name,
            argument_index,
            ty,
        })
    }

    /// Attach the MIR source-scope id that owns this source local.
    pub fn set_debug_local_source_scope(ctx: &mut Context, op: Ptr<Operation>, scope: u32) {
        set_string_attr(ctx, op, DEBUG_LOCAL_SCOPE_KEY, scope.to_string());
    }

    /// Read the MIR source-scope id that owns this source local.
    pub fn debug_local_source_scope(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
        get_string_attr(ctx, op, DEBUG_LOCAL_SCOPE_KEY).and_then(|scope| scope.parse().ok())
    }

    /// Attach a function's MIR source-scope table.
    pub fn set_debug_source_scope_map(
        ctx: &mut Context,
        op: Ptr<Operation>,
        map: &DebugSourceScopeMap,
    ) {
        // The reader (`debug_source_scope_map`) reconstructs scope ids as
        // `0..count`, so the writer's per-scope attr keys must use exactly those
        // ids. rustc's `SourceScope` indices are dense `0..len`, which makes this
        // hold today. Assert it so a future sparse/reordered producer fails
        // loudly here instead of silently mislabeling parent/scope links.
        debug_assert!(
            map.scopes
                .iter()
                .enumerate()
                .all(|(idx, scope)| scope.id as usize == idx),
            "DebugSourceScopeMap scope ids must be dense 0..len to round-trip"
        );
        set_string_attr(
            ctx,
            op,
            DEBUG_SOURCE_SCOPE_COUNT_KEY,
            map.scopes.len().to_string(),
        );
        set_string_attr(
            ctx,
            op,
            DEBUG_SOURCE_SCOPE_LOCATION_COUNT_KEY,
            map.locations.len().to_string(),
        );

        for scope in &map.scopes {
            let id = scope.id;
            if let Some(parent) = scope.parent {
                set_string_attr(ctx, op, &debug_scope_key(id, "parent"), parent.to_string());
            }
            if let Some(span) = &scope.span {
                set_debug_position_attrs(ctx, op, id, "span", span);
            }
            if let Some(inlined) = &scope.inlined {
                set_string_attr(
                    ctx,
                    op,
                    &debug_scope_key(id, "callee"),
                    inlined.callee_name.clone(),
                );
                if let Some(callsite) = &inlined.callsite {
                    set_debug_position_attrs(ctx, op, id, "callsite", callsite);
                }
            }
        }

        for (idx, location) in map.locations.iter().enumerate() {
            set_string_attr(
                ctx,
                op,
                &debug_scope_location_key(idx, "scope"),
                location.scope.to_string(),
            );
            set_debug_scope_location_position_attrs(ctx, op, idx, &location.pos);
        }
    }

    /// Read a function's MIR source-scope table.
    pub fn debug_source_scope_map(
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Option<DebugSourceScopeMap> {
        let count = get_string_attr(ctx, op, DEBUG_SOURCE_SCOPE_COUNT_KEY)?
            .parse()
            .ok()?;
        let mut scopes = Vec::with_capacity(count);

        for id in 0..count as u32 {
            let parent = get_string_attr(ctx, op, &debug_scope_key(id, "parent"))
                .and_then(|v| v.parse().ok());
            let span = debug_position_attrs(ctx, op, id, "span");
            let inlined = get_string_attr(ctx, op, &debug_scope_key(id, "callee")).map(|name| {
                DebugInlinedScope {
                    callee_name: name,
                    callsite: debug_position_attrs(ctx, op, id, "callsite"),
                }
            });
            scopes.push(DebugSourceScope {
                id,
                parent,
                span,
                inlined,
            });
        }

        let location_count = get_string_attr(ctx, op, DEBUG_SOURCE_SCOPE_LOCATION_COUNT_KEY)
            .and_then(|count| count.parse().ok())
            .unwrap_or(0);
        let mut locations = Vec::with_capacity(location_count);

        for idx in 0..location_count {
            let scope = get_string_attr(ctx, op, &debug_scope_location_key(idx, "scope"))
                .and_then(|v| v.parse().ok())?;
            let pos = debug_scope_location_position_attrs(ctx, op, idx)?;
            locations.push(DebugSourceScopeLocation { pos, scope });
        }

        Some(DebugSourceScopeMap { scopes, locations })
    }

    /// Copy debug source-scope attrs from one operation to another.
    pub fn copy_debug_source_scope_map(
        ctx: &mut Context,
        from: Ptr<Operation>,
        to: Ptr<Operation>,
    ) {
        let Some(map) = debug_source_scope_map(ctx, from) else {
            return;
        };
        set_debug_source_scope_map(ctx, to, &map);
    }

    /// Read an optional source declaration location for a debug local.
    ///
    /// Promoted `dbg.value` records have two useful locations: the operation
    /// location where the value is current, and the source declaration location
    /// for the `DILocalVariable`. This helper returns the latter when it was
    /// preserved during MIR mem2reg promotion.
    pub fn debug_local_declaration_location(
        ctx: &Context,
        op: Ptr<Operation>,
    ) -> Option<(PathBuf, SourcePosition)> {
        let file = PathBuf::from(get_string_attr(ctx, op, DEBUG_LOCAL_DECL_FILE_KEY)?);
        let line = get_string_attr(ctx, op, DEBUG_LOCAL_DECL_LINE_KEY)?
            .parse()
            .ok()?;
        let column = get_string_attr(ctx, op, DEBUG_LOCAL_DECL_COLUMN_KEY)?
            .parse()
            .ok()?;
        if line <= 0 || column <= 0 {
            return None;
        }

        Some((file, SourcePosition { line, column }))
    }

    /// Attach the source declaration location for a debug local.
    pub fn set_debug_local_declaration_location(
        ctx: &mut Context,
        op: Ptr<Operation>,
        file: PathBuf,
        line: i32,
        column: i32,
    ) {
        set_string_attr(
            ctx,
            op,
            DEBUG_LOCAL_DECL_FILE_KEY,
            file.to_string_lossy().into_owned(),
        );
        set_string_attr(ctx, op, DEBUG_LOCAL_DECL_LINE_KEY, line.to_string());
        set_string_attr(ctx, op, DEBUG_LOCAL_DECL_COLUMN_KEY, column.to_string());
    }

    fn set_debug_position_attrs(
        ctx: &mut Context,
        op: Ptr<Operation>,
        scope: u32,
        prefix: &str,
        pos: &DebugSourcePosition,
    ) {
        set_string_attr(
            ctx,
            op,
            &debug_scope_key(scope, &format!("{prefix}_file")),
            pos.file.to_string_lossy().into_owned(),
        );
        set_string_attr(
            ctx,
            op,
            &debug_scope_key(scope, &format!("{prefix}_line")),
            pos.line.to_string(),
        );
        set_string_attr(
            ctx,
            op,
            &debug_scope_key(scope, &format!("{prefix}_column")),
            pos.column.to_string(),
        );
    }

    fn debug_position_attrs(
        ctx: &Context,
        op: Ptr<Operation>,
        scope: u32,
        prefix: &str,
    ) -> Option<DebugSourcePosition> {
        let file = PathBuf::from(get_string_attr(
            ctx,
            op,
            &debug_scope_key(scope, &format!("{prefix}_file")),
        )?);
        let line = get_string_attr(ctx, op, &debug_scope_key(scope, &format!("{prefix}_line")))?
            .parse()
            .ok()?;
        let column = get_string_attr(
            ctx,
            op,
            &debug_scope_key(scope, &format!("{prefix}_column")),
        )?
        .parse()
        .ok()?;
        if line <= 0 || column <= 0 {
            return None;
        }

        Some(DebugSourcePosition { file, line, column })
    }

    fn set_debug_scope_location_position_attrs(
        ctx: &mut Context,
        op: Ptr<Operation>,
        idx: usize,
        pos: &DebugSourcePosition,
    ) {
        set_string_attr(
            ctx,
            op,
            &debug_scope_location_key(idx, "file"),
            pos.file.to_string_lossy().into_owned(),
        );
        set_string_attr(
            ctx,
            op,
            &debug_scope_location_key(idx, "line"),
            pos.line.to_string(),
        );
        set_string_attr(
            ctx,
            op,
            &debug_scope_location_key(idx, "column"),
            pos.column.to_string(),
        );
    }

    fn debug_scope_location_position_attrs(
        ctx: &Context,
        op: Ptr<Operation>,
        idx: usize,
    ) -> Option<DebugSourcePosition> {
        let file = PathBuf::from(get_string_attr(
            ctx,
            op,
            &debug_scope_location_key(idx, "file"),
        )?);
        let line = get_string_attr(ctx, op, &debug_scope_location_key(idx, "line"))?
            .parse()
            .ok()?;
        let column = get_string_attr(ctx, op, &debug_scope_location_key(idx, "column"))?
            .parse()
            .ok()?;
        if line <= 0 || column <= 0 {
            return None;
        }

        Some(DebugSourcePosition { file, line, column })
    }

    fn debug_scope_key(scope: u32, field: &str) -> String {
        format!("cuda_oxide_debug_scope_{scope}_{field}")
    }

    fn debug_scope_location_key(idx: usize, field: &str) -> String {
        format!("cuda_oxide_debug_scope_location_{idx}_{field}")
    }

    fn set_string_attr(ctx: &mut Context, op: Ptr<Operation>, key: &str, value: String) {
        let key = Identifier::try_new(key.to_string()).expect("valid identifier");
        op.deref_mut(ctx)
            .attributes
            .set(key, StringAttr::new(value));
    }

    fn get_string_attr(ctx: &Context, op: Ptr<Operation>, key: &str) -> Option<String> {
        let key = Identifier::try_new(key.to_string()).expect("valid identifier");
        op.deref(ctx)
            .attributes
            .get::<StringAttr>(&key)
            .map(|a| String::from((*a).clone()))
    }

    fn debug_type_encoding(ctx: &Context, op: Ptr<Operation>) -> Option<&'static str> {
        match get_string_attr(ctx, op, DEBUG_LOCAL_TYPE_ENCODING_KEY)?.as_str() {
            "DW_ATE_boolean" => Some("DW_ATE_boolean"),
            "DW_ATE_float" => Some("DW_ATE_float"),
            "DW_ATE_signed" => Some("DW_ATE_signed"),
            "DW_ATE_unsigned" => Some("DW_ATE_unsigned"),
            _ => None,
        }
    }

    /// LLVM debug-value marker used by the textual exporter.
    ///
    /// This is not a runtime instruction. It lowers to an `llvm.dbg.value`
    /// intrinsic call that tells LLVM/DWARF where a source local lives after a
    /// MIR stack slot has been promoted to an SSA value.
    #[pliron_op(
        name = "llvm.dbg_value",
        format,
        interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<0>]
    )]
    pub struct DebugValueOp;

    impl DebugValueOp {
        pub fn new(ctx: &mut Context, value: Value) -> Self {
            let op = Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![value],
                vec![],
                0,
            );
            DebugValueOp { op }
        }

        pub fn value(&self, ctx: &Context) -> Value {
            self.get_operation().deref(ctx).get_operand(0)
        }
    }

    impl Verify for DebugValueOp {
        fn verify(&self, _ctx: &Context) -> Result<(), Error> {
            Ok(())
        }
    }

    /// Alignment helpers re-homed from the pre-migration local `GlobalOp`.
    /// Upstream `GlobalOp` carries type/linkage/addrspace but no alignment, so
    /// we keep the alignment in the op's generic attribute dictionary. Address
    /// space uses upstream's native `address_space` / `set_address_space`.
    pub trait GlobalOpExt {
        /// Build a `GlobalOp` carrying an explicit alignment (bytes).
        fn new_with_alignment(
            ctx: &mut Context,
            name: Identifier,
            ty: Ptr<TypeObj>,
            alignment: u64,
        ) -> Self;
        /// Read the explicit alignment (bytes), if one was set.
        fn get_alignment(&self, ctx: &Context) -> Option<u64>;
    }

    impl GlobalOpExt for GlobalOp {
        fn new_with_alignment(
            ctx: &mut Context,
            name: Identifier,
            ty: Ptr<TypeObj>,
            alignment: u64,
        ) -> Self {
            let op = GlobalOp::new(ctx, name, ty);
            let key =
                Identifier::try_new(GLOBAL_ALIGNMENT_KEY.to_string()).expect("valid identifier");
            op.get_operation()
                .deref_mut(ctx)
                .attributes
                .set(key, AlignmentAttr(alignment as u32));
            op
        }

        fn get_alignment(&self, ctx: &Context) -> Option<u64> {
            let key =
                Identifier::try_new(GLOBAL_ALIGNMENT_KEY.to_string()).expect("valid identifier");
            self.get_operation()
                .deref(ctx)
                .attributes
                .get::<AlignmentAttr>(&key)
                .map(|a| a.0 as u64)
        }
    }
}

/// LLVM op-interfaces, re-exported from pliron-llvm.
pub mod op_interfaces {
    pub use pliron_llvm::op_interfaces::*;
}

use pliron::builtin::attributes::FPHalfAttr;
use pliron::utils::apfloat::{Float, Half};

/// Build an `FPHalfAttr` from a raw 16-bit IEEE half pattern. pliron's
/// `FPHalfAttr` wraps `apfloat::Half`, whose bit access is `u128`-wide via the
/// `Float` trait, so we widen here.
pub fn fp16_attr_from_bits(bits: u16) -> FPHalfAttr {
    FPHalfAttr(Half::from_bits(bits as u128))
}

/// Extract the raw 16-bit IEEE half pattern from an `FPHalfAttr`.
pub fn fp16_attr_to_bits(attr: &FPHalfAttr) -> u16 {
    attr.0.to_bits() as u16
}
