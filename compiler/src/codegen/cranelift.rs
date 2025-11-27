//! Cranelift JIT backend
//!
//! This module provides fast JIT compilation using Cranelift.
//! Cranelift is optimized for fast compilation rather than peak runtime performance,
//! making it ideal for development and scripting use cases.

use crate::hlir::HlirModule;

#[cfg(feature = "jit")]
use crate::hlir::{
    BinaryOp, BlockId, HlirConstant, HlirFunction, HlirTerminator, HlirType, Op, UnaryOp, ValueId,
};
use std::collections::HashMap;

#[cfg(feature = "jit")]
use cranelift_codegen::Context;
#[cfg(feature = "jit")]
use cranelift_codegen::ir::{
    AbiParam, Function, InstBuilder, MemFlags, Signature, UserFuncName, types,
};
#[cfg(feature = "jit")]
use cranelift_codegen::isa::CallConv;
#[cfg(feature = "jit")]
use cranelift_codegen::settings::{self, Configurable};
#[cfg(feature = "jit")]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
#[cfg(feature = "jit")]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(feature = "jit")]
use cranelift_module::{DataDescription, Linkage, Module};

/// Cranelift JIT compiler
pub struct CraneliftJit {
    /// Whether to enable optimization
    optimize: bool,
}

impl CraneliftJit {
    pub fn new() -> Self {
        Self { optimize: false }
    }

    pub fn with_optimization(mut self) -> Self {
        self.optimize = true;
        self
    }

    /// Compile and immediately run the module, returning the result of main()
    #[cfg(feature = "jit")]
    pub fn compile_and_run(&self, module: &HlirModule) -> Result<i64, String> {
        let compiled = self.compile(module)?;
        unsafe { compiled.call_i64("main") }
    }

    #[cfg(not(feature = "jit"))]
    pub fn compile_and_run(&self, _module: &HlirModule) -> Result<i64, String> {
        Err("JIT backend not enabled. Compile with --features jit".to_string())
    }

    /// Compile the module and return a handle to the compiled code
    #[cfg(feature = "jit")]
    pub fn compile(&self, module: &HlirModule) -> Result<CompiledModule, String> {
        let mut compiler = JitCompiler::new(self.optimize)?;
        compiler.compile_module(module)?;
        compiler.finalize()
    }

    #[cfg(not(feature = "jit"))]
    pub fn compile(&self, _module: &HlirModule) -> Result<CompiledModule, String> {
        Err("JIT backend not enabled. Compile with --features jit".to_string())
    }
}

impl Default for CraneliftJit {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle to compiled JIT code
pub struct CompiledModule {
    #[cfg(feature = "jit")]
    jit_module: JITModule,
    /// Function entry points
    functions: HashMap<String, *const u8>,
}

// SAFETY: The JIT module owns the compiled code and manages its lifetime
unsafe impl Send for CompiledModule {}
unsafe impl Sync for CompiledModule {}

impl CompiledModule {
    /// Get a function pointer by name
    pub fn get_function(&self, name: &str) -> Option<*const u8> {
        self.functions.get(name).copied()
    }

    /// Call a function with no arguments returning i64
    ///
    /// # Safety
    /// The caller must ensure the function signature matches.
    pub unsafe fn call_i64(&self, name: &str) -> Result<i64, String> {
        let ptr = self
            .get_function(name)
            .ok_or_else(|| format!("Function not found: {}", name))?;

        let func: extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
        Ok(func())
    }

    /// Call a function with one i64 argument returning i64
    ///
    /// # Safety
    /// The caller must ensure the function signature matches.
    pub unsafe fn call_i64_i64(&self, name: &str, arg: i64) -> Result<i64, String> {
        let ptr = self
            .get_function(name)
            .ok_or_else(|| format!("Function not found: {}", name))?;

        let func: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        Ok(func(arg))
    }

    /// Call a function with two i64 arguments returning i64
    ///
    /// # Safety
    /// The caller must ensure the function signature matches.
    pub unsafe fn call_i64_i64_i64(&self, name: &str, a: i64, b: i64) -> Result<i64, String> {
        let ptr = self
            .get_function(name)
            .ok_or_else(|| format!("Function not found: {}", name))?;

        let func: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        Ok(func(a, b))
    }
}

/// JIT compilation settings
pub struct JitSettings {
    /// Enable basic optimizations
    pub optimize: bool,
    /// Enable bounds checking
    pub bounds_check: bool,
    /// Enable overflow checking
    pub overflow_check: bool,
    /// Stack size in bytes
    pub stack_size: usize,
}

impl Default for JitSettings {
    fn default() -> Self {
        Self {
            optimize: false,
            bounds_check: true,
            overflow_check: true,
            stack_size: 1024 * 1024, // 1 MB
        }
    }
}

impl JitSettings {
    pub fn release() -> Self {
        Self {
            optimize: true,
            bounds_check: false,
            overflow_check: false,
            stack_size: 8 * 1024 * 1024, // 8 MB
        }
    }
}

// ==================== JIT Compiler Implementation ====================

#[cfg(feature = "jit")]
struct JitCompiler {
    jit_module: JITModule,
    ctx: Context,
    func_ctx: FunctionBuilderContext,
    /// Map from HLIR function names to Cranelift function IDs
    func_ids: HashMap<String, cranelift_module::FuncId>,
}

#[cfg(feature = "jit")]
impl JitCompiler {
    fn new(optimize: bool) -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();

        if optimize {
            flag_builder.set("opt_level", "speed").unwrap();
        } else {
            flag_builder.set("opt_level", "none").unwrap();
        }

        let isa_builder = cranelift_native::builder()
            .map_err(|e| format!("Failed to create ISA builder: {}", e))?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| format!("Failed to create ISA: {}", e))?;

        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        let jit_module = JITModule::new(jit_builder);
        let ctx = jit_module.make_context();

        Ok(Self {
            jit_module,
            ctx,
            func_ctx: FunctionBuilderContext::new(),
            func_ids: HashMap::new(),
        })
    }

    fn compile_module(&mut self, module: &HlirModule) -> Result<(), String> {
        // First pass: declare all functions
        for func in &module.functions {
            let sig = self.create_signature(func);
            let func_id = self
                .jit_module
                .declare_function(&func.name, Linkage::Export, &sig)
                .map_err(|e| format!("Failed to declare function {}: {}", func.name, e))?;
            self.func_ids.insert(func.name.clone(), func_id);
        }

        // Second pass: compile all functions
        for func in &module.functions {
            self.compile_function(func)?;
        }

        Ok(())
    }

    fn create_signature(&self, func: &HlirFunction) -> Signature {
        let call_conv = self.jit_module.isa().default_call_conv();
        let mut sig = Signature::new(call_conv);

        for param in &func.params {
            sig.params
                .push(AbiParam::new(self.hlir_to_cranelift_type(&param.ty)));
        }

        if func.return_type != HlirType::Void {
            sig.returns.push(AbiParam::new(
                self.hlir_to_cranelift_type(&func.return_type),
            ));
        }

        sig
    }

    fn hlir_to_cranelift_type(&self, ty: &HlirType) -> types::Type {
        match ty {
            HlirType::Void => types::I64, // Use I64 for void to avoid issues
            HlirType::Bool => types::I8,
            HlirType::I8 | HlirType::U8 => types::I8,
            HlirType::I16 | HlirType::U16 => types::I16,
            HlirType::I32 | HlirType::U32 => types::I32,
            HlirType::I64 | HlirType::U64 => types::I64,
            HlirType::I128 | HlirType::U128 => types::I128,
            HlirType::F32 => types::F32,
            HlirType::F64 => types::F64,
            HlirType::Ptr(_) => types::I64,
            HlirType::Array(_, _) => types::I64, // Pointer to array
            HlirType::Struct(_) => types::I64,   // Pointer to struct
            HlirType::Tuple(_) => types::I64,    // Pointer to tuple or packed
            HlirType::Function { .. } => types::I64, // Function pointer
        }
    }

    fn compile_function(&mut self, func: &HlirFunction) -> Result<(), String> {
        let func_id = self.func_ids[&func.name];

        // Create function signature
        self.ctx.func.signature = self.create_signature(func);
        self.ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        // Build function body
        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);
            let mut translator = FunctionTranslator::new(&mut builder, &self.func_ids, func);
            translator.translate(func)?;
            builder.finalize();
        }

        // Compile the function
        self.jit_module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| format!("Failed to define function {}: {}", func.name, e))?;

        self.jit_module.clear_context(&mut self.ctx);

        Ok(())
    }

    fn finalize(mut self) -> Result<CompiledModule, String> {
        self.jit_module
            .finalize_definitions()
            .map_err(|e| format!("Failed to finalize: {}", e))?;

        let mut functions = HashMap::new();
        for (name, func_id) in &self.func_ids {
            let ptr = self.jit_module.get_finalized_function(*func_id);
            functions.insert(name.clone(), ptr);
        }

        Ok(CompiledModule {
            jit_module: self.jit_module,
            functions,
        })
    }
}

#[cfg(feature = "jit")]
struct FunctionTranslator<'a> {
    builder: &'a mut FunctionBuilder<'a>,
    func_ids: &'a HashMap<String, cranelift_module::FuncId>,
    /// Map from HLIR ValueId to Cranelift Value
    values: HashMap<ValueId, cranelift_codegen::ir::Value>,
    /// Map from HLIR BlockId to Cranelift Block
    blocks: HashMap<BlockId, cranelift_codegen::ir::Block>,
    /// Variables for mutable locals
    variables: HashMap<ValueId, Variable>,
    next_var: usize,
    /// The HLIR function being compiled
    hlir_func: &'a HlirFunction,
}

#[cfg(feature = "jit")]
impl<'a> FunctionTranslator<'a> {
    fn new(
        builder: &'a mut FunctionBuilder<'a>,
        func_ids: &'a HashMap<String, cranelift_module::FuncId>,
        hlir_func: &'a HlirFunction,
    ) -> Self {
        Self {
            builder,
            func_ids,
            values: HashMap::new(),
            blocks: HashMap::new(),
            variables: HashMap::new(),
            next_var: 0,
            hlir_func,
        }
    }

    fn translate(&mut self, func: &HlirFunction) -> Result<(), String> {
        // Create all blocks first
        for block in &func.blocks {
            let cl_block = self.builder.create_block();
            self.blocks.insert(block.id, cl_block);
        }

        // Entry block parameters (function arguments)
        if let Some(entry) = func.blocks.first() {
            let entry_block = self.blocks[&entry.id];
            self.builder.switch_to_block(entry_block);
            self.builder.seal_block(entry_block);

            // Add function parameters
            for (i, param) in func.params.iter().enumerate() {
                let ty = self.hlir_to_type(&param.ty);
                let val = self.builder.append_block_param(entry_block, ty);
                self.values.insert(param.value, val);
            }
        }

        // Translate each block
        for block in &func.blocks {
            self.translate_block(block)?;
        }

        Ok(())
    }

    fn translate_block(&mut self, block: &HlirBlock) -> Result<(), String> {
        let cl_block = self.blocks[&block.id];

        // Only switch if not already on this block
        if self.builder.current_block() != Some(cl_block) {
            self.builder.switch_to_block(cl_block);
        }

        // Translate instructions
        for instr in &block.instructions {
            let result = self.translate_instruction(instr)?;
            if let (Some(res_id), Some(val)) = (instr.result, result) {
                self.values.insert(res_id, val);
            }
        }

        // Translate terminator
        self.translate_terminator(&block.terminator)?;

        // Seal the block if all predecessors are known
        self.builder.seal_block(cl_block);

        Ok(())
    }

    fn translate_instruction(
        &mut self,
        instr: &crate::hlir::HlirInstr,
    ) -> Result<Option<cranelift_codegen::ir::Value>, String> {
        let ty = self.hlir_to_type(&instr.ty);

        match &instr.op {
            Op::Const(constant) => {
                let val = self.translate_constant(constant, &instr.ty)?;
                Ok(Some(val))
            }

            Op::Copy(src) => {
                let src_val = self.get_value(*src)?;
                Ok(Some(src_val))
            }

            Op::Binary { op, left, right } => {
                let lhs = self.get_value(*left)?;
                let rhs = self.get_value(*right)?;
                let result = self.translate_binary_op(*op, lhs, rhs, &instr.ty)?;
                Ok(Some(result))
            }

            Op::Unary { op, operand } => {
                let val = self.get_value(*operand)?;
                let result = self.translate_unary_op(*op, val, &instr.ty)?;
                Ok(Some(result))
            }

            Op::CallDirect { name, args } => {
                let arg_vals: Vec<_> = args
                    .iter()
                    .map(|a| self.get_value(*a))
                    .collect::<Result<_, _>>()?;

                if let Some(&func_id) = self.func_ids.get(name) {
                    let func_ref = self.builder.ins().func_addr(types::I64, func_id.into());
                    // For now, use direct call syntax
                    let sig = self
                        .builder
                        .func
                        .dfg
                        .signatures
                        .push(self.builder.func.signature.clone());
                    let call = self.builder.ins().call_indirect(sig, func_ref, &arg_vals);
                    let results = self.builder.inst_results(call);
                    if results.is_empty() {
                        Ok(None)
                    } else {
                        Ok(Some(results[0]))
                    }
                } else {
                    // Unknown function - return zero
                    let zero = self.builder.ins().iconst(ty, 0);
                    Ok(Some(zero))
                }
            }

            Op::Call { func, args } => {
                let func_val = self.get_value(*func)?;
                let arg_vals: Vec<_> = args
                    .iter()
                    .map(|a| self.get_value(*a))
                    .collect::<Result<_, _>>()?;

                // Indirect call
                let sig = self
                    .builder
                    .func
                    .dfg
                    .signatures
                    .push(self.builder.func.signature.clone());
                let call = self.builder.ins().call_indirect(sig, func_val, &arg_vals);
                let results = self.builder.inst_results(call);
                if results.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(results[0]))
                }
            }

            Op::Load { ptr } => {
                let ptr_val = self.get_value(*ptr)?;
                let loaded = self.builder.ins().load(ty, MemFlags::new(), ptr_val, 0);
                Ok(Some(loaded))
            }

            Op::Store { ptr, value } => {
                let ptr_val = self.get_value(*ptr)?;
                let val = self.get_value(*value)?;
                self.builder.ins().store(MemFlags::new(), val, ptr_val, 0);
                Ok(None)
            }

            Op::Alloca { ty: alloc_ty } => {
                let size = alloc_ty.size_bits() / 8;
                let size = if size == 0 { 8 } else { size };
                let slot = self.builder.create_sized_stack_slot(
                    cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        size as u32,
                        0,
                    ),
                );
                let addr = self.builder.ins().stack_addr(types::I64, slot, 0);
                Ok(Some(addr))
            }

            Op::GetFieldPtr { base, field } => {
                let base_val = self.get_value(*base)?;
                let offset = (*field * 8) as i32; // Assume 8-byte fields
                let ptr = self.builder.ins().iadd_imm(base_val, offset as i64);
                Ok(Some(ptr))
            }

            Op::GetElementPtr { base, index } => {
                let base_val = self.get_value(*base)?;
                let idx_val = self.get_value(*index)?;
                let elem_size = 8i64; // Assume 8-byte elements
                let offset = self.builder.ins().imul_imm(idx_val, elem_size);
                let ptr = self.builder.ins().iadd(base_val, offset);
                Ok(Some(ptr))
            }

            Op::Cast { value, target } => {
                let val = self.get_value(*value)?;
                let target_ty = self.hlir_to_type(target);
                let val_ty = self.builder.func.dfg.value_type(val);

                if val_ty == target_ty {
                    Ok(Some(val))
                } else if val_ty.bits() < target_ty.bits() {
                    // Extend
                    let extended = if target_ty.is_int() {
                        self.builder.ins().sextend(target_ty, val)
                    } else {
                        self.builder.ins().fpromote(target_ty, val)
                    };
                    Ok(Some(extended))
                } else {
                    // Truncate
                    let truncated = if target_ty.is_int() {
                        self.builder.ins().ireduce(target_ty, val)
                    } else {
                        self.builder.ins().fdemote(target_ty, val)
                    };
                    Ok(Some(truncated))
                }
            }

            Op::Phi { incoming } => {
                // Phi nodes should be handled as block parameters in Cranelift
                // For now, return first incoming value if available
                if let Some((_, first_val)) = incoming.first() {
                    let val = self.get_value(*first_val)?;
                    Ok(Some(val))
                } else {
                    let zero = self.builder.ins().iconst(ty, 0);
                    Ok(Some(zero))
                }
            }

            Op::ExtractValue { base, index } => {
                // For tuples/structs stored as aggregates
                let base_val = self.get_value(*base)?;
                // Simplified: treat as field access
                let offset = (*index * 8) as i32;
                let ptr = self.builder.ins().iadd_imm(base_val, offset as i64);
                let loaded = self.builder.ins().load(ty, MemFlags::new(), ptr, 0);
                Ok(Some(loaded))
            }

            Op::InsertValue { base, value, index } => {
                let base_val = self.get_value(*base)?;
                let val = self.get_value(*value)?;
                // Simplified: treat as field store
                let offset = (*index * 8) as i32;
                let ptr = self.builder.ins().iadd_imm(base_val, offset as i64);
                self.builder.ins().store(MemFlags::new(), val, ptr, 0);
                Ok(Some(base_val))
            }

            Op::Tuple(vals) | Op::Array(vals) => {
                // Allocate space and store values
                let size = (vals.len() * 8) as u32;
                let slot = self.builder.create_sized_stack_slot(
                    cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        size,
                        0,
                    ),
                );
                let base = self.builder.ins().stack_addr(types::I64, slot, 0);

                for (i, v) in vals.iter().enumerate() {
                    let val = self.get_value(*v)?;
                    let offset = (i * 8) as i32;
                    self.builder.ins().store(MemFlags::new(), val, base, offset);
                }

                Ok(Some(base))
            }

            Op::Struct { name: _, fields } => {
                // Similar to tuple
                let size = (fields.len() * 8) as u32;
                let slot = self.builder.create_sized_stack_slot(
                    cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        size,
                        0,
                    ),
                );
                let base = self.builder.ins().stack_addr(types::I64, slot, 0);

                for (i, (_, v)) in fields.iter().enumerate() {
                    let val = self.get_value(*v)?;
                    let offset = (i * 8) as i32;
                    self.builder.ins().store(MemFlags::new(), val, base, offset);
                }

                Ok(Some(base))
            }

            Op::PerformEffect { .. } => {
                // Effects not supported in JIT yet
                let zero = self.builder.ins().iconst(ty, 0);
                Ok(Some(zero))
            }
        }
    }

    fn translate_constant(
        &mut self,
        constant: &HlirConstant,
        ty: &HlirType,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        let cl_ty = self.hlir_to_type(ty);

        match constant {
            HlirConstant::Unit => Ok(self.builder.ins().iconst(types::I64, 0)),
            HlirConstant::Bool(b) => Ok(self.builder.ins().iconst(types::I8, *b as i64)),
            HlirConstant::Int(i, _) => Ok(self.builder.ins().iconst(cl_ty, *i)),
            HlirConstant::Float(f, _) => {
                if cl_ty == types::F32 {
                    Ok(self.builder.ins().f32const(*f as f32))
                } else {
                    Ok(self.builder.ins().f64const(*f))
                }
            }
            HlirConstant::String(_) => {
                // Strings need special handling - for now return null ptr
                Ok(self.builder.ins().iconst(types::I64, 0))
            }
            HlirConstant::Null(_) => Ok(self.builder.ins().iconst(types::I64, 0)),
            HlirConstant::Undef(_) => Ok(self.builder.ins().iconst(cl_ty, 0)),
            HlirConstant::FunctionRef(name) => {
                if let Some(&func_id) = self.func_ids.get(name) {
                    Ok(self.builder.ins().func_addr(types::I64, func_id.into()))
                } else {
                    Ok(self.builder.ins().iconst(types::I64, 0))
                }
            }
            HlirConstant::GlobalRef(_) => Ok(self.builder.ins().iconst(types::I64, 0)),
            HlirConstant::Array(_) | HlirConstant::Struct(_) => {
                // Complex constants - return null for now
                Ok(self.builder.ins().iconst(types::I64, 0))
            }
        }
    }

    fn translate_binary_op(
        &mut self,
        op: BinaryOp,
        lhs: cranelift_codegen::ir::Value,
        rhs: cranelift_codegen::ir::Value,
        result_ty: &HlirType,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};

        let result = match op {
            // Integer arithmetic
            BinaryOp::Add => self.builder.ins().iadd(lhs, rhs),
            BinaryOp::Sub => self.builder.ins().isub(lhs, rhs),
            BinaryOp::Mul => self.builder.ins().imul(lhs, rhs),
            BinaryOp::SDiv => self.builder.ins().sdiv(lhs, rhs),
            BinaryOp::UDiv => self.builder.ins().udiv(lhs, rhs),
            BinaryOp::SRem => self.builder.ins().srem(lhs, rhs),
            BinaryOp::URem => self.builder.ins().urem(lhs, rhs),

            // Float arithmetic
            BinaryOp::FAdd => self.builder.ins().fadd(lhs, rhs),
            BinaryOp::FSub => self.builder.ins().fsub(lhs, rhs),
            BinaryOp::FMul => self.builder.ins().fmul(lhs, rhs),
            BinaryOp::FDiv => self.builder.ins().fdiv(lhs, rhs),
            BinaryOp::FRem => {
                // Cranelift doesn't have frem, use a workaround
                let div = self.builder.ins().fdiv(lhs, rhs);
                let trunc = self.builder.ins().trunc(div);
                let mul = self.builder.ins().fmul(trunc, rhs);
                self.builder.ins().fsub(lhs, mul)
            }

            // Bitwise
            BinaryOp::And => self.builder.ins().band(lhs, rhs),
            BinaryOp::Or => self.builder.ins().bor(lhs, rhs),
            BinaryOp::Xor => self.builder.ins().bxor(lhs, rhs),
            BinaryOp::Shl => self.builder.ins().ishl(lhs, rhs),
            BinaryOp::AShr => self.builder.ins().sshr(lhs, rhs),
            BinaryOp::LShr => self.builder.ins().ushr(lhs, rhs),

            // Integer comparison
            BinaryOp::Eq => self.builder.ins().icmp(IntCC::Equal, lhs, rhs),
            BinaryOp::Ne => self.builder.ins().icmp(IntCC::NotEqual, lhs, rhs),
            BinaryOp::SLt => self.builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs),
            BinaryOp::SLe => self
                .builder
                .ins()
                .icmp(IntCC::SignedLessThanOrEqual, lhs, rhs),
            BinaryOp::SGt => self.builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs),
            BinaryOp::SGe => self
                .builder
                .ins()
                .icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs),
            BinaryOp::ULt => self.builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs),
            BinaryOp::ULe => self
                .builder
                .ins()
                .icmp(IntCC::UnsignedLessThanOrEqual, lhs, rhs),
            BinaryOp::UGt => self
                .builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThan, lhs, rhs),
            BinaryOp::UGe => self
                .builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, lhs, rhs),

            // Float comparison
            BinaryOp::FOEq => self.builder.ins().fcmp(FloatCC::Equal, lhs, rhs),
            BinaryOp::FONe => self.builder.ins().fcmp(FloatCC::NotEqual, lhs, rhs),
            BinaryOp::FOLt => self.builder.ins().fcmp(FloatCC::LessThan, lhs, rhs),
            BinaryOp::FOLe => self.builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs, rhs),
            BinaryOp::FOGt => self.builder.ins().fcmp(FloatCC::GreaterThan, lhs, rhs),
            BinaryOp::FOGe => self
                .builder
                .ins()
                .fcmp(FloatCC::GreaterThanOrEqual, lhs, rhs),
        };

        // Comparisons return i8, may need to extend for result type
        let result_cl_ty = self.hlir_to_type(result_ty);
        let result_ty_current = self.builder.func.dfg.value_type(result);

        if result_ty_current != result_cl_ty && result_cl_ty.is_int() {
            if result_ty_current.bits() < result_cl_ty.bits() {
                Ok(self.builder.ins().uextend(result_cl_ty, result))
            } else {
                Ok(result)
            }
        } else {
            Ok(result)
        }
    }

    fn translate_unary_op(
        &mut self,
        op: UnaryOp,
        val: cranelift_codegen::ir::Value,
        ty: &HlirType,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        match op {
            UnaryOp::Neg => Ok(self.builder.ins().ineg(val)),
            UnaryOp::FNeg => Ok(self.builder.ins().fneg(val)),
            UnaryOp::Not => {
                // Logical not: xor with all 1s
                let ones = self.builder.ins().iconst(self.hlir_to_type(ty), -1);
                Ok(self.builder.ins().bxor(val, ones))
            }
        }
    }

    fn translate_terminator(&mut self, term: &HlirTerminator) -> Result<(), String> {
        match term {
            HlirTerminator::Return(val) => {
                if let Some(v) = val {
                    let ret_val = self.get_value(*v)?;
                    self.builder.ins().return_(&[ret_val]);
                } else {
                    self.builder.ins().return_(&[]);
                }
            }

            HlirTerminator::Branch(target) => {
                let target_block = self.blocks[target];
                self.builder.ins().jump(target_block, &[]);
            }

            HlirTerminator::CondBranch {
                condition,
                then_block,
                else_block,
            } => {
                let cond = self.get_value(*condition)?;
                let then_b = self.blocks[then_block];
                let else_b = self.blocks[else_block];
                self.builder.ins().brif(cond, then_b, &[], else_b, &[]);
            }

            HlirTerminator::Switch {
                value,
                default,
                cases,
            } => {
                let val = self.get_value(*value)?;
                let default_block = self.blocks[default];

                // Build switch using a chain of conditionals
                // (Cranelift has br_table but it's more complex)
                let mut current_block = self.builder.current_block().unwrap();

                for (case_val, target) in cases {
                    let target_block = self.blocks[target];
                    let case_const = self.builder.ins().iconst(types::I64, *case_val);
                    let cmp = self.builder.ins().icmp(
                        cranelift_codegen::ir::condcodes::IntCC::Equal,
                        val,
                        case_const,
                    );

                    let next_block = self.builder.create_block();
                    self.builder
                        .ins()
                        .brif(cmp, target_block, &[], next_block, &[]);
                    self.builder.seal_block(next_block);
                    self.builder.switch_to_block(next_block);
                }

                self.builder.ins().jump(default_block, &[]);
            }

            HlirTerminator::Unreachable => {
                self.builder
                    .ins()
                    .trap(cranelift_codegen::ir::TrapCode::User(0));
            }
        }

        Ok(())
    }

    fn get_value(&self, id: ValueId) -> Result<cranelift_codegen::ir::Value, String> {
        self.values
            .get(&id)
            .copied()
            .ok_or_else(|| format!("Value not found: {:?}", id))
    }

    fn hlir_to_type(&self, ty: &HlirType) -> types::Type {
        match ty {
            HlirType::Void => types::I64,
            HlirType::Bool => types::I8,
            HlirType::I8 | HlirType::U8 => types::I8,
            HlirType::I16 | HlirType::U16 => types::I16,
            HlirType::I32 | HlirType::U32 => types::I32,
            HlirType::I64 | HlirType::U64 => types::I64,
            HlirType::I128 | HlirType::U128 => types::I128,
            HlirType::F32 => types::F32,
            HlirType::F64 => types::F64,
            HlirType::Ptr(_) => types::I64,
            HlirType::Array(_, _) => types::I64,
            HlirType::Struct(_) => types::I64,
            HlirType::Tuple(_) => types::I64,
            HlirType::Function { .. } => types::I64,
        }
    }
}

// Non-JIT placeholder implementation
#[cfg(not(feature = "jit"))]
impl CompiledModule {
    fn new_stub() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }
}
