//! GPU IR Builder
//!
//! Provides a fluent API for constructing GPU IR programs.

use super::effects::EffectSet;
use super::function::{BasicBlock, FunctionKind, GpuFunction, GpuModule, Parameter};
use super::inst::*;
use super::types::*;

/// Builder for constructing GPU functions
pub struct FunctionBuilder {
    /// The function being built
    func: GpuFunction,
    /// Current block being built
    current_block: BlockId,
    /// Next value ID
    next_value: u32,
    /// Declared effects
    effects: EffectSet,
}

impl FunctionBuilder {
    /// Create a new kernel builder
    pub fn kernel(name: impl Into<String>) -> Self {
        FunctionBuilder {
            func: GpuFunction::kernel(name),
            current_block: BlockId(0),
            next_value: 0,
            effects: EffectSet::new(),
        }
    }

    /// Create a new device function builder
    pub fn device(name: impl Into<String>) -> Self {
        FunctionBuilder {
            func: GpuFunction::device(name),
            current_block: BlockId(0),
            next_value: 0,
            effects: EffectSet::new(),
        }
    }

    /// Add a parameter
    pub fn param(&mut self, name: impl Into<String>, ty: GpuType) -> ValueId {
        let id = self.new_value();
        self.func.params.push(Parameter::new(name, ty.clone()));
        self.func.register_value(id, ty);
        id
    }

    /// Set return type
    pub fn returns(&mut self, ty: GpuType) -> &mut Self {
        self.func.return_type = Some(ty);
        self
    }

    /// Set max threads hint
    pub fn max_threads(&mut self, max: u32) -> &mut Self {
        self.func.max_threads = Some(max);
        self
    }

    /// Set min blocks hint
    pub fn min_blocks(&mut self, min: u32) -> &mut Self {
        self.func.min_blocks = Some(min);
        self
    }

    /// Add declared effects
    pub fn with_effects(&mut self, effects: EffectSet) -> &mut Self {
        self.effects = effects;
        self
    }

    // Block management

    /// Create a new basic block and return its ID
    pub fn new_block(&mut self) -> BlockId {
        self.func.create_block()
    }

    /// Create a new labeled block
    pub fn new_labeled_block(&mut self, label: impl Into<String>) -> BlockId {
        self.func.create_labeled_block(label)
    }

    /// Switch to building a different block
    pub fn switch_to(&mut self, block: BlockId) -> &mut Self {
        self.current_block = block;
        self
    }

    /// Get the current block ID
    pub fn current_block(&self) -> BlockId {
        self.current_block
    }

    // Value generation

    fn new_value(&mut self) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value += 1;
        id
    }

    fn emit(&mut self, inst: Instruction) {
        if let Some(block) = self.func.get_block_mut(self.current_block) {
            block.push(inst);
        }
    }

    // Constants

    /// Create an i32 constant
    pub fn const_i32(&mut self, value: i32) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::I32));
        self.emit(Instruction::Const {
            dst: id,
            value: Constant::I32(value),
        });
        id
    }

    /// Create a u32 constant
    pub fn const_u32(&mut self, value: u32) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::U32));
        self.emit(Instruction::Const {
            dst: id,
            value: Constant::U32(value),
        });
        id
    }

    /// Create an i64 constant
    pub fn const_i64(&mut self, value: i64) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::I64));
        self.emit(Instruction::Const {
            dst: id,
            value: Constant::I64(value),
        });
        id
    }

    /// Create a u64 constant
    pub fn const_u64(&mut self, value: u64) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::U64));
        self.emit(Instruction::Const {
            dst: id,
            value: Constant::U64(value),
        });
        id
    }

    /// Create an f32 constant
    pub fn const_f32(&mut self, value: f32) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::F32));
        self.emit(Instruction::Const {
            dst: id,
            value: Constant::F32(value),
        });
        id
    }

    /// Create an f64 constant
    pub fn const_f64(&mut self, value: f64) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::F64));
        self.emit(Instruction::Const {
            dst: id,
            value: Constant::F64(value),
        });
        id
    }

    /// Create a boolean constant
    pub fn const_bool(&mut self, value: bool) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::Bool));
        self.emit(Instruction::Const {
            dst: id,
            value: Constant::Bool(value),
        });
        id
    }

    // Binary operations

    fn binop(&mut self, op: BinOp, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        let id = self.new_value();
        self.func.register_value(id, GpuType::Scalar(ty));
        self.emit(Instruction::BinOp {
            dst: id,
            op,
            lhs,
            rhs,
            ty,
        });
        id
    }

    pub fn add(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::Add, lhs, rhs, ty)
    }

    pub fn sub(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::Sub, lhs, rhs, ty)
    }

    pub fn mul(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::Mul, lhs, rhs, ty)
    }

    pub fn div(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::Div, lhs, rhs, ty)
    }

    pub fn rem(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::Rem, lhs, rhs, ty)
    }

    pub fn fadd(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::FAdd, lhs, rhs, ty)
    }

    pub fn fsub(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::FSub, lhs, rhs, ty)
    }

    pub fn fmul(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::FMul, lhs, rhs, ty)
    }

    pub fn fdiv(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::FDiv, lhs, rhs, ty)
    }

    pub fn and(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::And, lhs, rhs, ty)
    }

    pub fn or(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::Or, lhs, rhs, ty)
    }

    pub fn xor(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::Xor, lhs, rhs, ty)
    }

    pub fn shl(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::Shl, lhs, rhs, ty)
    }

    pub fn shr(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::Shr, lhs, rhs, ty)
    }

    pub fn min(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::Min, lhs, rhs, ty)
    }

    pub fn max(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::Max, lhs, rhs, ty)
    }

    pub fn fmin(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::FMin, lhs, rhs, ty)
    }

    pub fn fmax(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.binop(BinOp::FMax, lhs, rhs, ty)
    }

    // Unary operations

    fn unaryop(&mut self, op: UnaryOp, src: ValueId, ty: ScalarType) -> ValueId {
        let id = self.new_value();
        self.func.register_value(id, GpuType::Scalar(ty));
        self.emit(Instruction::UnaryOp {
            dst: id,
            op,
            src,
            ty,
        });
        id
    }

    pub fn neg(&mut self, src: ValueId, ty: ScalarType) -> ValueId {
        self.unaryop(UnaryOp::Neg, src, ty)
    }

    pub fn not(&mut self, src: ValueId, ty: ScalarType) -> ValueId {
        self.unaryop(UnaryOp::Not, src, ty)
    }

    pub fn fneg(&mut self, src: ValueId, ty: ScalarType) -> ValueId {
        self.unaryop(UnaryOp::FNeg, src, ty)
    }

    pub fn fabs(&mut self, src: ValueId, ty: ScalarType) -> ValueId {
        self.unaryop(UnaryOp::FAbs, src, ty)
    }

    pub fn sqrt(&mut self, src: ValueId, ty: ScalarType) -> ValueId {
        self.unaryop(UnaryOp::FSqrt, src, ty)
    }

    pub fn rsqrt(&mut self, src: ValueId, ty: ScalarType) -> ValueId {
        self.unaryop(UnaryOp::FRsqrt, src, ty)
    }

    pub fn sin(&mut self, src: ValueId, ty: ScalarType) -> ValueId {
        self.unaryop(UnaryOp::FSin, src, ty)
    }

    pub fn cos(&mut self, src: ValueId, ty: ScalarType) -> ValueId {
        self.unaryop(UnaryOp::FCos, src, ty)
    }

    pub fn exp2(&mut self, src: ValueId, ty: ScalarType) -> ValueId {
        self.unaryop(UnaryOp::FExp2, src, ty)
    }

    pub fn log2(&mut self, src: ValueId, ty: ScalarType) -> ValueId {
        self.unaryop(UnaryOp::FLog2, src, ty)
    }

    // Comparison

    pub fn cmp(&mut self, op: CmpOp, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::Bool));
        self.emit(Instruction::Cmp {
            dst: id,
            op,
            lhs,
            rhs,
            ty,
        });
        id
    }

    pub fn eq(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.cmp(CmpOp::Eq, lhs, rhs, ty)
    }

    pub fn ne(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.cmp(CmpOp::Ne, lhs, rhs, ty)
    }

    pub fn lt(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.cmp(CmpOp::Lt, lhs, rhs, ty)
    }

    pub fn le(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.cmp(CmpOp::Le, lhs, rhs, ty)
    }

    pub fn gt(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.cmp(CmpOp::Gt, lhs, rhs, ty)
    }

    pub fn ge(&mut self, lhs: ValueId, rhs: ValueId, ty: ScalarType) -> ValueId {
        self.cmp(CmpOp::Ge, lhs, rhs, ty)
    }

    // Type conversion

    pub fn convert(&mut self, src: ValueId, from: ScalarType, to: ScalarType) -> ValueId {
        let id = self.new_value();
        self.func.register_value(id, GpuType::Scalar(to));
        self.emit(Instruction::Convert {
            dst: id,
            src,
            from,
            to,
        });
        id
    }

    pub fn bitcast(&mut self, src: ValueId, to: GpuType) -> ValueId {
        let id = self.new_value();
        self.func.register_value(id, to.clone());
        self.emit(Instruction::Bitcast { dst: id, src, to });
        id
    }

    // Memory operations

    pub fn load(&mut self, ptr: ValueId, ty: GpuType, space: AddressSpace) -> ValueId {
        let id = self.new_value();
        self.func.register_value(id, ty.clone());
        self.emit(Instruction::Load {
            dst: id,
            ptr,
            ty,
            space,
            volatile: false,
        });
        id
    }

    pub fn load_volatile(&mut self, ptr: ValueId, ty: GpuType, space: AddressSpace) -> ValueId {
        let id = self.new_value();
        self.func.register_value(id, ty.clone());
        self.emit(Instruction::Load {
            dst: id,
            ptr,
            ty,
            space,
            volatile: true,
        });
        id
    }

    pub fn store(&mut self, ptr: ValueId, value: ValueId, ty: GpuType, space: AddressSpace) {
        self.emit(Instruction::Store {
            ptr,
            value,
            ty,
            space,
            volatile: false,
        });
    }

    pub fn store_volatile(
        &mut self,
        ptr: ValueId,
        value: ValueId,
        ty: GpuType,
        space: AddressSpace,
    ) {
        self.emit(Instruction::Store {
            ptr,
            value,
            ty,
            space,
            volatile: true,
        });
    }

    // Address calculation

    pub fn gep(&mut self, base: ValueId, indices: Vec<ValueId>, pointee_ty: GpuType) -> ValueId {
        let id = self.new_value();
        // Result is a pointer to the element type
        self.func.register_value(
            id,
            GpuType::Ptr(Box::new(pointee_ty.clone()), AddressSpace::Global),
        );
        self.emit(Instruction::GetElementPtr {
            dst: id,
            base,
            indices,
            pointee_ty,
        });
        id
    }

    // Atomic operations

    pub fn atomic(
        &mut self,
        op: AtomicOp,
        ptr: ValueId,
        value: ValueId,
        ty: ScalarType,
        space: AddressSpace,
        order: MemoryOrder,
    ) -> ValueId {
        let id = self.new_value();
        self.func.register_value(id, GpuType::Scalar(ty));
        self.emit(Instruction::Atomic {
            dst: id,
            op,
            ptr,
            value,
            ty,
            space,
            order,
        });
        id
    }

    pub fn atomic_add(&mut self, ptr: ValueId, value: ValueId, ty: ScalarType) -> ValueId {
        self.atomic(
            AtomicOp::Add,
            ptr,
            value,
            ty,
            AddressSpace::Global,
            MemoryOrder::Relaxed,
        )
    }

    pub fn atomic_cas(
        &mut self,
        ptr: ValueId,
        expected: ValueId,
        desired: ValueId,
        ty: ScalarType,
        space: AddressSpace,
    ) -> ValueId {
        let id = self.new_value();
        self.func.register_value(id, GpuType::Scalar(ty));
        self.emit(Instruction::AtomicCAS {
            dst: id,
            ptr,
            expected,
            desired,
            ty,
            space,
        });
        id
    }

    // Thread identification

    pub fn thread_idx(&mut self, dim: Dimension) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::U32));
        self.emit(Instruction::ThreadIdx { dst: id, dim });
        id
    }

    pub fn block_idx(&mut self, dim: Dimension) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::U32));
        self.emit(Instruction::BlockIdx { dst: id, dim });
        id
    }

    pub fn block_dim(&mut self, dim: Dimension) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::U32));
        self.emit(Instruction::BlockDim { dst: id, dim });
        id
    }

    pub fn grid_dim(&mut self, dim: Dimension) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::U32));
        self.emit(Instruction::GridDim { dst: id, dim });
        id
    }

    /// Compute global thread ID: blockIdx * blockDim + threadIdx
    pub fn global_thread_id(&mut self, dim: Dimension) -> ValueId {
        let block_idx = self.block_idx(dim);
        let block_dim = self.block_dim(dim);
        let thread_idx = self.thread_idx(dim);
        let tmp = self.mul(block_idx, block_dim, ScalarType::U32);
        self.add(tmp, thread_idx, ScalarType::U32)
    }

    // Synchronization

    pub fn barrier(&mut self, scope: BarrierScope) {
        self.emit(Instruction::Barrier { scope });
    }

    pub fn block_barrier(&mut self) {
        self.barrier(BarrierScope::Block);
    }

    pub fn mem_fence(&mut self, scope: BarrierScope, order: MemoryOrder) {
        self.emit(Instruction::MemFence { scope, order });
    }

    // Warp operations

    pub fn warp_id(&mut self) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::U32));
        self.emit(Instruction::WarpId { dst: id });
        id
    }

    pub fn lane_id(&mut self) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::U32));
        self.emit(Instruction::LaneId { dst: id });
        id
    }

    pub fn warp_shuffle(&mut self, src: ValueId, lane: ValueId, ty: ScalarType) -> ValueId {
        let id = self.new_value();
        self.func.register_value(id, GpuType::Scalar(ty));
        self.emit(Instruction::WarpShuffle {
            dst: id,
            src,
            lane,
            ty,
        });
        id
    }

    pub fn warp_all(&mut self, pred: ValueId) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::Bool));
        self.emit(Instruction::WarpVote {
            dst: id,
            pred,
            kind: WarpVoteKind::All,
        });
        id
    }

    pub fn warp_any(&mut self, pred: ValueId) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Scalar(ScalarType::Bool));
        self.emit(Instruction::WarpVote {
            dst: id,
            pred,
            kind: WarpVoteKind::Any,
        });
        id
    }

    pub fn warp_reduce(&mut self, src: ValueId, op: BinOp, ty: ScalarType) -> ValueId {
        let id = self.new_value();
        self.func.register_value(id, GpuType::Scalar(ty));
        self.emit(Instruction::WarpReduce {
            dst: id,
            src,
            op,
            ty,
        });
        id
    }

    // Control flow

    pub fn branch(&mut self, target: BlockId) {
        self.emit(Instruction::Branch { target });
    }

    pub fn cond_branch(&mut self, cond: ValueId, true_target: BlockId, false_target: BlockId) {
        self.emit(Instruction::CondBranch {
            cond,
            true_target,
            false_target,
        });
    }

    pub fn ret(&mut self, value: Option<ValueId>) {
        self.emit(Instruction::Return { value });
    }

    pub fn ret_void(&mut self) {
        self.ret(None);
    }

    // Selection

    pub fn select(
        &mut self,
        cond: ValueId,
        true_val: ValueId,
        false_val: ValueId,
        ty: GpuType,
    ) -> ValueId {
        let id = self.new_value();
        self.func.register_value(id, ty);
        self.emit(Instruction::Select {
            dst: id,
            cond,
            true_val,
            false_val,
        });
        id
    }

    // Phi nodes

    pub fn phi(&mut self, incoming: Vec<(BlockId, ValueId)>, ty: GpuType) -> ValueId {
        let id = self.new_value();
        self.func.register_value(id, ty);
        self.emit(Instruction::Phi { dst: id, incoming });
        id
    }

    // FMA

    pub fn fma(&mut self, a: ValueId, b: ValueId, c: ValueId, ty: ScalarType) -> ValueId {
        let id = self.new_value();
        self.func.register_value(id, GpuType::Scalar(ty));
        self.emit(Instruction::FMA {
            dst: id,
            a,
            b,
            c,
            ty,
        });
        id
    }

    // Shared memory

    pub fn shared_alloc(&mut self, ty: GpuType, size: usize) -> ValueId {
        let id = self.new_value();
        self.func
            .register_value(id, GpuType::Ptr(Box::new(ty.clone()), AddressSpace::Shared));
        self.emit(Instruction::SharedAlloc { dst: id, ty, size });
        id
    }

    // Function calls

    pub fn call(
        &mut self,
        func: impl Into<String>,
        args: Vec<ValueId>,
        ret_ty: Option<GpuType>,
    ) -> Option<ValueId> {
        let dst = ret_ty.map(|ty| {
            let id = self.new_value();
            self.func.register_value(id, ty);
            id
        });
        self.emit(Instruction::Call {
            dst,
            func: func.into(),
            args,
        });
        dst
    }

    /// Finish building and return the function
    pub fn build(self) -> GpuFunction {
        self.func
    }
}

/// Builder for constructing GPU modules
pub struct ModuleBuilder {
    module: GpuModule,
}

impl ModuleBuilder {
    /// Create a new module builder
    pub fn new(name: impl Into<String>) -> Self {
        ModuleBuilder {
            module: GpuModule::new(name),
        }
    }

    /// Set PTX version
    pub fn ptx_version(mut self, major: u32, minor: u32) -> Self {
        self.module.ptx_version = (major, minor);
        self
    }

    /// Set SM version
    pub fn sm_version(mut self, version: u32) -> Self {
        self.module.sm_version = version;
        self
    }

    /// Add a function
    pub fn add_function(&mut self, func: GpuFunction) -> &mut Self {
        self.module.add_function(func);
        self
    }

    /// Add a constant
    pub fn add_constant(
        &mut self,
        name: impl Into<String>,
        ty: GpuType,
        data: Vec<u8>,
    ) -> &mut Self {
        self.module.add_constant(name, ty, data);
        self
    }

    /// Build the module
    pub fn build(self) -> GpuModule {
        self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_simple_kernel() {
        let mut builder = FunctionBuilder::kernel("vector_add");

        // Parameters
        let a = builder.param("a", GpuType::global_ptr(common::f32()));
        let b = builder.param("b", GpuType::global_ptr(common::f32()));
        let c = builder.param("c", GpuType::global_ptr(common::f32()));
        let n = builder.param("n", common::u32());

        builder.max_threads(256);

        // Body
        let idx = builder.global_thread_id(Dimension::X);
        let in_bounds = builder.lt(idx, n, ScalarType::U32);

        let then_block = builder.new_labeled_block("then");
        let exit_block = builder.new_labeled_block("exit");

        builder.cond_branch(in_bounds, then_block, exit_block);

        // Then block
        builder.switch_to(then_block);
        let idx_i64 = builder.convert(idx, ScalarType::U32, ScalarType::I64);
        let a_ptr = builder.gep(a, vec![idx_i64], common::f32());
        let b_ptr = builder.gep(b, vec![idx_i64], common::f32());
        let c_ptr = builder.gep(c, vec![idx_i64], common::f32());

        let a_val = builder.load(a_ptr, common::f32(), AddressSpace::Global);
        let b_val = builder.load(b_ptr, common::f32(), AddressSpace::Global);
        let sum = builder.fadd(a_val, b_val, ScalarType::F32);
        builder.store(c_ptr, sum, common::f32(), AddressSpace::Global);
        builder.branch(exit_block);

        // Exit block
        builder.switch_to(exit_block);
        builder.ret_void();

        let func = builder.build();

        assert_eq!(func.name, "vector_add");
        assert_eq!(func.kind, FunctionKind::Kernel);
        assert_eq!(func.params.len(), 4);
        assert_eq!(func.blocks.len(), 3);
        assert!(func.validate().is_ok());
    }

    #[test]
    fn test_build_module() {
        let kernel = FunctionBuilder::kernel("test").build();

        // Add return to make it valid
        let mut builder = FunctionBuilder::kernel("test");
        builder.ret_void();
        let kernel = builder.build();

        let mut module_builder = ModuleBuilder::new("test_module")
            .ptx_version(8, 0)
            .sm_version(86);

        module_builder.add_function(kernel);

        let module = module_builder.build();

        assert_eq!(module.name, "test_module");
        assert_eq!(module.ptx_version, (8, 0));
        assert_eq!(module.sm_version, 86);
        assert_eq!(module.functions.len(), 1);
    }
}
