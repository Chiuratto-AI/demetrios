//! CPU Simulation Backend
//!
//! Interprets GPU IR on the CPU for testing and debugging.

use crate::ir::*;
use crate::runtime::{GpuBuffer, LaunchConfig};
use std::collections::HashMap;
use thiserror::Error;

/// CPU simulation errors
#[derive(Debug, Error)]
pub enum CpuSimError {
    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("Out of bounds access: index {index}, size {size}")]
    OutOfBounds { index: usize, size: usize },

    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Division by zero")]
    DivisionByZero,
}

/// Runtime value for CPU simulation
#[derive(Debug, Clone)]
pub enum Value {
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Ptr(*mut u8),
}

impl Value {
    pub fn as_bool(&self) -> Result<bool, CpuSimError> {
        match self {
            Value::Bool(v) => Ok(*v),
            Value::I32(v) => Ok(*v != 0),
            Value::U32(v) => Ok(*v != 0),
            _ => Err(CpuSimError::TypeMismatch {
                expected: "bool".to_string(),
                actual: format!("{:?}", self),
            }),
        }
    }

    pub fn as_i32(&self) -> Result<i32, CpuSimError> {
        match self {
            Value::I32(v) => Ok(*v),
            Value::U32(v) => Ok(*v as i32),
            Value::I64(v) => Ok(*v as i32),
            Value::U64(v) => Ok(*v as i32),
            _ => Err(CpuSimError::TypeMismatch {
                expected: "i32".to_string(),
                actual: format!("{:?}", self),
            }),
        }
    }

    pub fn as_u32(&self) -> Result<u32, CpuSimError> {
        match self {
            Value::U32(v) => Ok(*v),
            Value::I32(v) => Ok(*v as u32),
            Value::U64(v) => Ok(*v as u32),
            Value::I64(v) => Ok(*v as u32),
            _ => Err(CpuSimError::TypeMismatch {
                expected: "u32".to_string(),
                actual: format!("{:?}", self),
            }),
        }
    }

    pub fn as_i64(&self) -> Result<i64, CpuSimError> {
        match self {
            Value::I64(v) => Ok(*v),
            Value::U64(v) => Ok(*v as i64),
            Value::I32(v) => Ok(*v as i64),
            Value::U32(v) => Ok(*v as i64),
            _ => Err(CpuSimError::TypeMismatch {
                expected: "i64".to_string(),
                actual: format!("{:?}", self),
            }),
        }
    }

    pub fn as_u64(&self) -> Result<u64, CpuSimError> {
        match self {
            Value::U64(v) => Ok(*v),
            Value::I64(v) => Ok(*v as u64),
            Value::U32(v) => Ok(*v as u64),
            Value::I32(v) => Ok(*v as u64),
            Value::Ptr(p) => Ok(*p as u64),
            _ => Err(CpuSimError::TypeMismatch {
                expected: "u64".to_string(),
                actual: format!("{:?}", self),
            }),
        }
    }

    pub fn as_f32(&self) -> Result<f32, CpuSimError> {
        match self {
            Value::F32(v) => Ok(*v),
            Value::F64(v) => Ok(*v as f32),
            Value::I32(v) => Ok(*v as f32),
            Value::U32(v) => Ok(*v as f32),
            _ => Err(CpuSimError::TypeMismatch {
                expected: "f32".to_string(),
                actual: format!("{:?}", self),
            }),
        }
    }

    pub fn as_f64(&self) -> Result<f64, CpuSimError> {
        match self {
            Value::F64(v) => Ok(*v),
            Value::F32(v) => Ok(*v as f64),
            Value::I64(v) => Ok(*v as f64),
            Value::U64(v) => Ok(*v as f64),
            _ => Err(CpuSimError::TypeMismatch {
                expected: "f64".to_string(),
                actual: format!("{:?}", self),
            }),
        }
    }

    pub fn as_ptr(&self) -> Result<*mut u8, CpuSimError> {
        match self {
            Value::Ptr(p) => Ok(*p),
            Value::U64(v) => Ok(*v as *mut u8),
            _ => Err(CpuSimError::TypeMismatch {
                expected: "ptr".to_string(),
                actual: format!("{:?}", self),
            }),
        }
    }

    pub fn from_constant(c: &Constant) -> Self {
        match c {
            Constant::Bool(v) => Value::Bool(*v),
            Constant::I8(v) => Value::I8(*v),
            Constant::I16(v) => Value::I16(*v),
            Constant::I32(v) => Value::I32(*v),
            Constant::I64(v) => Value::I64(*v),
            Constant::U8(v) => Value::U8(*v),
            Constant::U16(v) => Value::U16(*v),
            Constant::U32(v) => Value::U32(*v),
            Constant::U64(v) => Value::U64(*v),
            Constant::F32(v) => Value::F32(*v),
            Constant::F64(v) => Value::F64(*v),
        }
    }
}

/// Thread context for CPU simulation
pub struct ThreadContext {
    /// Thread index (x, y, z)
    pub thread_idx: (u32, u32, u32),
    /// Block index (x, y, z)
    pub block_idx: (u32, u32, u32),
    /// Block dimensions
    pub block_dim: (u32, u32, u32),
    /// Grid dimensions
    pub grid_dim: (u32, u32, u32),
    /// Local values
    pub values: HashMap<ValueId, Value>,
}

impl ThreadContext {
    pub fn new(
        thread_idx: (u32, u32, u32),
        block_idx: (u32, u32, u32),
        block_dim: (u32, u32, u32),
        grid_dim: (u32, u32, u32),
    ) -> Self {
        ThreadContext {
            thread_idx,
            block_idx,
            block_dim,
            grid_dim,
            values: HashMap::new(),
        }
    }

    pub fn global_thread_id(&self) -> (u64, u64, u64) {
        (
            self.block_idx.0 as u64 * self.block_dim.0 as u64 + self.thread_idx.0 as u64,
            self.block_idx.1 as u64 * self.block_dim.1 as u64 + self.thread_idx.1 as u64,
            self.block_idx.2 as u64 * self.block_dim.2 as u64 + self.thread_idx.2 as u64,
        )
    }

    pub fn set(&mut self, id: ValueId, value: Value) {
        self.values.insert(id, value);
    }

    pub fn get(&self, id: ValueId) -> Result<&Value, CpuSimError> {
        self.values
            .get(&id)
            .ok_or_else(|| CpuSimError::InvalidValue(format!("Value {} not found", id)))
    }
}

/// CPU simulation interpreter
pub struct CpuInterpreter {
    /// Shared memory simulation
    shared_memory: Vec<u8>,
}

impl CpuInterpreter {
    pub fn new() -> Self {
        CpuInterpreter {
            shared_memory: Vec::new(),
        }
    }

    /// Execute a kernel
    pub fn execute(
        &mut self,
        func: &GpuFunction,
        config: &LaunchConfig,
        args: &[Value],
    ) -> Result<(), CpuSimError> {
        // Validate argument count
        if args.len() != func.params.len() {
            return Err(CpuSimError::InvalidValue(format!(
                "Expected {} arguments, got {}",
                func.params.len(),
                args.len()
            )));
        }

        // Allocate shared memory
        let shared_size: usize = func
            .shared_mem
            .iter()
            .map(|s| s.ty.size_bytes().unwrap_or(0) * s.size)
            .sum();
        self.shared_memory
            .resize(shared_size + config.shared_mem, 0);

        let grid = &config.grid;
        let block = &config.block;

        // Iterate over all blocks
        for bz in 0..grid.z {
            for by in 0..grid.y {
                for bx in 0..grid.x {
                    // Iterate over all threads in the block
                    for tz in 0..block.z {
                        for ty in 0..block.y {
                            for tx in 0..block.x {
                                let mut ctx = ThreadContext::new(
                                    (tx, ty, tz),
                                    (bx, by, bz),
                                    (block.x, block.y, block.z),
                                    (grid.x, grid.y, grid.z),
                                );

                                // Load parameters
                                for (i, arg) in args.iter().enumerate() {
                                    ctx.set(ValueId(i as u32), arg.clone());
                                }

                                // Execute the function
                                self.execute_function(func, &mut ctx)?;
                            }
                        }
                    }

                    // Block barrier (implicit at end of block)
                    // In real GPU, this would sync threads
                }
            }
        }

        Ok(())
    }

    fn execute_function(
        &mut self,
        func: &GpuFunction,
        ctx: &mut ThreadContext,
    ) -> Result<Option<Value>, CpuSimError> {
        let mut current_block = func.entry;

        loop {
            let block = func.get_block(current_block).ok_or_else(|| {
                CpuSimError::InvalidValue(format!("Block {} not found", current_block))
            })?;

            for inst in &block.instructions {
                match self.execute_instruction(inst, ctx)? {
                    ControlFlow::Continue => {}
                    ControlFlow::Branch(target) => {
                        current_block = target;
                        break;
                    }
                    ControlFlow::Return(value) => {
                        return Ok(value);
                    }
                }
            }

            // If we didn't branch, we've reached the end
            if !block.has_terminator() {
                break;
            }
        }

        Ok(None)
    }

    fn execute_instruction(
        &mut self,
        inst: &Instruction,
        ctx: &mut ThreadContext,
    ) -> Result<ControlFlow, CpuSimError> {
        match inst {
            Instruction::Const { dst, value } => {
                ctx.set(*dst, Value::from_constant(value));
            }

            Instruction::BinOp {
                dst,
                op,
                lhs,
                rhs,
                ty,
            } => {
                let lhs_val = ctx.get(*lhs)?;
                let rhs_val = ctx.get(*rhs)?;
                let result = self.execute_binop(*op, lhs_val, rhs_val, *ty)?;
                ctx.set(*dst, result);
            }

            Instruction::UnaryOp { dst, op, src, ty } => {
                let src_val = ctx.get(*src)?;
                let result = self.execute_unaryop(*op, src_val, *ty)?;
                ctx.set(*dst, result);
            }

            Instruction::Cmp {
                dst,
                op,
                lhs,
                rhs,
                ty,
            } => {
                let lhs_val = ctx.get(*lhs)?;
                let rhs_val = ctx.get(*rhs)?;
                let result = self.execute_cmp(*op, lhs_val, rhs_val, *ty)?;
                ctx.set(*dst, Value::Bool(result));
            }

            Instruction::Convert { dst, src, from, to } => {
                let src_val = ctx.get(*src)?;
                let result = self.execute_convert(src_val, *from, *to)?;
                ctx.set(*dst, result);
            }

            Instruction::Load { dst, ptr, ty, .. } => {
                let ptr_val = ctx.get(*ptr)?.as_ptr()?;
                let result = self.load_value(ptr_val, ty)?;
                ctx.set(*dst, result);
            }

            Instruction::Store { ptr, value, ty, .. } => {
                let ptr_val = ctx.get(*ptr)?.as_ptr()?;
                let val = ctx.get(*value)?;
                self.store_value(ptr_val, val, ty)?;
            }

            Instruction::GetElementPtr {
                dst,
                base,
                indices,
                pointee_ty,
            } => {
                let base_ptr = ctx.get(*base)?.as_ptr()?;
                let mut offset = 0usize;

                for idx in indices {
                    let idx_val = ctx.get(*idx)?.as_i64()? as usize;
                    offset += idx_val * pointee_ty.size_bytes().unwrap_or(1);
                }

                let result = unsafe { base_ptr.add(offset) };
                ctx.set(*dst, Value::Ptr(result));
            }

            Instruction::Branch { target } => {
                return Ok(ControlFlow::Branch(*target));
            }

            Instruction::CondBranch {
                cond,
                true_target,
                false_target,
            } => {
                let cond_val = ctx.get(*cond)?.as_bool()?;
                let target = if cond_val {
                    *true_target
                } else {
                    *false_target
                };
                return Ok(ControlFlow::Branch(target));
            }

            Instruction::Return { value } => {
                let ret_val = value.map(|v| ctx.get(v).cloned()).transpose()?;
                return Ok(ControlFlow::Return(ret_val));
            }

            Instruction::ThreadIdx { dst, dim } => {
                let val = match dim {
                    Dimension::X => ctx.thread_idx.0,
                    Dimension::Y => ctx.thread_idx.1,
                    Dimension::Z => ctx.thread_idx.2,
                };
                ctx.set(*dst, Value::U32(val));
            }

            Instruction::BlockIdx { dst, dim } => {
                let val = match dim {
                    Dimension::X => ctx.block_idx.0,
                    Dimension::Y => ctx.block_idx.1,
                    Dimension::Z => ctx.block_idx.2,
                };
                ctx.set(*dst, Value::U32(val));
            }

            Instruction::BlockDim { dst, dim } => {
                let val = match dim {
                    Dimension::X => ctx.block_dim.0,
                    Dimension::Y => ctx.block_dim.1,
                    Dimension::Z => ctx.block_dim.2,
                };
                ctx.set(*dst, Value::U32(val));
            }

            Instruction::GridDim { dst, dim } => {
                let val = match dim {
                    Dimension::X => ctx.grid_dim.0,
                    Dimension::Y => ctx.grid_dim.1,
                    Dimension::Z => ctx.grid_dim.2,
                };
                ctx.set(*dst, Value::U32(val));
            }

            Instruction::Select {
                dst,
                cond,
                true_val,
                false_val,
            } => {
                let cond_val = ctx.get(*cond)?.as_bool()?;
                let result = if cond_val {
                    ctx.get(*true_val)?.clone()
                } else {
                    ctx.get(*false_val)?.clone()
                };
                ctx.set(*dst, result);
            }

            Instruction::FMA { dst, a, b, c, ty } => {
                let a_val = ctx.get(*a)?;
                let b_val = ctx.get(*b)?;
                let c_val = ctx.get(*c)?;

                let result = match ty {
                    ScalarType::F32 => {
                        let av = a_val.as_f32()?;
                        let bv = b_val.as_f32()?;
                        let cv = c_val.as_f32()?;
                        Value::F32(av.mul_add(bv, cv))
                    }
                    ScalarType::F64 => {
                        let av = a_val.as_f64()?;
                        let bv = b_val.as_f64()?;
                        let cv = c_val.as_f64()?;
                        Value::F64(av.mul_add(bv, cv))
                    }
                    _ => {
                        return Err(CpuSimError::UnsupportedOperation(
                            "FMA on non-float".to_string(),
                        ))
                    }
                };
                ctx.set(*dst, result);
            }

            Instruction::Barrier { .. } | Instruction::MemFence { .. } => {
                // No-op in single-threaded simulation
            }

            // Atomic operations (simplified for single-threaded)
            Instruction::Atomic {
                dst,
                op,
                ptr,
                value,
                ty,
                ..
            } => {
                let ptr_val = ctx.get(*ptr)?.as_ptr()?;
                let val = ctx.get(*value)?.clone();
                let old = self.load_value(ptr_val, &GpuType::Scalar(*ty))?;

                let new_val = self.execute_binop(
                    match op {
                        AtomicOp::Add => BinOp::Add,
                        AtomicOp::Sub => BinOp::Sub,
                        AtomicOp::And => BinOp::And,
                        AtomicOp::Or => BinOp::Or,
                        AtomicOp::Xor => BinOp::Xor,
                        AtomicOp::Min => BinOp::Min,
                        AtomicOp::Max => BinOp::Max,
                        AtomicOp::Exch | AtomicOp::CAS => {
                            ctx.set(*dst, old);
                            self.store_value(ptr_val, &val, &GpuType::Scalar(*ty))?;
                            return Ok(ControlFlow::Continue);
                        }
                    },
                    &old,
                    &val,
                    *ty,
                )?;

                ctx.set(*dst, old);
                self.store_value(ptr_val, &new_val, &GpuType::Scalar(*ty))?;
            }

            // Skip complex operations in simulation
            _ => {
                // Log unsupported operation but continue
            }
        }

        Ok(ControlFlow::Continue)
    }

    fn execute_binop(
        &self,
        op: BinOp,
        lhs: &Value,
        rhs: &Value,
        ty: ScalarType,
    ) -> Result<Value, CpuSimError> {
        match ty {
            ScalarType::I32 => {
                let a = lhs.as_i32()?;
                let b = rhs.as_i32()?;
                let result = match op {
                    BinOp::Add | BinOp::FAdd => a.wrapping_add(b),
                    BinOp::Sub | BinOp::FSub => a.wrapping_sub(b),
                    BinOp::Mul | BinOp::FMul => a.wrapping_mul(b),
                    BinOp::Div | BinOp::FDiv => {
                        if b == 0 {
                            return Err(CpuSimError::DivisionByZero);
                        }
                        a / b
                    }
                    BinOp::Rem | BinOp::FRem => {
                        if b == 0 {
                            return Err(CpuSimError::DivisionByZero);
                        }
                        a % b
                    }
                    BinOp::And => a & b,
                    BinOp::Or => a | b,
                    BinOp::Xor => a ^ b,
                    BinOp::Shl => a << (b as u32),
                    BinOp::Shr => a >> (b as u32),
                    BinOp::Min | BinOp::FMin => a.min(b),
                    BinOp::Max | BinOp::FMax => a.max(b),
                };
                Ok(Value::I32(result))
            }
            ScalarType::U32 => {
                let a = lhs.as_u32()?;
                let b = rhs.as_u32()?;
                let result = match op {
                    BinOp::Add | BinOp::FAdd => a.wrapping_add(b),
                    BinOp::Sub | BinOp::FSub => a.wrapping_sub(b),
                    BinOp::Mul | BinOp::FMul => a.wrapping_mul(b),
                    BinOp::Div | BinOp::FDiv => {
                        if b == 0 {
                            return Err(CpuSimError::DivisionByZero);
                        }
                        a / b
                    }
                    BinOp::Rem | BinOp::FRem => {
                        if b == 0 {
                            return Err(CpuSimError::DivisionByZero);
                        }
                        a % b
                    }
                    BinOp::And => a & b,
                    BinOp::Or => a | b,
                    BinOp::Xor => a ^ b,
                    BinOp::Shl => a << b,
                    BinOp::Shr => a >> b,
                    BinOp::Min | BinOp::FMin => a.min(b),
                    BinOp::Max | BinOp::FMax => a.max(b),
                };
                Ok(Value::U32(result))
            }
            ScalarType::F32 => {
                let a = lhs.as_f32()?;
                let b = rhs.as_f32()?;
                let result = match op {
                    BinOp::Add | BinOp::FAdd => a + b,
                    BinOp::Sub | BinOp::FSub => a - b,
                    BinOp::Mul | BinOp::FMul => a * b,
                    BinOp::Div | BinOp::FDiv => a / b,
                    BinOp::Rem | BinOp::FRem => a % b,
                    BinOp::Min | BinOp::FMin => a.min(b),
                    BinOp::Max | BinOp::FMax => a.max(b),
                    _ => {
                        return Err(CpuSimError::UnsupportedOperation(format!(
                            "{:?} on f32",
                            op
                        )))
                    }
                };
                Ok(Value::F32(result))
            }
            ScalarType::F64 => {
                let a = lhs.as_f64()?;
                let b = rhs.as_f64()?;
                let result = match op {
                    BinOp::Add | BinOp::FAdd => a + b,
                    BinOp::Sub | BinOp::FSub => a - b,
                    BinOp::Mul | BinOp::FMul => a * b,
                    BinOp::Div | BinOp::FDiv => a / b,
                    BinOp::Rem | BinOp::FRem => a % b,
                    BinOp::Min | BinOp::FMin => a.min(b),
                    BinOp::Max | BinOp::FMax => a.max(b),
                    _ => {
                        return Err(CpuSimError::UnsupportedOperation(format!(
                            "{:?} on f64",
                            op
                        )))
                    }
                };
                Ok(Value::F64(result))
            }
            _ => Err(CpuSimError::UnsupportedOperation(format!(
                "{:?} on {:?}",
                op, ty
            ))),
        }
    }

    fn execute_unaryop(
        &self,
        op: UnaryOp,
        src: &Value,
        ty: ScalarType,
    ) -> Result<Value, CpuSimError> {
        match ty {
            ScalarType::I32 => {
                let v = src.as_i32()?;
                let result = match op {
                    UnaryOp::Neg | UnaryOp::FNeg => -v,
                    UnaryOp::Not => !v,
                    _ => {
                        return Err(CpuSimError::UnsupportedOperation(format!(
                            "{:?} on i32",
                            op
                        )))
                    }
                };
                Ok(Value::I32(result))
            }
            ScalarType::F32 => {
                let v = src.as_f32()?;
                let result = match op {
                    UnaryOp::FNeg | UnaryOp::Neg => -v,
                    UnaryOp::FAbs => v.abs(),
                    UnaryOp::FSqrt => v.sqrt(),
                    UnaryOp::FRsqrt => 1.0 / v.sqrt(),
                    UnaryOp::FRcp => 1.0 / v,
                    UnaryOp::FSin => v.sin(),
                    UnaryOp::FCos => v.cos(),
                    UnaryOp::FExp2 => v.exp2(),
                    UnaryOp::FLog2 => v.log2(),
                    UnaryOp::FFloor => v.floor(),
                    UnaryOp::FCeil => v.ceil(),
                    UnaryOp::FTrunc => v.trunc(),
                    UnaryOp::FRound => v.round(),
                    UnaryOp::Not => {
                        return Err(CpuSimError::UnsupportedOperation("Not on f32".to_string()))
                    }
                };
                Ok(Value::F32(result))
            }
            ScalarType::F64 => {
                let v = src.as_f64()?;
                let result = match op {
                    UnaryOp::FNeg | UnaryOp::Neg => -v,
                    UnaryOp::FAbs => v.abs(),
                    UnaryOp::FSqrt => v.sqrt(),
                    UnaryOp::FRsqrt => 1.0 / v.sqrt(),
                    UnaryOp::FRcp => 1.0 / v,
                    UnaryOp::FSin => v.sin(),
                    UnaryOp::FCos => v.cos(),
                    UnaryOp::FExp2 => v.exp2(),
                    UnaryOp::FLog2 => v.log2(),
                    UnaryOp::FFloor => v.floor(),
                    UnaryOp::FCeil => v.ceil(),
                    UnaryOp::FTrunc => v.trunc(),
                    UnaryOp::FRound => v.round(),
                    UnaryOp::Not => {
                        return Err(CpuSimError::UnsupportedOperation("Not on f64".to_string()))
                    }
                };
                Ok(Value::F64(result))
            }
            _ => Err(CpuSimError::UnsupportedOperation(format!(
                "{:?} on {:?}",
                op, ty
            ))),
        }
    }

    fn execute_cmp(
        &self,
        op: CmpOp,
        lhs: &Value,
        rhs: &Value,
        ty: ScalarType,
    ) -> Result<bool, CpuSimError> {
        match ty {
            ScalarType::I32 => {
                let a = lhs.as_i32()?;
                let b = rhs.as_i32()?;
                Ok(match op {
                    CmpOp::Eq => a == b,
                    CmpOp::Ne => a != b,
                    CmpOp::Lt => a < b,
                    CmpOp::Le => a <= b,
                    CmpOp::Gt => a > b,
                    CmpOp::Ge => a >= b,
                })
            }
            ScalarType::U32 => {
                let a = lhs.as_u32()?;
                let b = rhs.as_u32()?;
                Ok(match op {
                    CmpOp::Eq => a == b,
                    CmpOp::Ne => a != b,
                    CmpOp::Lt => a < b,
                    CmpOp::Le => a <= b,
                    CmpOp::Gt => a > b,
                    CmpOp::Ge => a >= b,
                })
            }
            ScalarType::F32 => {
                let a = lhs.as_f32()?;
                let b = rhs.as_f32()?;
                Ok(match op {
                    CmpOp::Eq => a == b,
                    CmpOp::Ne => a != b,
                    CmpOp::Lt => a < b,
                    CmpOp::Le => a <= b,
                    CmpOp::Gt => a > b,
                    CmpOp::Ge => a >= b,
                })
            }
            ScalarType::F64 => {
                let a = lhs.as_f64()?;
                let b = rhs.as_f64()?;
                Ok(match op {
                    CmpOp::Eq => a == b,
                    CmpOp::Ne => a != b,
                    CmpOp::Lt => a < b,
                    CmpOp::Le => a <= b,
                    CmpOp::Gt => a > b,
                    CmpOp::Ge => a >= b,
                })
            }
            _ => Err(CpuSimError::UnsupportedOperation(format!(
                "{:?} on {:?}",
                op, ty
            ))),
        }
    }

    fn execute_convert(
        &self,
        src: &Value,
        from: ScalarType,
        to: ScalarType,
    ) -> Result<Value, CpuSimError> {
        match (from, to) {
            (ScalarType::I32, ScalarType::F32) => Ok(Value::F32(src.as_i32()? as f32)),
            (ScalarType::I32, ScalarType::F64) => Ok(Value::F64(src.as_i32()? as f64)),
            (ScalarType::I32, ScalarType::I64) => Ok(Value::I64(src.as_i32()? as i64)),
            (ScalarType::I32, ScalarType::U32) => Ok(Value::U32(src.as_i32()? as u32)),
            (ScalarType::U32, ScalarType::F32) => Ok(Value::F32(src.as_u32()? as f32)),
            (ScalarType::U32, ScalarType::F64) => Ok(Value::F64(src.as_u32()? as f64)),
            (ScalarType::U32, ScalarType::I64) => Ok(Value::I64(src.as_u32()? as i64)),
            (ScalarType::U32, ScalarType::U64) => Ok(Value::U64(src.as_u32()? as u64)),
            (ScalarType::I64, ScalarType::I32) => Ok(Value::I32(src.as_i64()? as i32)),
            (ScalarType::I64, ScalarType::U64) => Ok(Value::U64(src.as_i64()? as u64)),
            (ScalarType::U64, ScalarType::I64) => Ok(Value::I64(src.as_u64()? as i64)),
            (ScalarType::U64, ScalarType::U32) => Ok(Value::U32(src.as_u64()? as u32)),
            (ScalarType::F32, ScalarType::I32) => Ok(Value::I32(src.as_f32()? as i32)),
            (ScalarType::F32, ScalarType::F64) => Ok(Value::F64(src.as_f32()? as f64)),
            (ScalarType::F64, ScalarType::F32) => Ok(Value::F32(src.as_f64()? as f32)),
            (ScalarType::F64, ScalarType::I64) => Ok(Value::I64(src.as_f64()? as i64)),
            _ => Err(CpuSimError::UnsupportedOperation(format!(
                "convert {:?} -> {:?}",
                from, to
            ))),
        }
    }

    fn load_value(&self, ptr: *mut u8, ty: &GpuType) -> Result<Value, CpuSimError> {
        unsafe {
            match ty {
                GpuType::Scalar(ScalarType::I32) => Ok(Value::I32(*(ptr as *const i32))),
                GpuType::Scalar(ScalarType::U32) => Ok(Value::U32(*(ptr as *const u32))),
                GpuType::Scalar(ScalarType::I64) => Ok(Value::I64(*(ptr as *const i64))),
                GpuType::Scalar(ScalarType::U64) => Ok(Value::U64(*(ptr as *const u64))),
                GpuType::Scalar(ScalarType::F32) => Ok(Value::F32(*(ptr as *const f32))),
                GpuType::Scalar(ScalarType::F64) => Ok(Value::F64(*(ptr as *const f64))),
                GpuType::Ptr(_, _) => Ok(Value::Ptr(*(ptr as *const *mut u8))),
                _ => Err(CpuSimError::UnsupportedOperation(format!("load {:?}", ty))),
            }
        }
    }

    fn store_value(&self, ptr: *mut u8, val: &Value, ty: &GpuType) -> Result<(), CpuSimError> {
        unsafe {
            match (ty, val) {
                (GpuType::Scalar(ScalarType::I32), Value::I32(v)) => {
                    *(ptr as *mut i32) = *v;
                }
                (GpuType::Scalar(ScalarType::U32), Value::U32(v)) => {
                    *(ptr as *mut u32) = *v;
                }
                (GpuType::Scalar(ScalarType::I64), Value::I64(v)) => {
                    *(ptr as *mut i64) = *v;
                }
                (GpuType::Scalar(ScalarType::U64), Value::U64(v)) => {
                    *(ptr as *mut u64) = *v;
                }
                (GpuType::Scalar(ScalarType::F32), Value::F32(v)) => {
                    *(ptr as *mut f32) = *v;
                }
                (GpuType::Scalar(ScalarType::F64), Value::F64(v)) => {
                    *(ptr as *mut f64) = *v;
                }
                _ => {
                    return Err(CpuSimError::UnsupportedOperation(format!(
                        "store {:?} <- {:?}",
                        ty, val
                    )))
                }
            }
        }
        Ok(())
    }
}

impl Default for CpuInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

/// Control flow result
enum ControlFlow {
    Continue,
    Branch(BlockId),
    Return(Option<Value>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::FunctionBuilder;
    use crate::ir::types::common;

    #[test]
    fn test_simple_kernel() {
        // Create a simple kernel that adds 1 to each element
        let mut builder = FunctionBuilder::kernel("add_one");
        let data = builder.param("data", common::f32_ptr());
        let n = builder.param("n", common::u32());

        let idx = builder.global_thread_id(Dimension::X);
        let in_bounds = builder.lt(idx, n, ScalarType::U32);

        let then_block = builder.new_block();
        let exit_block = builder.new_block();

        builder.cond_branch(in_bounds, then_block, exit_block);

        builder.switch_to(then_block);
        let idx_i64 = builder.convert(idx, ScalarType::U32, ScalarType::I64);
        let ptr = builder.gep(data, vec![idx_i64], common::f32());
        let val = builder.load(ptr, common::f32(), AddressSpace::Global);
        let one = builder.const_f32(1.0);
        let result = builder.fadd(val, one, ScalarType::F32);
        builder.store(ptr, result, common::f32(), AddressSpace::Global);
        builder.branch(exit_block);

        builder.switch_to(exit_block);
        builder.ret_void();

        let func = builder.build();

        // Create test data
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        let n = data.len() as u32;

        // Execute
        let mut interp = CpuInterpreter::new();
        let config = LaunchConfig::for_elements(data.len(), 4);

        let args = vec![Value::Ptr(data.as_mut_ptr() as *mut u8), Value::U32(n)];

        interp.execute(&func, &config, &args).unwrap();

        // Verify results
        assert_eq!(data, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_value_conversion() {
        let v = Value::I32(42);
        assert_eq!(v.as_i32().unwrap(), 42);
        assert_eq!(v.as_u32().unwrap(), 42);
        assert_eq!(v.as_f32().unwrap(), 42.0);
    }
}
