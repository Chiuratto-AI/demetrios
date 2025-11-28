//! PTX (Parallel Thread Execution) Code Generator
//!
//! Generates NVIDIA PTX assembly from GPU IR.
//!
//! References:
//! - PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
//! - CUDA C Programming Guide

use std::fmt::Write;

use super::ir::*;

/// PTX code generator
pub struct PtxCodegen {
    /// Output buffer
    output: String,

    /// Target compute capability
    sm_version: (u32, u32),

    /// PTX version
    ptx_version: (u32, u32),

    /// Current indentation level
    indent: usize,

    /// Value to register mapping
    registers: Vec<String>,

    /// Next register number per type
    reg_counters: RegCounters,

    /// Type tracking for values
    value_types: Vec<GpuType>,
}

#[derive(Default)]
struct RegCounters {
    pred: u32, // Predicate registers
    b16: u32,  // 16-bit
    b32: u32,  // 32-bit
    b64: u32,  // 64-bit
    f32: u32,  // 32-bit float
    f64: u32,  // 64-bit float
}

impl PtxCodegen {
    pub fn new(sm_version: (u32, u32)) -> Self {
        Self {
            output: String::new(),
            sm_version,
            ptx_version: (8, 0),
            indent: 0,
            registers: Vec::new(),
            reg_counters: RegCounters::default(),
            value_types: Vec::new(),
        }
    }

    /// Generate PTX code from GPU module
    pub fn generate(&mut self, module: &GpuModule) -> String {
        self.output.clear();
        self.emit_header(module);

        // Emit constants
        for constant in &module.constants {
            self.emit_constant(constant);
        }

        // Emit device functions
        for (_, func) in &module.device_functions {
            self.emit_device_function(func);
        }

        // Emit kernels
        for (_, kernel) in &module.kernels {
            self.emit_kernel(kernel);
        }

        self.output.clone()
    }

    fn emit_header(&mut self, _module: &GpuModule) {
        writeln!(
            self.output,
            ".version {}.{}",
            self.ptx_version.0, self.ptx_version.1
        )
        .unwrap();

        writeln!(
            self.output,
            ".target sm_{}{}",
            self.sm_version.0, self.sm_version.1
        )
        .unwrap();

        writeln!(self.output, ".address_size 64").unwrap();

        writeln!(self.output).unwrap();
    }

    fn emit_kernel(&mut self, kernel: &GpuKernel) {
        // Reset registers
        self.registers.clear();
        self.reg_counters = RegCounters::default();
        self.value_types.clear();

        // Kernel entry
        writeln!(self.output, ".visible .entry {}(", kernel.name).unwrap();

        // Parameters
        for (i, param) in kernel.params.iter().enumerate() {
            let ptx_type = self.gpu_type_to_ptx(&param.ty);
            let comma = if i < kernel.params.len() - 1 { "," } else { "" };
            writeln!(
                self.output,
                "\t.param {} param_{}{}",
                ptx_type, param.name, comma
            )
            .unwrap();
        }

        writeln!(self.output, ")").unwrap();

        // Max threads hint
        if let Some(max_threads) = kernel.max_threads {
            writeln!(self.output, ".maxntid {}, 1, 1", max_threads).unwrap();
        }

        writeln!(self.output, "{{").unwrap();

        self.indent = 1;

        // Declare registers
        self.emit_register_declarations(kernel);

        // Shared memory declarations
        for shared in &kernel.shared_memory {
            self.emit_shared_memory(shared);
        }

        writeln!(self.output).unwrap();

        // Basic blocks
        for block in &kernel.blocks {
            self.emit_block(block);
        }

        self.indent = 0;
        writeln!(self.output, "}}").unwrap();
        writeln!(self.output).unwrap();
    }

    fn emit_device_function(&mut self, func: &GpuFunction) {
        self.registers.clear();
        self.reg_counters = RegCounters::default();
        self.value_types.clear();

        let ret_type = self.gpu_type_to_ptx(&func.return_type);

        if func.return_type != GpuType::Void {
            writeln!(self.output, ".func ({} retval) {}(", ret_type, func.name).unwrap();
        } else {
            writeln!(self.output, ".func {}(", func.name).unwrap();
        }

        for (i, param) in func.params.iter().enumerate() {
            let ptx_type = self.gpu_type_to_ptx(&param.ty);
            let comma = if i < func.params.len() - 1 { "," } else { "" };
            writeln!(
                self.output,
                "\t.param {} param_{}{}",
                ptx_type, param.name, comma
            )
            .unwrap();
        }

        writeln!(self.output, ")").unwrap();
        writeln!(self.output, "{{").unwrap();

        self.indent = 1;
        self.emit_register_declarations_func(func);

        for block in &func.blocks {
            self.emit_block(block);
        }

        self.indent = 0;
        writeln!(self.output, "}}").unwrap();
        writeln!(self.output).unwrap();
    }

    fn emit_block(&mut self, block: &GpuBlock) {
        // Block label
        writeln!(self.output, "{}:", block.label).unwrap();

        // Instructions
        for (value_id, op) in &block.instructions {
            self.emit_instruction(*value_id, op);
        }

        // Terminator
        self.emit_terminator(&block.terminator);
    }

    fn emit_instruction(&mut self, _value_id: ValueId, op: &GpuOp) {
        let indent = "\t".repeat(self.indent);

        match op {
            // Constants
            GpuOp::ConstInt(n, ty) => {
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(ty);
                writeln!(self.output, "{}mov.{} {}, {};", indent, suffix, reg, n).unwrap();
            }

            GpuOp::ConstFloat(n, ty) => {
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F32) {
                    "f32"
                } else {
                    "f64"
                };
                // Format float with proper PTX representation
                if n.is_nan() {
                    writeln!(self.output, "{}mov.{} {}, 0x7FC00000;", indent, suffix, reg).unwrap();
                } else if n.is_infinite() {
                    let val = if *n > 0.0 { "0x7F800000" } else { "0xFF800000" };
                    writeln!(self.output, "{}mov.{} {}, {};", indent, suffix, reg, val).unwrap();
                } else {
                    writeln!(
                        self.output,
                        "{}mov.{} {}, 0F{:08X};",
                        indent,
                        suffix,
                        reg,
                        (*n as f32).to_bits()
                    )
                    .unwrap();
                }
            }

            GpuOp::ConstBool(b) => {
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let val = if *b { 1 } else { 0 };
                writeln!(self.output, "{}setp.eq.u32 {}, {}, 1;", indent, reg, val).unwrap();
            }

            // Integer Arithmetic
            GpuOp::Add(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}add.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Sub(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}sub.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Mul(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}mul.lo.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Div(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}div.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Rem(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}rem.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Neg(val) => {
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(self.output, "{}neg.{} {}, {};", indent, suffix, reg, v).unwrap();
            }

            // Float arithmetic
            GpuOp::FAdd(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}add.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FSub(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}sub.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FMul(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}mul.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FDiv(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}div.approx.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FNeg(val) => {
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(self.output, "{}neg.{} {}, {};", indent, suffix, reg, v).unwrap();
            }

            GpuOp::FMulAdd(a, b, c) => {
                let ra = self.get_register(*a);
                let rb = self.get_register(*b);
                let rc = self.get_register(*c);
                let ty = self.get_value_type(*a);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}fma.rn.{} {}, {}, {}, {};",
                    indent, suffix, reg, ra, rb, rc
                )
                .unwrap();
            }

            // Fast math
            GpuOp::FastSin(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}sin.approx.f32 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FastCos(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}cos.approx.f32 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FastExp(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}ex2.approx.f32 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FastLog(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}lg2.approx.f32 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FastSqrt(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}sqrt.approx.f32 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FastRsqrt(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}rsqrt.approx.f32 {}, {};", indent, reg, v).unwrap();
            }

            // Integer Comparisons
            GpuOp::Lt(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}setp.lt.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Le(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}setp.le.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Gt(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}setp.gt.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Ge(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}setp.ge.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Eq(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}setp.eq.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Ne(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}setp.ne.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            // Float Comparisons
            GpuOp::FLt(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}setp.lt.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FLe(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}setp.le.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FGt(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}setp.gt.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FGe(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}setp.ge.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FEq(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}setp.eq.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FNe(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}setp.ne.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            // Logical operations
            GpuOp::And(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                writeln!(self.output, "{}and.pred {}, {}, {};", indent, reg, l, r).unwrap();
            }

            GpuOp::Or(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                writeln!(self.output, "{}or.pred {}, {}, {};", indent, reg, l, r).unwrap();
            }

            GpuOp::Xor(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                writeln!(self.output, "{}xor.pred {}, {}, {};", indent, reg, l, r).unwrap();
            }

            GpuOp::Not(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                writeln!(self.output, "{}not.pred {}, {};", indent, reg, v).unwrap();
            }

            // Bit operations
            GpuOp::Shl(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}shl.b{} {}, {}, {};",
                    indent, bits, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Shr(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}shr.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::LShr(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}shr.b{} {}, {}, {};",
                    indent, bits, reg, l, r
                )
                .unwrap();
            }

            GpuOp::BitAnd(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}and.b{} {}, {}, {};",
                    indent, bits, reg, l, r
                )
                .unwrap();
            }

            GpuOp::BitOr(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(self.output, "{}or.b{} {}, {}, {};", indent, bits, reg, l, r).unwrap();
            }

            GpuOp::BitXor(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}xor.b{} {}, {}, {};",
                    indent, bits, reg, l, r
                )
                .unwrap();
            }

            GpuOp::BitNot(val) => {
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(self.output, "{}not.b{} {}, {};", indent, bits, reg, v).unwrap();
            }

            GpuOp::PopCount(val) => {
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                let bits = ty.size_bytes() * 8;
                writeln!(self.output, "{}popc.b{} {}, {};", indent, bits, reg, v).unwrap();
            }

            GpuOp::Clz(val) => {
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                let bits = ty.size_bytes() * 8;
                writeln!(self.output, "{}clz.b{} {}, {};", indent, bits, reg, v).unwrap();
            }

            GpuOp::Ctz(val) => {
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                // PTX doesn't have ctz, emulate with bfind
                let bits = ty.size_bytes() * 8;
                writeln!(self.output, "{}bfind.u{} {}, {};", indent, bits, reg, v).unwrap();
            }

            // Conversions
            GpuOp::Trunc(val, ty) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = self.type_suffix(ty);
                writeln!(
                    self.output,
                    "{}cvt.{}.s64 {}, {};",
                    indent, dst_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::ZExt(val, ty) => {
                let v = self.get_register(*val);
                let src_ty = self.get_value_type(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = self.type_suffix(ty);
                let src_suffix = self.type_suffix(&src_ty);
                writeln!(
                    self.output,
                    "{}cvt.{}.{} {}, {};",
                    indent, dst_suffix, src_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::SExt(val, ty) => {
                let v = self.get_register(*val);
                let src_ty = self.get_value_type(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = self.type_suffix(ty);
                let src_suffix = self.type_suffix(&src_ty);
                writeln!(
                    self.output,
                    "{}cvt.{}.{} {}, {};",
                    indent, dst_suffix, src_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::FpTrunc(val, ty) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                writeln!(self.output, "{}cvt.rn.f32.f64 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FpExt(val, ty) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                writeln!(self.output, "{}cvt.f64.f32 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FpToSi(val, ty) => {
                let v = self.get_register(*val);
                let src_ty = self.get_value_type(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = self.type_suffix(ty);
                let src_suffix = if matches!(src_ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}cvt.rzi.{}.{} {}, {};",
                    indent, dst_suffix, src_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::FpToUi(val, ty) => {
                let v = self.get_register(*val);
                let src_ty = self.get_value_type(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = self.type_suffix(ty);
                let src_suffix = if matches!(src_ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}cvt.rzi.{}.{} {}, {};",
                    indent, dst_suffix, src_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::SiToFp(val, ty) => {
                let v = self.get_register(*val);
                let src_ty = self.get_value_type(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                let src_suffix = self.type_suffix(&src_ty);
                writeln!(
                    self.output,
                    "{}cvt.rn.{}.{} {}, {};",
                    indent, dst_suffix, src_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::UiToFp(val, ty) => {
                let v = self.get_register(*val);
                let src_ty = self.get_value_type(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                let src_suffix = self.type_suffix(&src_ty);
                writeln!(
                    self.output,
                    "{}cvt.rn.{}.{} {}, {};",
                    indent, dst_suffix, src_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::Bitcast(val, ty) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(self.output, "{}mov.b{} {}, {};", indent, bits, reg, v).unwrap();
            }

            // Memory operations
            GpuOp::Load(ptr, space) => {
                let p = self.get_register(*ptr);
                let reg = self.alloc_register(&GpuType::U64);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U64);
                let space_str = self.memory_space_to_ptx(*space);
                writeln!(
                    self.output,
                    "{}ld{}.u64 {}, [{}];",
                    indent, space_str, reg, p
                )
                .unwrap();
            }

            GpuOp::Store(ptr, val, space) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let space_str = self.memory_space_to_ptx(*space);
                self.registers.push("_".to_string()); // Dummy for void op
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}st{}.u64 [{}], {};", indent, space_str, p, v).unwrap();
            }

            // Atomic operations
            GpuOp::AtomicAdd(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}atom.global.add.{} {}, [{}], {};",
                    indent, suffix, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicSub(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let neg_reg = self.alloc_register(&ty);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                // Atomic sub via negation + add
                writeln!(self.output, "{}neg.{} {}, {};", indent, suffix, neg_reg, v).unwrap();
                writeln!(
                    self.output,
                    "{}atom.global.add.{} {}, [{}], {};",
                    indent, suffix, reg, p, neg_reg
                )
                .unwrap();
            }

            GpuOp::AtomicMin(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}atom.global.min.{} {}, [{}], {};",
                    indent, suffix, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicMax(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}atom.global.max.{} {}, [{}], {};",
                    indent, suffix, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicAnd(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}atom.global.and.b{} {}, [{}], {};",
                    indent, bits, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicOr(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}atom.global.or.b{} {}, [{}], {};",
                    indent, bits, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicXor(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}atom.global.xor.b{} {}, [{}], {};",
                    indent, bits, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicExch(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}atom.global.exch.b{} {}, [{}], {};",
                    indent, bits, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicCas(ptr, cmp, val) => {
                let p = self.get_register(*ptr);
                let c = self.get_register(*cmp);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}atom.global.cas.b{} {}, [{}], {}, {};",
                    indent, bits, reg, p, c, v
                )
                .unwrap();
            }

            // Address computation
            GpuOp::GetElementPtr(ptr, indices) => {
                let p = self.get_register(*ptr);
                let reg = self.alloc_register(&GpuType::U64);
                self.registers.push(reg.clone());
                self.value_types
                    .push(GpuType::Ptr(Box::new(GpuType::U8), MemorySpace::Global));
                // Simple offset calculation
                if indices.is_empty() {
                    writeln!(self.output, "{}mov.u64 {}, {};", indent, reg, p).unwrap();
                } else {
                    let idx = self.get_register(indices[0]);
                    writeln!(self.output, "{}add.u64 {}, {}, {};", indent, reg, p, idx).unwrap();
                }
            }

            GpuOp::PtrToInt(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::U64);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U64);
                writeln!(self.output, "{}mov.u64 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::IntToPtr(val, ty) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                writeln!(self.output, "{}mov.u64 {}, {};", indent, reg, v).unwrap();
            }

            // GPU intrinsics
            GpuOp::ThreadIdX => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %tid.x;", indent, reg).unwrap();
            }

            GpuOp::ThreadIdY => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %tid.y;", indent, reg).unwrap();
            }

            GpuOp::ThreadIdZ => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %tid.z;", indent, reg).unwrap();
            }

            GpuOp::BlockIdX => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %ctaid.x;", indent, reg).unwrap();
            }

            GpuOp::BlockIdY => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %ctaid.y;", indent, reg).unwrap();
            }

            GpuOp::BlockIdZ => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %ctaid.z;", indent, reg).unwrap();
            }

            GpuOp::BlockDimX => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %ntid.x;", indent, reg).unwrap();
            }

            GpuOp::BlockDimY => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %ntid.y;", indent, reg).unwrap();
            }

            GpuOp::BlockDimZ => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %ntid.z;", indent, reg).unwrap();
            }

            GpuOp::GridDimX => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %nctaid.x;", indent, reg).unwrap();
            }

            GpuOp::GridDimY => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %nctaid.y;", indent, reg).unwrap();
            }

            GpuOp::GridDimZ => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %nctaid.z;", indent, reg).unwrap();
            }

            GpuOp::WarpId => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %warpid;", indent, reg).unwrap();
            }

            GpuOp::LaneId => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %laneid;", indent, reg).unwrap();
            }

            GpuOp::WarpSize => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, WARP_SZ;", indent, reg).unwrap();
            }

            // Synchronization
            GpuOp::SyncThreads => {
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}bar.sync 0;", indent).unwrap();
            }

            GpuOp::SyncWarp(mask) => {
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}bar.warp.sync 0x{:08x};", indent, mask).unwrap();
            }

            GpuOp::MemoryFence(space) => {
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                let fence_type = match space {
                    MemorySpace::Global => "membar.gl;",
                    MemorySpace::Shared => "membar.cta;",
                    _ => "membar.sys;",
                };
                writeln!(self.output, "{}{}", indent, fence_type).unwrap();
            }

            // Warp operations
            GpuOp::WarpShuffle(val, lane) => {
                let v = self.get_register(*val);
                let l = self.get_register(*lane);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);
                writeln!(
                    self.output,
                    "{}shfl.sync.idx.b32 {}, {}, {}, 31, 0xffffffff;",
                    indent, reg, v, l
                )
                .unwrap();
            }

            GpuOp::WarpShuffleUp(val, delta) => {
                let v = self.get_register(*val);
                let d = self.get_register(*delta);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);
                writeln!(
                    self.output,
                    "{}shfl.sync.up.b32 {}, {}, {}, 0, 0xffffffff;",
                    indent, reg, v, d
                )
                .unwrap();
            }

            GpuOp::WarpShuffleDown(val, delta) => {
                let v = self.get_register(*val);
                let d = self.get_register(*delta);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);
                writeln!(
                    self.output,
                    "{}shfl.sync.down.b32 {}, {}, {}, 31, 0xffffffff;",
                    indent, reg, v, d
                )
                .unwrap();
            }

            GpuOp::WarpShuffleXor(val, mask) => {
                let v = self.get_register(*val);
                let m = self.get_register(*mask);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);
                writeln!(
                    self.output,
                    "{}shfl.sync.bfly.b32 {}, {}, {}, 31, 0xffffffff;",
                    indent, reg, v, m
                )
                .unwrap();
            }

            GpuOp::WarpVote(vote_op, val) => {
                let v = self.get_register(*val);
                let reg = match vote_op {
                    WarpVoteOp::Ballot => {
                        let r = self.alloc_register(&GpuType::U32);
                        writeln!(
                            self.output,
                            "{}vote.sync.ballot.b32 {}, {}, 0xffffffff;",
                            indent, r, v
                        )
                        .unwrap();
                        self.value_types.push(GpuType::U32);
                        r
                    }
                    WarpVoteOp::All => {
                        let r = self.alloc_pred_register();
                        writeln!(
                            self.output,
                            "{}vote.sync.all.pred {}, {}, 0xffffffff;",
                            indent, r, v
                        )
                        .unwrap();
                        self.value_types.push(GpuType::Bool);
                        r
                    }
                    WarpVoteOp::Any => {
                        let r = self.alloc_pred_register();
                        writeln!(
                            self.output,
                            "{}vote.sync.any.pred {}, {}, 0xffffffff;",
                            indent, r, v
                        )
                        .unwrap();
                        self.value_types.push(GpuType::Bool);
                        r
                    }
                    WarpVoteOp::Eq => {
                        let r = self.alloc_pred_register();
                        writeln!(
                            self.output,
                            "{}vote.sync.uni.pred {}, {}, 0xffffffff;",
                            indent, r, v
                        )
                        .unwrap();
                        self.value_types.push(GpuType::Bool);
                        r
                    }
                };
                self.registers.push(reg);
            }

            GpuOp::WarpReduce(reduce_op, val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);
                let op_name = match reduce_op {
                    WarpReduceOp::Add => "add",
                    WarpReduceOp::Min => "min",
                    WarpReduceOp::Max => "max",
                    WarpReduceOp::And => "and",
                    WarpReduceOp::Or => "or",
                    WarpReduceOp::Xor => "xor",
                };
                writeln!(
                    self.output,
                    "{}redux.sync.{}.s32 {}, {}, 0xffffffff;",
                    indent, op_name, reg, v
                )
                .unwrap();
            }

            GpuOp::WarpMatch(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(
                    self.output,
                    "{}match.sync.any.b32 {}, {}, 0xffffffff;",
                    indent, reg, v
                )
                .unwrap();
            }

            // Texture operations (simplified)
            GpuOp::TexFetch(tex, coord) => {
                let _t = self.get_register(*tex);
                let _c = self.get_register(*coord);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}// tex.1d.v4.f32 not implemented", indent).unwrap();
            }

            GpuOp::TexFetch2D(tex, x, y) => {
                let _t = self.get_register(*tex);
                let _xr = self.get_register(*x);
                let _yr = self.get_register(*y);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}// tex.2d.v4.f32 not implemented", indent).unwrap();
            }

            GpuOp::SurfRead(surf, coord) => {
                let _s = self.get_register(*surf);
                let _c = self.get_register(*coord);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}// suld.b.1d not implemented", indent).unwrap();
            }

            GpuOp::SurfWrite(surf, coord, val) => {
                let _s = self.get_register(*surf);
                let _c = self.get_register(*coord);
                let _v = self.get_register(*val);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}// sust.b.1d not implemented", indent).unwrap();
            }

            // Select
            GpuOp::Select(cond, t, f) => {
                let c = self.get_register(*cond);
                let tv = self.get_register(*t);
                let fv = self.get_register(*f);
                let ty = self.get_value_type(*t);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}selp.{} {}, {}, {}, {};",
                    indent, suffix, reg, tv, fv, c
                )
                .unwrap();
            }

            // Function call
            GpuOp::Call(name, args) => {
                let arg_regs: Vec<_> = args
                    .iter()
                    .map(|a| self.get_register(*a).to_string())
                    .collect();
                let reg = self.alloc_register(&GpuType::I64);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I64);

                writeln!(self.output, "{}{{", indent).unwrap();
                writeln!(self.output, "{}\t.param .b64 retval;", indent).unwrap();
                for (i, arg) in arg_regs.iter().enumerate() {
                    writeln!(self.output, "{}\t.param .b64 arg{};", indent, i).unwrap();
                    writeln!(self.output, "{}\tst.param.b64 [arg{}], {};", indent, i, arg).unwrap();
                }
                write!(self.output, "{}\tcall (retval), {}, (", indent, name).unwrap();
                for (i, _) in args.iter().enumerate() {
                    if i > 0 {
                        write!(self.output, ", ").unwrap();
                    }
                    write!(self.output, "arg{}", i).unwrap();
                }
                writeln!(self.output, ");").unwrap();
                writeln!(self.output, "{}\tld.param.b64 {}, [retval];", indent, reg).unwrap();
                writeln!(self.output, "{}}}", indent).unwrap();
            }

            // Parameter
            GpuOp::Param(idx) => {
                let reg = self.alloc_register(&GpuType::U64);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U64);
                writeln!(
                    self.output,
                    "{}ld.param.u64 {}, [param_{}];",
                    indent, reg, idx
                )
                .unwrap();
            }

            // Shared memory address
            GpuOp::SharedAddr(name) => {
                let reg = self.alloc_register(&GpuType::U64);
                self.registers.push(reg.clone());
                self.value_types
                    .push(GpuType::Ptr(Box::new(GpuType::U8), MemorySpace::Shared));
                writeln!(self.output, "{}mov.u64 {}, {};", indent, reg, name).unwrap();
            }

            // Phi (should be lowered before PTX emission)
            GpuOp::Phi(_) => {
                // Phi nodes should be eliminated before PTX generation
                self.registers.push("phi_placeholder".to_string());
                self.value_types.push(GpuType::I64);
            }
        }
    }

    fn emit_terminator(&mut self, term: &GpuTerminator) {
        let indent = "\t".repeat(self.indent);

        match term {
            GpuTerminator::Br(target) => {
                writeln!(self.output, "{}bra BB{};", indent, target.0).unwrap();
            }

            GpuTerminator::CondBr(cond, then_block, else_block) => {
                let c = self.get_register(*cond);
                writeln!(self.output, "{}@{} bra BB{};", indent, c, then_block.0).unwrap();
                writeln!(self.output, "{}bra BB{};", indent, else_block.0).unwrap();
            }

            GpuTerminator::ReturnVoid => {
                writeln!(self.output, "{}ret;", indent).unwrap();
            }

            GpuTerminator::Return(val) => {
                let v = self.get_register(*val);
                writeln!(self.output, "{}st.param.b64 [retval], {};", indent, v).unwrap();
                writeln!(self.output, "{}ret;", indent).unwrap();
            }

            GpuTerminator::Unreachable => {
                writeln!(self.output, "{}trap;", indent).unwrap();
            }
        }
    }

    fn emit_shared_memory(&mut self, shared: &SharedMemDecl) {
        let indent = "\t".repeat(self.indent);
        let ptx_type = self.gpu_type_to_ptx(&shared.elem_type);

        writeln!(
            self.output,
            "{}.shared .align {} {} {}[{}];",
            indent, shared.align, ptx_type, shared.name, shared.size
        )
        .unwrap();
    }

    fn emit_register_declarations(&mut self, _kernel: &GpuKernel) {
        let indent = "\t".repeat(self.indent);

        writeln!(self.output, "{}// Register declarations", indent).unwrap();
        writeln!(self.output, "{}.reg .pred p<64>;", indent).unwrap();
        writeln!(self.output, "{}.reg .b16 r16_<64>;", indent).unwrap();
        writeln!(self.output, "{}.reg .b32 r32_<128>;", indent).unwrap();
        writeln!(self.output, "{}.reg .b64 r64_<128>;", indent).unwrap();
        writeln!(self.output, "{}.reg .f32 f32_<128>;", indent).unwrap();
        writeln!(self.output, "{}.reg .f64 f64_<64>;", indent).unwrap();
    }

    fn emit_register_declarations_func(&mut self, _func: &GpuFunction) {
        let indent = "\t".repeat(self.indent);

        writeln!(self.output, "{}.reg .pred p<64>;", indent).unwrap();
        writeln!(self.output, "{}.reg .b16 r16_<64>;", indent).unwrap();
        writeln!(self.output, "{}.reg .b32 r32_<128>;", indent).unwrap();
        writeln!(self.output, "{}.reg .b64 r64_<128>;", indent).unwrap();
        writeln!(self.output, "{}.reg .f32 f32_<128>;", indent).unwrap();
        writeln!(self.output, "{}.reg .f64 f64_<64>;", indent).unwrap();
        writeln!(self.output).unwrap();
    }

    fn emit_constant(&mut self, constant: &GpuConstant) {
        let ptx_type = self.gpu_type_to_ptx(&constant.ty);

        write!(self.output, ".const {} {} = ", ptx_type, constant.name).unwrap();
        self.emit_const_value(&constant.value);
        writeln!(self.output, ";").unwrap();
    }

    fn emit_const_value(&mut self, value: &GpuConstValue) {
        match value {
            GpuConstValue::Int(n) => write!(self.output, "{}", n).unwrap(),
            GpuConstValue::Float(n) => write!(self.output, "{:.15e}", n).unwrap(),
            GpuConstValue::Bool(b) => write!(self.output, "{}", if *b { 1 } else { 0 }).unwrap(),
            GpuConstValue::Array(elems) => {
                write!(self.output, "{{").unwrap();
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(self.output, ", ").unwrap();
                    }
                    self.emit_const_value(elem);
                }
                write!(self.output, "}}").unwrap();
            }
            GpuConstValue::Struct(fields) => {
                write!(self.output, "{{").unwrap();
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(self.output, ", ").unwrap();
                    }
                    self.emit_const_value(field);
                }
                write!(self.output, "}}").unwrap();
            }
        }
    }

    fn alloc_register(&mut self, ty: &GpuType) -> String {
        match ty {
            GpuType::I16 | GpuType::U16 | GpuType::F16 => {
                let n = self.reg_counters.b16;
                self.reg_counters.b16 += 1;
                format!("r16_{}", n)
            }
            GpuType::I32 | GpuType::U32 => {
                let n = self.reg_counters.b32;
                self.reg_counters.b32 += 1;
                format!("r32_{}", n)
            }
            GpuType::I64 | GpuType::U64 | GpuType::Ptr(_, _) => {
                let n = self.reg_counters.b64;
                self.reg_counters.b64 += 1;
                format!("r64_{}", n)
            }
            GpuType::F32 => {
                let n = self.reg_counters.f32;
                self.reg_counters.f32 += 1;
                format!("f32_{}", n)
            }
            GpuType::F64 => {
                let n = self.reg_counters.f64;
                self.reg_counters.f64 += 1;
                format!("f64_{}", n)
            }
            _ => {
                let n = self.reg_counters.b64;
                self.reg_counters.b64 += 1;
                format!("r64_{}", n)
            }
        }
    }

    fn alloc_pred_register(&mut self) -> String {
        let n = self.reg_counters.pred;
        self.reg_counters.pred += 1;
        format!("p{}", n)
    }

    fn get_register(&self, id: ValueId) -> String {
        self.registers[id.0 as usize].clone()
    }

    fn get_value_type(&self, id: ValueId) -> GpuType {
        self.value_types
            .get(id.0 as usize)
            .cloned()
            .unwrap_or(GpuType::I64)
    }

    fn gpu_type_to_ptx(&self, ty: &GpuType) -> &'static str {
        match ty {
            GpuType::Void => ".b32",
            GpuType::Bool => ".pred",
            GpuType::I8 | GpuType::U8 => ".b8",
            GpuType::I16 | GpuType::U16 => ".b16",
            GpuType::I32 | GpuType::U32 => ".b32",
            GpuType::I64 | GpuType::U64 => ".b64",
            GpuType::F16 => ".f16",
            GpuType::F32 => ".f32",
            GpuType::F64 => ".f64",
            GpuType::Ptr(_, _) => ".b64",
            _ => ".b64",
        }
    }

    fn type_suffix(&self, ty: &GpuType) -> &'static str {
        match ty {
            GpuType::I8 => "s8",
            GpuType::I16 => "s16",
            GpuType::I32 => "s32",
            GpuType::I64 => "s64",
            GpuType::U8 => "u8",
            GpuType::U16 => "u16",
            GpuType::U32 => "u32",
            GpuType::U64 => "u64",
            GpuType::F32 => "f32",
            GpuType::F64 => "f64",
            _ => "b64",
        }
    }

    fn memory_space_to_ptx(&self, space: MemorySpace) -> &'static str {
        match space {
            MemorySpace::Global => ".global",
            MemorySpace::Shared => ".shared",
            MemorySpace::Local => ".local",
            MemorySpace::Constant => ".const",
            MemorySpace::Generic => "",
            MemorySpace::Texture => ".tex",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_header() {
        let mut codegen = PtxCodegen::new((7, 5));
        let module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );
        let ptx = codegen.generate(&module);

        assert!(ptx.contains(".version 8.0"));
        assert!(ptx.contains(".target sm_75"));
        assert!(ptx.contains(".address_size 64"));
    }

    #[test]
    fn test_ptx_simple_kernel() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );

        let mut kernel = GpuKernel::new("add_one");
        kernel.add_param(GpuParam {
            name: "data".to_string(),
            ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });

        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ThreadIdX);
        block.add_instruction(ValueId(1), GpuOp::BlockIdX);
        block.add_instruction(ValueId(2), GpuOp::BlockDimX);
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains(".visible .entry add_one"));
        assert!(ptx.contains("%tid.x"));
        assert!(ptx.contains("%ctaid.x"));
        assert!(ptx.contains("%ntid.x"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn test_ptx_shared_memory() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );

        let mut kernel = GpuKernel::new("reduce");
        kernel.add_shared_memory(SharedMemDecl {
            name: "cache".to_string(),
            elem_type: GpuType::F32,
            size: 256,
            align: 4,
        });

        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::SyncThreads);
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains(".shared"));
        assert!(ptx.contains("cache"));
        assert!(ptx.contains("bar.sync 0"));
    }

    #[test]
    fn test_ptx_arithmetic() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );

        let mut kernel = GpuKernel::new("math");

        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstInt(10, GpuType::I32));
        block.add_instruction(ValueId(1), GpuOp::ConstInt(20, GpuType::I32));
        block.add_instruction(ValueId(2), GpuOp::Add(ValueId(0), ValueId(1)));
        block.add_instruction(ValueId(3), GpuOp::ConstFloat(3.14, GpuType::F32));
        block.add_instruction(ValueId(4), GpuOp::ConstFloat(2.0, GpuType::F32));
        block.add_instruction(ValueId(5), GpuOp::FMul(ValueId(3), ValueId(4)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("add.s32"));
        assert!(ptx.contains("mul.f32"));
    }
}
