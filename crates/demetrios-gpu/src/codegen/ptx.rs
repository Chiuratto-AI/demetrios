//! PTX Code Generation
//!
//! Generates NVIDIA PTX assembly from GPU IR.

use crate::ir::*;
use std::fmt::Write;
use thiserror::Error;

/// PTX generation errors
#[derive(Debug, Error)]
pub enum PtxError {
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    #[error("Invalid IR: {0}")]
    InvalidIr(String),

    #[error("Formatting error: {0}")]
    FormatError(#[from] std::fmt::Error),
}

/// PTX code generator
pub struct PtxGenerator {
    /// Output buffer
    output: String,
    /// PTX version
    ptx_version: (u32, u32),
    /// SM version
    sm_version: u32,
    /// Register counter for temps
    reg_counter: u32,
    /// Indentation level
    indent: usize,
}

impl PtxGenerator {
    /// Create a new PTX generator
    pub fn new() -> Self {
        PtxGenerator {
            output: String::new(),
            ptx_version: (8, 0),
            sm_version: 75,
            reg_counter: 0,
            indent: 0,
        }
    }

    /// Set PTX version
    pub fn with_ptx_version(mut self, major: u32, minor: u32) -> Self {
        self.ptx_version = (major, minor);
        self
    }

    /// Set SM version
    pub fn with_sm_version(mut self, version: u32) -> Self {
        self.sm_version = version;
        self
    }

    /// Generate PTX for a module
    pub fn generate_module(&mut self, module: &GpuModule) -> Result<String, PtxError> {
        self.output.clear();

        // Header
        self.emit_header(module)?;

        // Global constants
        for constant in &module.constants {
            self.emit_global_constant(constant)?;
        }

        // Functions
        for func in module.functions.values() {
            self.emit_function(func)?;
        }

        Ok(self.output.clone())
    }

    /// Generate PTX for a single function
    pub fn generate_function(&mut self, func: &GpuFunction) -> Result<String, PtxError> {
        self.output.clear();
        self.emit_function(func)?;
        Ok(self.output.clone())
    }

    fn emit_header(&mut self, module: &GpuModule) -> Result<(), PtxError> {
        writeln!(
            self.output,
            ".version {}.{}",
            module.ptx_version.0, module.ptx_version.1
        )?;
        writeln!(self.output, ".target sm_{}", module.sm_version)?;
        writeln!(self.output, ".address_size 64")?;
        writeln!(self.output)?;
        Ok(())
    }

    fn emit_global_constant(
        &mut self,
        constant: &function::GlobalConstant,
    ) -> Result<(), PtxError> {
        write!(
            self.output,
            ".const .align 16 .b8 {}[{}]",
            constant.name,
            constant.data.len()
        )?;

        if !constant.data.is_empty() {
            write!(self.output, " = {{")?;
            for (i, byte) in constant.data.iter().enumerate() {
                if i > 0 {
                    write!(self.output, ", ")?;
                }
                write!(self.output, "{}", byte)?;
            }
            write!(self.output, "}}")?;
        }

        writeln!(self.output, ";")?;
        Ok(())
    }

    fn emit_function(&mut self, func: &GpuFunction) -> Result<(), PtxError> {
        // Function directive
        let directive = func.kind.ptx_directive();
        write!(self.output, "{} ", directive)?;

        // Return type
        if let Some(ref ret_ty) = func.return_type {
            write!(self.output, "({}) ", self.type_to_ptx(ret_ty))?;
        }

        // Function name and parameters
        write!(self.output, "{}(", func.name)?;
        for (i, param) in func.params.iter().enumerate() {
            if i > 0 {
                write!(self.output, ", ")?;
            }
            write!(
                self.output,
                ".param {} {}",
                self.type_to_ptx_param(&param.ty),
                param.name
            )?;
        }
        writeln!(self.output, ")")?;

        // Performance hints
        if let Some(max_threads) = func.max_threads {
            writeln!(self.output, ".maxntid {}", max_threads)?;
        }
        if let Some(min_blocks) = func.min_blocks {
            writeln!(self.output, ".minnctapersm {}", min_blocks)?;
        }

        writeln!(self.output, "{{")?;
        self.indent += 1;

        // Register declarations
        self.emit_register_decls(func)?;

        // Shared memory declarations
        for shared in &func.shared_mem {
            self.emit_indent()?;
            writeln!(
                self.output,
                ".shared .align 16 .b8 {}[{}];",
                shared.name,
                shared.ty.size_bytes().unwrap_or(0) * shared.size
            )?;
        }

        // Parameter loading
        for (i, param) in func.params.iter().enumerate() {
            self.emit_param_load(i, param)?;
        }

        writeln!(self.output)?;

        // Basic blocks
        for (block_id, block) in &func.blocks {
            self.emit_block(block_id, block)?;
        }

        self.indent -= 1;
        writeln!(self.output, "}}")?;
        writeln!(self.output)?;

        Ok(())
    }

    fn emit_register_decls(&mut self, func: &GpuFunction) -> Result<(), PtxError> {
        // Count registers needed per type
        let mut pred_count = 0u32;
        let mut b16_count = 0u32;
        let mut b32_count = 0u32;
        let mut b64_count = 0u32;
        let mut f32_count = 0u32;
        let mut f64_count = 0u32;

        for (_, ty) in &func.value_types {
            match ty {
                GpuType::Scalar(ScalarType::Bool) => pred_count += 1,
                GpuType::Scalar(
                    ScalarType::I16 | ScalarType::U16 | ScalarType::F16 | ScalarType::BF16,
                ) => b16_count += 1,
                GpuType::Scalar(ScalarType::I32 | ScalarType::U32) => b32_count += 1,
                GpuType::Scalar(ScalarType::I64 | ScalarType::U64) | GpuType::Ptr(_, _) => {
                    b64_count += 1
                }
                GpuType::Scalar(ScalarType::F32) => f32_count += 1,
                GpuType::Scalar(ScalarType::F64) => f64_count += 1,
                GpuType::Scalar(ScalarType::I8 | ScalarType::U8) => b16_count += 1, // Use 16-bit for 8-bit
                _ => {}
            }
        }

        // Emit register declarations
        if pred_count > 0 {
            self.emit_indent()?;
            writeln!(self.output, ".reg .pred %p<{}>;", pred_count + 1)?;
        }
        if b16_count > 0 {
            self.emit_indent()?;
            writeln!(self.output, ".reg .b16 %h<{}>;", b16_count + 1)?;
        }
        if b32_count > 0 {
            self.emit_indent()?;
            writeln!(self.output, ".reg .b32 %r<{}>;", b32_count + 1)?;
        }
        if b64_count > 0 {
            self.emit_indent()?;
            writeln!(self.output, ".reg .b64 %rd<{}>;", b64_count + 1)?;
        }
        if f32_count > 0 {
            self.emit_indent()?;
            writeln!(self.output, ".reg .f32 %f<{}>;", f32_count + 1)?;
        }
        if f64_count > 0 {
            self.emit_indent()?;
            writeln!(self.output, ".reg .f64 %fd<{}>;", f64_count + 1)?;
        }

        writeln!(self.output)?;
        Ok(())
    }

    fn emit_param_load(&mut self, index: usize, param: &Parameter) -> Result<(), PtxError> {
        self.emit_indent()?;

        let reg = self.value_to_reg(ValueId(index as u32), &param.ty);
        let ptx_ty = self.type_to_ptx_load(&param.ty);

        writeln!(
            self.output,
            "ld.param.{} {}, [{}];",
            ptx_ty, reg, param.name
        )?;

        Ok(())
    }

    fn emit_block(&mut self, block_id: &BlockId, block: &BasicBlock) -> Result<(), PtxError> {
        // Block label
        if let Some(ref label) = block.label {
            writeln!(self.output, "{}:", label)?;
        } else {
            writeln!(self.output, "BB{}:", block_id.0)?;
        }

        // Instructions
        for inst in &block.instructions {
            self.emit_instruction(inst)?;
        }

        Ok(())
    }

    fn emit_instruction(&mut self, inst: &Instruction) -> Result<(), PtxError> {
        self.emit_indent()?;

        match inst {
            Instruction::Const { dst, value } => {
                let ty = value.gpu_type();
                let reg = self.value_to_reg(*dst, &ty);
                writeln!(
                    self.output,
                    "mov.{} {}, {};",
                    value.scalar_type().ptx_suffix(),
                    reg,
                    value
                )?;
            }

            Instruction::BinOp {
                dst,
                op,
                lhs,
                rhs,
                ty,
            } => {
                let dst_ty = GpuType::Scalar(*ty);
                let dst_reg = self.value_to_reg(*dst, &dst_ty);
                let lhs_reg = self.value_to_reg(*lhs, &dst_ty);
                let rhs_reg = self.value_to_reg(*rhs, &dst_ty);

                let suffix = ty.ptx_suffix();
                writeln!(
                    self.output,
                    "{}.{} {}, {}, {};",
                    op.ptx_name(),
                    suffix,
                    dst_reg,
                    lhs_reg,
                    rhs_reg
                )?;
            }

            Instruction::UnaryOp { dst, op, src, ty } => {
                let gpu_ty = GpuType::Scalar(*ty);
                let dst_reg = self.value_to_reg(*dst, &gpu_ty);
                let src_reg = self.value_to_reg(*src, &gpu_ty);

                let suffix = ty.ptx_suffix();
                writeln!(
                    self.output,
                    "{}.{} {}, {};",
                    op.ptx_name(),
                    suffix,
                    dst_reg,
                    src_reg
                )?;
            }

            Instruction::Cmp {
                dst,
                op,
                lhs,
                rhs,
                ty,
            } => {
                let src_ty = GpuType::Scalar(*ty);
                let dst_reg = self.value_to_reg(*dst, &GpuType::Scalar(ScalarType::Bool));
                let lhs_reg = self.value_to_reg(*lhs, &src_ty);
                let rhs_reg = self.value_to_reg(*rhs, &src_ty);

                let cmp_suffix = op.ptx_suffix(ty.is_signed());
                let ty_suffix = ty.ptx_suffix();
                writeln!(
                    self.output,
                    "setp.{}.{} {}, {}, {};",
                    cmp_suffix, ty_suffix, dst_reg, lhs_reg, rhs_reg
                )?;
            }

            Instruction::Convert { dst, src, from, to } => {
                let dst_ty = GpuType::Scalar(*to);
                let src_ty = GpuType::Scalar(*from);
                let dst_reg = self.value_to_reg(*dst, &dst_ty);
                let src_reg = self.value_to_reg(*src, &src_ty);

                // Determine conversion instruction
                let (inst, round) = if from.is_float() && to.is_float() {
                    ("cvt", ".rn")
                } else if from.is_float() && to.is_integer() {
                    ("cvt", ".rzi")
                } else if from.is_integer() && to.is_float() {
                    ("cvt", ".rn")
                } else {
                    ("cvt", "")
                };

                writeln!(
                    self.output,
                    "{}{}.{}.{} {}, {};",
                    inst,
                    round,
                    to.ptx_suffix(),
                    from.ptx_suffix(),
                    dst_reg,
                    src_reg
                )?;
            }

            Instruction::Load {
                dst,
                ptr,
                ty,
                space,
                volatile,
            } => {
                let dst_reg = self.value_to_reg(*dst, ty);
                let ptr_reg = self.value_to_reg(*ptr, &GpuType::Ptr(Box::new(ty.clone()), *space));

                let vol = if *volatile { ".volatile" } else { "" };
                let space_str = if *space == AddressSpace::Global {
                    ".global".to_string()
                } else {
                    format!(".{}", space.ptx_name())
                };
                let ty_str = self.type_to_ptx_load(ty);

                writeln!(
                    self.output,
                    "ld{}{}.{} {}, [{}];",
                    vol, space_str, ty_str, dst_reg, ptr_reg
                )?;
            }

            Instruction::Store {
                ptr,
                value,
                ty,
                space,
                volatile,
            } => {
                let ptr_reg = self.value_to_reg(*ptr, &GpuType::Ptr(Box::new(ty.clone()), *space));
                let val_reg = self.value_to_reg(*value, ty);

                let vol = if *volatile { ".volatile" } else { "" };
                let space_str = if *space == AddressSpace::Global {
                    ".global".to_string()
                } else {
                    format!(".{}", space.ptx_name())
                };
                let ty_str = self.type_to_ptx_load(ty);

                writeln!(
                    self.output,
                    "st{}{}.{} [{}], {};",
                    vol, space_str, ty_str, ptr_reg, val_reg
                )?;
            }

            Instruction::Atomic {
                dst,
                op,
                ptr,
                value,
                ty,
                space,
                ..
            } => {
                let dst_ty = GpuType::Scalar(*ty);
                let dst_reg = self.value_to_reg(*dst, &dst_ty);
                let ptr_reg =
                    self.value_to_reg(*ptr, &GpuType::Ptr(Box::new(dst_ty.clone()), *space));
                let val_reg = self.value_to_reg(*value, &dst_ty);

                let space_str = space.ptx_name();
                writeln!(
                    self.output,
                    "atom.{}.{}.{} {}, [{}], {};",
                    space_str,
                    op.ptx_name(),
                    ty.ptx_suffix(),
                    dst_reg,
                    ptr_reg,
                    val_reg
                )?;
            }

            Instruction::GetElementPtr {
                dst,
                base,
                indices,
                pointee_ty,
            } => {
                let dst_reg = self.value_to_reg(
                    *dst,
                    &GpuType::Ptr(Box::new(pointee_ty.clone()), AddressSpace::Global),
                );
                let base_reg = self.value_to_reg(
                    *base,
                    &GpuType::Ptr(Box::new(pointee_ty.clone()), AddressSpace::Global),
                );

                if let Some(idx) = indices.first() {
                    let idx_reg = self.value_to_reg(*idx, &GpuType::Scalar(ScalarType::I64));
                    let elem_size = pointee_ty.size_bytes().unwrap_or(1);

                    // offset = idx * elem_size
                    writeln!(
                        self.output,
                        "mul.wide.s32 %rd_tmp, {}, {};",
                        idx_reg, elem_size
                    )?;
                    self.emit_indent()?;
                    writeln!(self.output, "add.s64 {}, {}, %rd_tmp;", dst_reg, base_reg)?;
                } else {
                    writeln!(self.output, "mov.u64 {}, {};", dst_reg, base_reg)?;
                }
            }

            Instruction::Branch { target } => {
                writeln!(self.output, "bra BB{};", target.0)?;
            }

            Instruction::CondBranch {
                cond,
                true_target,
                false_target,
            } => {
                let cond_reg = self.value_to_reg(*cond, &GpuType::Scalar(ScalarType::Bool));
                writeln!(self.output, "@{} bra BB{};", cond_reg, true_target.0)?;
                self.emit_indent()?;
                writeln!(self.output, "bra BB{};", false_target.0)?;
            }

            Instruction::Return { value } => {
                if let Some(val) = value {
                    // Store return value
                    writeln!(self.output, "// return value: %{}", val.0)?;
                }
                writeln!(self.output, "ret;")?;
            }

            Instruction::Barrier { scope } => match scope {
                BarrierScope::Block => writeln!(self.output, "bar.sync 0;")?,
                BarrierScope::Device => writeln!(self.output, "membar.gl;")?,
                BarrierScope::System => writeln!(self.output, "membar.sys;")?,
            },

            Instruction::MemFence { scope, .. } => match scope {
                BarrierScope::Block => writeln!(self.output, "membar.cta;")?,
                BarrierScope::Device => writeln!(self.output, "membar.gl;")?,
                BarrierScope::System => writeln!(self.output, "membar.sys;")?,
            },

            Instruction::ThreadIdx { dst, dim } => {
                let reg = self.value_to_reg(*dst, &GpuType::Scalar(ScalarType::U32));
                writeln!(self.output, "mov.u32 {}, %tid.{};", reg, dim.ptx_suffix())?;
            }

            Instruction::BlockIdx { dst, dim } => {
                let reg = self.value_to_reg(*dst, &GpuType::Scalar(ScalarType::U32));
                writeln!(self.output, "mov.u32 {}, %ctaid.{};", reg, dim.ptx_suffix())?;
            }

            Instruction::BlockDim { dst, dim } => {
                let reg = self.value_to_reg(*dst, &GpuType::Scalar(ScalarType::U32));
                writeln!(self.output, "mov.u32 {}, %ntid.{};", reg, dim.ptx_suffix())?;
            }

            Instruction::GridDim { dst, dim } => {
                let reg = self.value_to_reg(*dst, &GpuType::Scalar(ScalarType::U32));
                writeln!(
                    self.output,
                    "mov.u32 {}, %nctaid.{};",
                    reg,
                    dim.ptx_suffix()
                )?;
            }

            Instruction::Select {
                dst,
                cond,
                true_val,
                false_val,
            } => {
                // Get the type from true_val (assuming same as false_val)
                let ty = GpuType::Scalar(ScalarType::U32); // Default, should be inferred
                let dst_reg = self.value_to_reg(*dst, &ty);
                let cond_reg = self.value_to_reg(*cond, &GpuType::Scalar(ScalarType::Bool));
                let true_reg = self.value_to_reg(*true_val, &ty);
                let false_reg = self.value_to_reg(*false_val, &ty);

                writeln!(
                    self.output,
                    "selp.b32 {}, {}, {}, {};",
                    dst_reg, true_reg, false_reg, cond_reg
                )?;
            }

            Instruction::FMA { dst, a, b, c, ty } => {
                let gpu_ty = GpuType::Scalar(*ty);
                let dst_reg = self.value_to_reg(*dst, &gpu_ty);
                let a_reg = self.value_to_reg(*a, &gpu_ty);
                let b_reg = self.value_to_reg(*b, &gpu_ty);
                let c_reg = self.value_to_reg(*c, &gpu_ty);

                writeln!(
                    self.output,
                    "fma.rn.{} {}, {}, {}, {};",
                    ty.ptx_suffix(),
                    dst_reg,
                    a_reg,
                    b_reg,
                    c_reg
                )?;
            }

            Instruction::Call { dst, func, args } => {
                if let Some(d) = dst {
                    let reg = self.value_to_reg(*d, &GpuType::Scalar(ScalarType::U64));
                    write!(self.output, "call ({}) ", reg)?;
                } else {
                    write!(self.output, "call ")?;
                }

                write!(self.output, "{}", func)?;

                if !args.is_empty() {
                    write!(self.output, ", (")?;
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(self.output, ", ")?;
                        }
                        write!(self.output, "%{}", arg.0)?;
                    }
                    write!(self.output, ")")?;
                }

                writeln!(self.output, ";")?;
            }

            // Skip phi nodes in PTX (should be eliminated before codegen)
            Instruction::Phi { .. } => {
                writeln!(self.output, "// phi node (should be eliminated)")?;
            }

            // Warp operations
            Instruction::WarpId { dst } => {
                let reg = self.value_to_reg(*dst, &GpuType::Scalar(ScalarType::U32));
                writeln!(self.output, "mov.u32 {}, %warpid;", reg)?;
            }

            Instruction::LaneId { dst } => {
                let reg = self.value_to_reg(*dst, &GpuType::Scalar(ScalarType::U32));
                writeln!(self.output, "mov.u32 {}, %laneid;", reg)?;
            }

            Instruction::WarpShuffle { dst, src, lane, ty } => {
                let gpu_ty = GpuType::Scalar(*ty);
                let dst_reg = self.value_to_reg(*dst, &gpu_ty);
                let src_reg = self.value_to_reg(*src, &gpu_ty);
                let lane_reg = self.value_to_reg(*lane, &GpuType::Scalar(ScalarType::U32));

                writeln!(
                    self.output,
                    "shfl.sync.idx.b32 {}, {}, {}, 31, 0xffffffff;",
                    dst_reg, src_reg, lane_reg
                )?;
            }

            Instruction::WarpVote { dst, pred, kind } => {
                let dst_reg = self.value_to_reg(*dst, &GpuType::Scalar(ScalarType::Bool));
                let pred_reg = self.value_to_reg(*pred, &GpuType::Scalar(ScalarType::Bool));

                let vote_type = match kind {
                    WarpVoteKind::All => "all",
                    WarpVoteKind::Any => "any",
                    WarpVoteKind::Uni => "uni",
                };

                writeln!(
                    self.output,
                    "vote.sync.{}.pred {}, {}, 0xffffffff;",
                    vote_type, dst_reg, pred_reg
                )?;
            }

            Instruction::WarpReduce { dst, src, op, ty } => {
                let gpu_ty = GpuType::Scalar(*ty);
                let dst_reg = self.value_to_reg(*dst, &gpu_ty);
                let src_reg = self.value_to_reg(*src, &gpu_ty);

                writeln!(
                    self.output,
                    "redux.sync.{}.{} {}, {}, 0xffffffff;",
                    op.ptx_name(),
                    ty.ptx_suffix(),
                    dst_reg,
                    src_reg
                )?;
            }

            Instruction::AtomicCAS {
                dst,
                ptr,
                expected,
                desired,
                ty,
                space,
            } => {
                let gpu_ty = GpuType::Scalar(*ty);
                let dst_reg = self.value_to_reg(*dst, &gpu_ty);
                let ptr_reg =
                    self.value_to_reg(*ptr, &GpuType::Ptr(Box::new(gpu_ty.clone()), *space));
                let exp_reg = self.value_to_reg(*expected, &gpu_ty);
                let des_reg = self.value_to_reg(*desired, &gpu_ty);

                writeln!(
                    self.output,
                    "atom.{}.cas.{} {}, [{}], {}, {};",
                    space.ptx_name(),
                    ty.ptx_suffix(),
                    dst_reg,
                    ptr_reg,
                    exp_reg,
                    des_reg
                )?;
            }

            Instruction::Bitcast { dst, src, to } => {
                let dst_reg = self.value_to_reg(*dst, to);
                let src_reg = self.value_to_reg(*src, to); // Approximate

                writeln!(
                    self.output,
                    "mov.b{} {}, {};",
                    to.size_bytes().unwrap_or(4) * 8,
                    dst_reg,
                    src_reg
                )?;
            }

            Instruction::SharedAlloc { dst, ty, size } => {
                let dst_reg = self.value_to_reg(
                    *dst,
                    &GpuType::Ptr(Box::new(ty.clone()), AddressSpace::Shared),
                );
                let total_size = ty.size_bytes().unwrap_or(1) * size;
                writeln!(
                    self.output,
                    "// shared alloc: {} bytes -> {}",
                    total_size, dst_reg
                )?;
            }
        }

        Ok(())
    }

    fn emit_indent(&mut self) -> Result<(), PtxError> {
        for _ in 0..self.indent {
            write!(self.output, "\t")?;
        }
        Ok(())
    }

    fn type_to_ptx(&self, ty: &GpuType) -> String {
        match ty {
            GpuType::Scalar(s) => format!(".{}", s.ptx_suffix()),
            GpuType::Vector(s, w) => format!(".v{}.{}", w.as_usize(), s.ptx_suffix()),
            GpuType::Ptr(_, _) => ".u64".to_string(),
            GpuType::Array(elem, _) => self.type_to_ptx(elem),
            GpuType::Struct(_) => ".b8".to_string(),
            GpuType::Void => "".to_string(),
        }
    }

    fn type_to_ptx_param(&self, ty: &GpuType) -> String {
        match ty {
            GpuType::Scalar(s) => format!(".{}", s.ptx_suffix()),
            GpuType::Ptr(_, _) => ".u64".to_string(),
            _ => ".b64".to_string(),
        }
    }

    fn type_to_ptx_load(&self, ty: &GpuType) -> String {
        match ty {
            GpuType::Scalar(s) => s.ptx_suffix().to_string(),
            GpuType::Vector(s, w) => format!("v{}.{}", w.as_usize(), s.ptx_suffix()),
            GpuType::Ptr(_, _) => "u64".to_string(),
            _ => "b64".to_string(),
        }
    }

    fn value_to_reg(&self, id: ValueId, ty: &GpuType) -> String {
        match ty {
            GpuType::Scalar(ScalarType::Bool) => format!("%p{}", id.0),
            GpuType::Scalar(
                ScalarType::I8
                | ScalarType::U8
                | ScalarType::I16
                | ScalarType::U16
                | ScalarType::F16
                | ScalarType::BF16,
            ) => {
                format!("%h{}", id.0)
            }
            GpuType::Scalar(ScalarType::I32 | ScalarType::U32) => format!("%r{}", id.0),
            GpuType::Scalar(ScalarType::I64 | ScalarType::U64) => format!("%rd{}", id.0),
            GpuType::Scalar(ScalarType::F32) => format!("%f{}", id.0),
            GpuType::Scalar(ScalarType::F64) => format!("%fd{}", id.0),
            GpuType::Ptr(_, _) => format!("%rd{}", id.0),
            _ => format!("%r{}", id.0),
        }
    }
}

impl Default for PtxGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::builder::FunctionBuilder;
    use crate::ir::types::common;

    #[test]
    fn test_simple_kernel() {
        let mut builder = FunctionBuilder::kernel("test");
        let a = builder.param("a", common::f32_ptr());
        let n = builder.param("n", common::u32());

        let idx = builder.thread_idx(Dimension::X);
        builder.ret_void();

        let func = builder.build();

        let mut gen = PtxGenerator::new();
        let ptx = gen.generate_function(&func).unwrap();

        assert!(ptx.contains(".entry test"));
        assert!(ptx.contains("mov.u32"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn test_module_generation() {
        let mut module = GpuModule::new("test_module")
            .with_ptx_version(8, 0)
            .with_sm_version(86);

        let mut builder = FunctionBuilder::kernel("vector_add");
        builder.param("a", common::f32_ptr());
        builder.param("b", common::f32_ptr());
        builder.param("c", common::f32_ptr());
        builder.param("n", common::u32());
        builder.ret_void();

        module.add_function(builder.build());

        let mut gen = PtxGenerator::new();
        let ptx = gen.generate_module(&module).unwrap();

        assert!(ptx.contains(".version 8.0"));
        assert!(ptx.contains(".target sm_86"));
        assert!(ptx.contains(".entry vector_add"));
    }
}
