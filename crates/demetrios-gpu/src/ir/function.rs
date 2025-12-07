//! GPU IR Functions and Basic Blocks
//!
//! Defines the structure for GPU kernels and device functions.

use super::inst::{BlockId, Instruction, ValueId};
use super::types::{AddressSpace, GpuType};
use indexmap::IndexMap;
use std::fmt;

/// Parameter for a GPU function
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub ty: GpuType,
    /// Address space (for pointers)
    pub space: Option<AddressSpace>,
}

impl Parameter {
    pub fn new(name: impl Into<String>, ty: GpuType) -> Self {
        let space = ty.address_space();
        Parameter {
            name: name.into(),
            ty,
            space,
        }
    }
}

/// A basic block in the control flow graph
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Block identifier
    pub id: BlockId,
    /// Block label (optional, for debugging)
    pub label: Option<String>,
    /// Instructions in this block
    pub instructions: Vec<Instruction>,
}

impl BasicBlock {
    pub fn new(id: BlockId) -> Self {
        BasicBlock {
            id,
            label: None,
            instructions: Vec::new(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Add an instruction to this block
    pub fn push(&mut self, inst: Instruction) {
        self.instructions.push(inst);
    }

    /// Check if this block has a terminator
    pub fn has_terminator(&self) -> bool {
        self.instructions
            .last()
            .map_or(false, |i| i.is_terminator())
    }

    /// Get the terminator instruction if present
    pub fn terminator(&self) -> Option<&Instruction> {
        self.instructions.last().filter(|i| i.is_terminator())
    }

    /// Get successor block IDs
    pub fn successors(&self) -> Vec<BlockId> {
        match self.terminator() {
            Some(Instruction::Branch { target }) => vec![*target],
            Some(Instruction::CondBranch {
                true_target,
                false_target,
                ..
            }) => vec![*true_target, *false_target],
            Some(Instruction::Return { .. }) => vec![],
            _ => vec![],
        }
    }
}

/// Function kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FunctionKind {
    /// Kernel (entry point, callable from host)
    Kernel,
    /// Device function (callable from device only)
    Device,
}

impl FunctionKind {
    pub fn ptx_directive(self) -> &'static str {
        match self {
            FunctionKind::Kernel => ".entry",
            FunctionKind::Device => ".func",
        }
    }
}

impl fmt::Display for FunctionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionKind::Kernel => write!(f, "kernel"),
            FunctionKind::Device => write!(f, "device"),
        }
    }
}

/// A GPU function or kernel
#[derive(Debug, Clone)]
pub struct GpuFunction {
    /// Function name
    pub name: String,
    /// Function kind (kernel or device)
    pub kind: FunctionKind,
    /// Parameters
    pub params: Vec<Parameter>,
    /// Return type (None for void)
    pub return_type: Option<GpuType>,
    /// Basic blocks (in order)
    pub blocks: IndexMap<BlockId, BasicBlock>,
    /// Entry block ID
    pub entry: BlockId,
    /// Value types (for type checking/codegen)
    pub value_types: IndexMap<ValueId, GpuType>,
    /// Shared memory declarations
    pub shared_mem: Vec<SharedMemDecl>,
    /// Required shared memory size (dynamic)
    pub dynamic_shared_mem: usize,
    /// Max threads per block (optional hint)
    pub max_threads: Option<u32>,
    /// Min blocks per SM (optional hint)
    pub min_blocks: Option<u32>,
}

/// Shared memory declaration
#[derive(Debug, Clone)]
pub struct SharedMemDecl {
    pub name: String,
    pub ty: GpuType,
    pub size: usize,
}

impl GpuFunction {
    /// Create a new kernel
    pub fn kernel(name: impl Into<String>) -> Self {
        let entry = BlockId(0);
        let mut blocks = IndexMap::new();
        blocks.insert(entry, BasicBlock::new(entry));

        GpuFunction {
            name: name.into(),
            kind: FunctionKind::Kernel,
            params: Vec::new(),
            return_type: None,
            blocks,
            entry,
            value_types: IndexMap::new(),
            shared_mem: Vec::new(),
            dynamic_shared_mem: 0,
            max_threads: None,
            min_blocks: None,
        }
    }

    /// Create a new device function
    pub fn device(name: impl Into<String>) -> Self {
        let entry = BlockId(0);
        let mut blocks = IndexMap::new();
        blocks.insert(entry, BasicBlock::new(entry));

        GpuFunction {
            name: name.into(),
            kind: FunctionKind::Device,
            params: Vec::new(),
            return_type: None,
            blocks,
            entry,
            value_types: IndexMap::new(),
            shared_mem: Vec::new(),
            dynamic_shared_mem: 0,
            max_threads: None,
            min_blocks: None,
        }
    }

    /// Add a parameter
    pub fn with_param(mut self, name: impl Into<String>, ty: GpuType) -> Self {
        self.params.push(Parameter::new(name, ty));
        self
    }

    /// Set return type
    pub fn with_return(mut self, ty: GpuType) -> Self {
        self.return_type = Some(ty);
        self
    }

    /// Set max threads hint
    pub fn with_max_threads(mut self, max: u32) -> Self {
        self.max_threads = Some(max);
        self
    }

    /// Set min blocks hint
    pub fn with_min_blocks(mut self, min: u32) -> Self {
        self.min_blocks = Some(min);
        self
    }

    /// Get the entry block
    pub fn entry_block(&self) -> &BasicBlock {
        self.blocks
            .get(&self.entry)
            .expect("entry block must exist")
    }

    /// Get a mutable reference to the entry block
    pub fn entry_block_mut(&mut self) -> &mut BasicBlock {
        self.blocks
            .get_mut(&self.entry)
            .expect("entry block must exist")
    }

    /// Get a block by ID
    pub fn get_block(&self, id: BlockId) -> Option<&BasicBlock> {
        self.blocks.get(&id)
    }

    /// Get a mutable block by ID
    pub fn get_block_mut(&mut self, id: BlockId) -> Option<&mut BasicBlock> {
        self.blocks.get_mut(&id)
    }

    /// Create a new basic block and return its ID
    pub fn create_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.insert(id, BasicBlock::new(id));
        id
    }

    /// Create a new basic block with a label
    pub fn create_labeled_block(&mut self, label: impl Into<String>) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks
            .insert(id, BasicBlock::new(id).with_label(label));
        id
    }

    /// Add shared memory declaration
    pub fn add_shared_mem(&mut self, name: impl Into<String>, ty: GpuType, size: usize) {
        self.shared_mem.push(SharedMemDecl {
            name: name.into(),
            ty,
            size,
        });
    }

    /// Set dynamic shared memory size
    pub fn set_dynamic_shared_mem(&mut self, size: usize) {
        self.dynamic_shared_mem = size;
    }

    /// Validate the function structure
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // Check that entry block exists
        if !self.blocks.contains_key(&self.entry) {
            errors.push("Entry block does not exist".to_string());
        }

        // Check that all blocks have terminators
        for (id, block) in &self.blocks {
            if !block.has_terminator() {
                errors.push(format!("Block {} has no terminator", id));
            }
        }

        // Check that all branch targets exist
        for (_, block) in &self.blocks {
            for succ in block.successors() {
                if !self.blocks.contains_key(&succ) {
                    errors.push(format!(
                        "Block {} references non-existent block {}",
                        block.id, succ
                    ));
                }
            }
        }

        // Kernels should not have return values
        if self.kind == FunctionKind::Kernel && self.return_type.is_some() {
            errors.push("Kernels cannot have return values".to_string());
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Get all value IDs used in the function
    pub fn values(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.value_types.keys().copied()
    }

    /// Register a value with its type
    pub fn register_value(&mut self, id: ValueId, ty: GpuType) {
        self.value_types.insert(id, ty);
    }

    /// Get the type of a value
    pub fn value_type(&self, id: ValueId) -> Option<&GpuType> {
        self.value_types.get(&id)
    }
}

/// GPU Module containing multiple functions
#[derive(Debug, Clone)]
pub struct GpuModule {
    /// Module name
    pub name: String,
    /// Functions in this module
    pub functions: IndexMap<String, GpuFunction>,
    /// Global constants
    pub constants: Vec<GlobalConstant>,
    /// PTX version to target
    pub ptx_version: (u32, u32),
    /// SM architecture to target
    pub sm_version: u32,
}

/// Global constant declaration
#[derive(Debug, Clone)]
pub struct GlobalConstant {
    pub name: String,
    pub ty: GpuType,
    pub data: Vec<u8>,
}

impl GpuModule {
    /// Create a new module
    pub fn new(name: impl Into<String>) -> Self {
        GpuModule {
            name: name.into(),
            functions: IndexMap::new(),
            constants: Vec::new(),
            ptx_version: (8, 0),
            sm_version: 75, // Default to SM 7.5 (Turing)
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

    /// Add a function
    pub fn add_function(&mut self, func: GpuFunction) {
        self.functions.insert(func.name.clone(), func);
    }

    /// Get a function by name
    pub fn get_function(&self, name: &str) -> Option<&GpuFunction> {
        self.functions.get(name)
    }

    /// Get a mutable function by name
    pub fn get_function_mut(&mut self, name: &str) -> Option<&mut GpuFunction> {
        self.functions.get_mut(name)
    }

    /// Add a global constant
    pub fn add_constant(&mut self, name: impl Into<String>, ty: GpuType, data: Vec<u8>) {
        self.constants.push(GlobalConstant {
            name: name.into(),
            ty,
            data,
        });
    }

    /// Validate all functions in the module
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut all_errors = Vec::new();

        for (name, func) in &self.functions {
            if let Err(errors) = func.validate() {
                for error in errors {
                    all_errors.push(format!("{}: {}", name, error));
                }
            }
        }

        if all_errors.is_empty() {
            Ok(())
        } else {
            Err(all_errors)
        }
    }

    /// Get all kernels in this module
    pub fn kernels(&self) -> impl Iterator<Item = &GpuFunction> {
        self.functions
            .values()
            .filter(|f| f.kind == FunctionKind::Kernel)
    }

    /// Get all device functions in this module
    pub fn device_functions(&self) -> impl Iterator<Item = &GpuFunction> {
        self.functions
            .values()
            .filter(|f| f.kind == FunctionKind::Device)
    }
}

impl Default for GpuModule {
    fn default() -> Self {
        Self::new("unnamed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::inst::Instruction;
    use crate::ir::types::common;

    #[test]
    fn test_create_kernel() {
        let kernel = GpuFunction::kernel("add_vectors")
            .with_param("a", common::f32_ptr())
            .with_param("b", common::f32_ptr())
            .with_param("c", common::f32_ptr())
            .with_param("n", common::u32())
            .with_max_threads(256);

        assert_eq!(kernel.name, "add_vectors");
        assert_eq!(kernel.kind, FunctionKind::Kernel);
        assert_eq!(kernel.params.len(), 4);
        assert_eq!(kernel.max_threads, Some(256));
    }

    #[test]
    fn test_basic_block_terminator() {
        let mut block = BasicBlock::new(BlockId(0));
        assert!(!block.has_terminator());

        block.push(Instruction::Return { value: None });
        assert!(block.has_terminator());
    }

    #[test]
    fn test_module_creation() {
        let mut module = GpuModule::new("test_module")
            .with_ptx_version(8, 0)
            .with_sm_version(86);

        let kernel = GpuFunction::kernel("test_kernel");
        module.add_function(kernel);

        assert_eq!(module.functions.len(), 1);
        assert!(module.get_function("test_kernel").is_some());
    }
}
