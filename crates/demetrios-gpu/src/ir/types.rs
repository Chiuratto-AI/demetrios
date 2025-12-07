//! GPU IR Type System
//!
//! Defines the type system for GPU intermediate representation,
//! including scalar types, vector types, pointer types, and array types.

use std::fmt;

/// Scalar types supported on GPU
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    /// Boolean (1-bit logical)
    Bool,
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// 16-bit floating point (half precision)
    F16,
    /// 32-bit floating point (single precision)
    F32,
    /// 64-bit floating point (double precision)
    F64,
    /// Brain floating point (bfloat16)
    BF16,
}

impl ScalarType {
    /// Get the size in bits
    pub fn bit_width(self) -> u32 {
        match self {
            ScalarType::Bool => 1,
            ScalarType::I8 | ScalarType::U8 => 8,
            ScalarType::I16 | ScalarType::U16 | ScalarType::F16 | ScalarType::BF16 => 16,
            ScalarType::I32 | ScalarType::U32 | ScalarType::F32 => 32,
            ScalarType::I64 | ScalarType::U64 | ScalarType::F64 => 64,
        }
    }

    /// Get the size in bytes
    pub fn byte_width(self) -> u32 {
        (self.bit_width() + 7) / 8
    }

    /// Check if this is a floating-point type
    pub fn is_float(self) -> bool {
        matches!(
            self,
            ScalarType::F16 | ScalarType::F32 | ScalarType::F64 | ScalarType::BF16
        )
    }

    /// Check if this is a signed integer type
    pub fn is_signed(self) -> bool {
        matches!(
            self,
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64
        )
    }

    /// Check if this is an unsigned integer type
    pub fn is_unsigned(self) -> bool {
        matches!(
            self,
            ScalarType::U8 | ScalarType::U16 | ScalarType::U32 | ScalarType::U64
        )
    }

    /// Check if this is an integer type
    pub fn is_integer(self) -> bool {
        self.is_signed() || self.is_unsigned()
    }

    /// Get the PTX type suffix
    pub fn ptx_suffix(self) -> &'static str {
        match self {
            ScalarType::Bool => "pred",
            ScalarType::I8 => "s8",
            ScalarType::I16 => "s16",
            ScalarType::I32 => "s32",
            ScalarType::I64 => "s64",
            ScalarType::U8 => "u8",
            ScalarType::U16 => "u16",
            ScalarType::U32 => "u32",
            ScalarType::U64 => "u64",
            ScalarType::F16 => "f16",
            ScalarType::F32 => "f32",
            ScalarType::F64 => "f64",
            ScalarType::BF16 => "bf16",
        }
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarType::Bool => write!(f, "bool"),
            ScalarType::I8 => write!(f, "i8"),
            ScalarType::I16 => write!(f, "i16"),
            ScalarType::I32 => write!(f, "i32"),
            ScalarType::I64 => write!(f, "i64"),
            ScalarType::U8 => write!(f, "u8"),
            ScalarType::U16 => write!(f, "u16"),
            ScalarType::U32 => write!(f, "u32"),
            ScalarType::U64 => write!(f, "u64"),
            ScalarType::F16 => write!(f, "f16"),
            ScalarType::F32 => write!(f, "f32"),
            ScalarType::F64 => write!(f, "f64"),
            ScalarType::BF16 => write!(f, "bf16"),
        }
    }
}

/// Vector width for SIMD operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VectorWidth {
    V2,
    V4,
    V8,
    V16,
}

impl VectorWidth {
    pub fn as_usize(self) -> usize {
        match self {
            VectorWidth::V2 => 2,
            VectorWidth::V4 => 4,
            VectorWidth::V8 => 8,
            VectorWidth::V16 => 16,
        }
    }
}

impl fmt::Display for VectorWidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorWidth::V2 => write!(f, "x2"),
            VectorWidth::V4 => write!(f, "x4"),
            VectorWidth::V8 => write!(f, "x8"),
            VectorWidth::V16 => write!(f, "x16"),
        }
    }
}

/// Memory address space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum AddressSpace {
    /// Global memory (device memory, accessible by all threads)
    #[default]
    Global,
    /// Shared memory (per-block, fast)
    Shared,
    /// Constant memory (read-only, cached)
    Constant,
    /// Local memory (per-thread, spilled registers)
    Local,
    /// Texture memory (read-only, spatially cached)
    Texture,
    /// Generic address space (resolved at runtime)
    Generic,
}

impl AddressSpace {
    /// Get the PTX state space name
    pub fn ptx_name(self) -> &'static str {
        match self {
            AddressSpace::Global => "global",
            AddressSpace::Shared => "shared",
            AddressSpace::Constant => "const",
            AddressSpace::Local => "local",
            AddressSpace::Texture => "tex",
            AddressSpace::Generic => "generic",
        }
    }

    /// Check if this is a fast memory space
    pub fn is_fast(self) -> bool {
        matches!(self, AddressSpace::Shared | AddressSpace::Constant)
    }
}

impl fmt::Display for AddressSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.ptx_name())
    }
}

/// GPU IR Types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GpuType {
    /// Scalar type
    Scalar(ScalarType),
    /// Vector type (scalar + width)
    Vector(ScalarType, VectorWidth),
    /// Pointer to memory
    Ptr(Box<GpuType>, AddressSpace),
    /// Fixed-size array
    Array(Box<GpuType>, usize),
    /// Struct type (named fields)
    Struct(StructType),
    /// Void type (for functions returning nothing)
    Void,
}

impl GpuType {
    /// Create a scalar type
    pub fn scalar(ty: ScalarType) -> Self {
        GpuType::Scalar(ty)
    }

    /// Create a vector type
    pub fn vector(ty: ScalarType, width: VectorWidth) -> Self {
        GpuType::Vector(ty, width)
    }

    /// Create a pointer type
    pub fn ptr(pointee: GpuType, space: AddressSpace) -> Self {
        GpuType::Ptr(Box::new(pointee), space)
    }

    /// Create a global pointer type
    pub fn global_ptr(pointee: GpuType) -> Self {
        Self::ptr(pointee, AddressSpace::Global)
    }

    /// Create a shared pointer type
    pub fn shared_ptr(pointee: GpuType) -> Self {
        Self::ptr(pointee, AddressSpace::Shared)
    }

    /// Create an array type
    pub fn array(element: GpuType, size: usize) -> Self {
        GpuType::Array(Box::new(element), size)
    }

    /// Get the size in bytes (returns None for unsized types)
    pub fn size_bytes(&self) -> Option<usize> {
        match self {
            GpuType::Scalar(s) => Some(s.byte_width() as usize),
            GpuType::Vector(s, w) => Some(s.byte_width() as usize * w.as_usize()),
            GpuType::Ptr(_, _) => Some(8), // 64-bit pointers
            GpuType::Array(elem, size) => elem.size_bytes().map(|s| s * size),
            GpuType::Struct(s) => s.size_bytes(),
            GpuType::Void => Some(0),
        }
    }

    /// Get the alignment in bytes
    pub fn alignment(&self) -> usize {
        match self {
            GpuType::Scalar(s) => s.byte_width() as usize,
            GpuType::Vector(s, w) => {
                let elem_size = s.byte_width() as usize;
                // Vectors align to their total size, capped at 16 bytes
                (elem_size * w.as_usize()).min(16)
            }
            GpuType::Ptr(_, _) => 8,
            GpuType::Array(elem, _) => elem.alignment(),
            GpuType::Struct(s) => s.alignment(),
            GpuType::Void => 1,
        }
    }

    /// Check if this is a scalar type
    pub fn is_scalar(&self) -> bool {
        matches!(self, GpuType::Scalar(_))
    }

    /// Check if this is a pointer type
    pub fn is_pointer(&self) -> bool {
        matches!(self, GpuType::Ptr(_, _))
    }

    /// Get the pointee type if this is a pointer
    pub fn pointee(&self) -> Option<&GpuType> {
        match self {
            GpuType::Ptr(pointee, _) => Some(pointee),
            _ => None,
        }
    }

    /// Get the address space if this is a pointer
    pub fn address_space(&self) -> Option<AddressSpace> {
        match self {
            GpuType::Ptr(_, space) => Some(*space),
            _ => None,
        }
    }
}

impl fmt::Display for GpuType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuType::Scalar(s) => write!(f, "{}", s),
            GpuType::Vector(s, w) => write!(f, "{}{}", s, w),
            GpuType::Ptr(pointee, space) => write!(f, "{}* {}", pointee, space),
            GpuType::Array(elem, size) => write!(f, "[{}; {}]", elem, size),
            GpuType::Struct(s) => write!(f, "struct {}", s.name),
            GpuType::Void => write!(f, "void"),
        }
    }
}

/// Struct type definition
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StructType {
    /// Struct name
    pub name: String,
    /// Fields (name, type)
    pub fields: Vec<(String, GpuType)>,
}

impl StructType {
    /// Create a new struct type
    pub fn new(name: impl Into<String>) -> Self {
        StructType {
            name: name.into(),
            fields: Vec::new(),
        }
    }

    /// Add a field to the struct
    pub fn with_field(mut self, name: impl Into<String>, ty: GpuType) -> Self {
        self.fields.push((name.into(), ty));
        self
    }

    /// Get the size in bytes with padding
    pub fn size_bytes(&self) -> Option<usize> {
        let mut offset = 0;
        for (_, ty) in &self.fields {
            let align = ty.alignment();
            offset = (offset + align - 1) / align * align; // Align
            offset += ty.size_bytes()?;
        }
        // Final alignment
        let struct_align = self.alignment();
        offset = (offset + struct_align - 1) / struct_align * struct_align;
        Some(offset)
    }

    /// Get the alignment (max of all field alignments)
    pub fn alignment(&self) -> usize {
        self.fields
            .iter()
            .map(|(_, ty)| ty.alignment())
            .max()
            .unwrap_or(1)
    }

    /// Get the offset of a field
    pub fn field_offset(&self, name: &str) -> Option<usize> {
        let mut offset = 0;
        for (field_name, ty) in &self.fields {
            let align = ty.alignment();
            offset = (offset + align - 1) / align * align;
            if field_name == name {
                return Some(offset);
            }
            offset += ty.size_bytes()?;
        }
        None
    }
}

/// Common GPU types
pub mod common {
    use super::*;

    pub fn i32() -> GpuType {
        GpuType::Scalar(ScalarType::I32)
    }

    pub fn u32() -> GpuType {
        GpuType::Scalar(ScalarType::U32)
    }

    pub fn f32() -> GpuType {
        GpuType::Scalar(ScalarType::F32)
    }

    pub fn f64() -> GpuType {
        GpuType::Scalar(ScalarType::F64)
    }

    pub fn bool() -> GpuType {
        GpuType::Scalar(ScalarType::Bool)
    }

    pub fn f32_ptr() -> GpuType {
        GpuType::global_ptr(f32())
    }

    pub fn f64_ptr() -> GpuType {
        GpuType::global_ptr(f64())
    }

    pub fn float4() -> GpuType {
        GpuType::Vector(ScalarType::F32, VectorWidth::V4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_sizes() {
        assert_eq!(ScalarType::Bool.bit_width(), 1);
        assert_eq!(ScalarType::I32.bit_width(), 32);
        assert_eq!(ScalarType::F64.bit_width(), 64);
    }

    #[test]
    fn test_type_sizes() {
        assert_eq!(common::i32().size_bytes(), Some(4));
        assert_eq!(common::f64().size_bytes(), Some(8));
        assert_eq!(common::float4().size_bytes(), Some(16));
    }

    #[test]
    fn test_struct_layout() {
        let s = StructType::new("Vec3")
            .with_field("x", common::f32())
            .with_field("y", common::f32())
            .with_field("z", common::f32());

        assert_eq!(s.size_bytes(), Some(12));
        assert_eq!(s.alignment(), 4);
        assert_eq!(s.field_offset("x"), Some(0));
        assert_eq!(s.field_offset("y"), Some(4));
        assert_eq!(s.field_offset("z"), Some(8));
    }

    #[test]
    fn test_pointer_types() {
        let ptr = GpuType::global_ptr(common::f32());
        assert!(ptr.is_pointer());
        assert_eq!(ptr.address_space(), Some(AddressSpace::Global));
        assert_eq!(ptr.size_bytes(), Some(8));
    }
}
