// mod.d - Linear Algebra Module for Demetrios
//
// Provides vector and matrix types for scientific computing:
// - Vec2, Vec3, Vec4: Small fixed-size vectors
// - Vec14: 14-element vector for PBPK compartment states
// - Mat2, Mat3, Mat4: Fixed-size matrices
//
// All types are stack-allocated and optimized for SIMD operations.

module linalg

// Export all vector operations
pub use vector::*

// Export all matrix operations
pub use matrix::*
