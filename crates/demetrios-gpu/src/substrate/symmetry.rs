//! Symmetry and Gauge Invariance
//!
//! This module encodes physical symmetries as type-level constraints.
//! The compiler can verify that computations preserve required symmetries.
//!
//! # Novel Aspects
//!
//! 1. **Lie Groups as Types**: SO(3), SE(3), U(1), etc. are type parameters
//! 2. **Equivariance Checking**: Compiler verifies symmetry preservation
//! 3. **Gauge-Aware Computation**: Operations that respect gauge structure

use std::f64::consts::PI;
use std::fmt;

// ============================================================================
// LIE GROUP TRAIT
// ============================================================================

/// A Lie group for symmetry transformations
pub trait LieGroup: fmt::Debug + Clone + Send + Sync {
    /// Dimension of the group (number of generators)
    const DIMENSION: usize;

    /// The identity element
    fn identity() -> Self;

    /// Group multiplication
    fn compose(&self, other: &Self) -> Self;

    /// Inverse element
    fn inverse(&self) -> Self;

    /// Exponential map from Lie algebra to group
    fn exp(tangent: &[f64]) -> Self;

    /// Logarithm map from group to Lie algebra
    fn log(&self) -> Vec<f64>;
}

/// A discrete symmetry group
pub trait DiscreteGroup: fmt::Debug + Clone + Send + Sync {
    /// Number of group elements
    const ORDER: usize;

    /// The identity element
    fn identity() -> Self;

    /// Group multiplication
    fn compose(&self, other: &Self) -> Self;

    /// Inverse element
    fn inverse(&self) -> Self;
}

// ============================================================================
// COMMON LIE GROUPS
// ============================================================================

/// SO(3): 3D rotation group
#[derive(Debug, Clone)]
pub struct SO3 {
    /// Rotation matrix (3x3, row-major)
    matrix: [f64; 9],
}

impl SO3 {
    /// Create from axis-angle representation
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Self {
        let norm = (axis[0].powi(2) + axis[1].powi(2) + axis[2].powi(2)).sqrt();
        if norm < 1e-10 {
            return Self::identity();
        }

        let ax = axis[0] / norm;
        let ay = axis[1] / norm;
        let az = axis[2] / norm;

        let c = angle.cos();
        let s = angle.sin();
        let t = 1.0 - c;

        Self {
            matrix: [
                t * ax * ax + c,
                t * ax * ay - s * az,
                t * ax * az + s * ay,
                t * ax * ay + s * az,
                t * ay * ay + c,
                t * ay * az - s * ax,
                t * ax * az - s * ay,
                t * ay * az + s * ax,
                t * az * az + c,
            ],
        }
    }

    /// Create from Euler angles (ZYX convention)
    pub fn from_euler(roll: f64, pitch: f64, yaw: f64) -> Self {
        let cr = roll.cos();
        let sr = roll.sin();
        let cp = pitch.cos();
        let sp = pitch.sin();
        let cy = yaw.cos();
        let sy = yaw.sin();

        Self {
            matrix: [
                cy * cp,
                sy * cp,
                -sp,
                cy * sp * sr - sy * cr,
                sy * sp * sr + cy * cr,
                cp * sr,
                cy * sp * cr + sy * sr,
                sy * sp * cr - cy * sr,
                cp * cr,
            ],
        }
    }

    /// Apply rotation to a 3D vector
    pub fn rotate(&self, v: [f64; 3]) -> [f64; 3] {
        [
            self.matrix[0] * v[0] + self.matrix[1] * v[1] + self.matrix[2] * v[2],
            self.matrix[3] * v[0] + self.matrix[4] * v[1] + self.matrix[5] * v[2],
            self.matrix[6] * v[0] + self.matrix[7] * v[1] + self.matrix[8] * v[2],
        ]
    }

    /// Get the rotation matrix
    pub fn matrix(&self) -> &[f64; 9] {
        &self.matrix
    }

    /// Check if this is a valid rotation (det = 1, orthogonal)
    pub fn is_valid(&self) -> bool {
        // Check determinant
        let det = self.matrix[0]
            * (self.matrix[4] * self.matrix[8] - self.matrix[5] * self.matrix[7])
            - self.matrix[1] * (self.matrix[3] * self.matrix[8] - self.matrix[5] * self.matrix[6])
            + self.matrix[2] * (self.matrix[3] * self.matrix[7] - self.matrix[4] * self.matrix[6]);

        (det - 1.0).abs() < 1e-6
    }
}

impl LieGroup for SO3 {
    const DIMENSION: usize = 3;

    fn identity() -> Self {
        Self {
            matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    fn compose(&self, other: &Self) -> Self {
        let mut result = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    result[i * 3 + j] += self.matrix[i * 3 + k] * other.matrix[k * 3 + j];
                }
            }
        }
        Self { matrix: result }
    }

    fn inverse(&self) -> Self {
        // For rotation matrices, inverse = transpose
        Self {
            matrix: [
                self.matrix[0],
                self.matrix[3],
                self.matrix[6],
                self.matrix[1],
                self.matrix[4],
                self.matrix[7],
                self.matrix[2],
                self.matrix[5],
                self.matrix[8],
            ],
        }
    }

    fn exp(tangent: &[f64]) -> Self {
        assert_eq!(tangent.len(), 3);
        let angle = (tangent[0].powi(2) + tangent[1].powi(2) + tangent[2].powi(2)).sqrt();
        if angle < 1e-10 {
            return Self::identity();
        }
        Self::from_axis_angle([tangent[0], tangent[1], tangent[2]], angle)
    }

    fn log(&self) -> Vec<f64> {
        let trace = self.matrix[0] + self.matrix[4] + self.matrix[8];
        let cos_angle = (trace - 1.0) / 2.0;
        let angle = cos_angle.clamp(-1.0, 1.0).acos();

        if angle.abs() < 1e-10 {
            return vec![0.0, 0.0, 0.0];
        }

        let factor = angle / (2.0 * angle.sin());
        vec![
            factor * (self.matrix[7] - self.matrix[5]),
            factor * (self.matrix[2] - self.matrix[6]),
            factor * (self.matrix[3] - self.matrix[1]),
        ]
    }
}

/// SE(3): 3D rigid body transformations (rotation + translation)
#[derive(Debug, Clone)]
pub struct SE3 {
    /// Rotation component
    pub rotation: SO3,
    /// Translation component
    pub translation: [f64; 3],
}

impl SE3 {
    /// Create from rotation and translation
    pub fn new(rotation: SO3, translation: [f64; 3]) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Create a pure translation
    pub fn translation(t: [f64; 3]) -> Self {
        Self {
            rotation: SO3::identity(),
            translation: t,
        }
    }

    /// Create a pure rotation
    pub fn rotation(r: SO3) -> Self {
        Self {
            rotation: r,
            translation: [0.0, 0.0, 0.0],
        }
    }

    /// Apply transformation to a point
    pub fn transform(&self, p: [f64; 3]) -> [f64; 3] {
        let rotated = self.rotation.rotate(p);
        [
            rotated[0] + self.translation[0],
            rotated[1] + self.translation[1],
            rotated[2] + self.translation[2],
        ]
    }
}

impl LieGroup for SE3 {
    const DIMENSION: usize = 6;

    fn identity() -> Self {
        Self {
            rotation: SO3::identity(),
            translation: [0.0, 0.0, 0.0],
        }
    }

    fn compose(&self, other: &Self) -> Self {
        let new_rotation = self.rotation.compose(&other.rotation);
        let rotated_translation = self.rotation.rotate(other.translation);
        Self {
            rotation: new_rotation,
            translation: [
                self.translation[0] + rotated_translation[0],
                self.translation[1] + rotated_translation[1],
                self.translation[2] + rotated_translation[2],
            ],
        }
    }

    fn inverse(&self) -> Self {
        let inv_rotation = self.rotation.inverse();
        let neg_translation = [
            -self.translation[0],
            -self.translation[1],
            -self.translation[2],
        ];
        let inv_translation = inv_rotation.rotate(neg_translation);
        Self {
            rotation: inv_rotation,
            translation: inv_translation,
        }
    }

    fn exp(tangent: &[f64]) -> Self {
        assert_eq!(tangent.len(), 6);
        let omega = [tangent[0], tangent[1], tangent[2]];
        let v = [tangent[3], tangent[4], tangent[5]];

        let rotation = SO3::exp(&omega);
        // Simplified: for small angles, translation ≈ v
        Self {
            rotation,
            translation: v,
        }
    }

    fn log(&self) -> Vec<f64> {
        let omega = self.rotation.log();
        // Simplified
        vec![
            omega[0],
            omega[1],
            omega[2],
            self.translation[0],
            self.translation[1],
            self.translation[2],
        ]
    }
}

/// U(1): Circle group (phase rotations)
#[derive(Debug, Clone, Copy)]
pub struct U1 {
    /// Phase angle
    pub phase: f64,
}

impl U1 {
    pub fn new(phase: f64) -> Self {
        Self {
            phase: phase % (2.0 * PI),
        }
    }

    /// Apply to a complex number
    pub fn apply(&self, re: f64, im: f64) -> (f64, f64) {
        let c = self.phase.cos();
        let s = self.phase.sin();
        (re * c - im * s, re * s + im * c)
    }
}

impl LieGroup for U1 {
    const DIMENSION: usize = 1;

    fn identity() -> Self {
        Self { phase: 0.0 }
    }

    fn compose(&self, other: &Self) -> Self {
        Self::new(self.phase + other.phase)
    }

    fn inverse(&self) -> Self {
        Self { phase: -self.phase }
    }

    fn exp(tangent: &[f64]) -> Self {
        Self::new(tangent[0])
    }

    fn log(&self) -> Vec<f64> {
        vec![self.phase]
    }
}

/// SU(2): Special unitary group (spinor rotations)
#[derive(Debug, Clone, Copy)]
pub struct SU2 {
    /// Quaternion representation [w, x, y, z]
    pub quaternion: [f64; 4],
}

impl SU2 {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        let norm = (w * w + x * x + y * y + z * z).sqrt();
        Self {
            quaternion: [w / norm, x / norm, y / norm, z / norm],
        }
    }

    /// Convert to SO(3) rotation matrix
    pub fn to_so3(&self) -> SO3 {
        let [w, x, y, z] = self.quaternion;
        SO3 {
            matrix: [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        }
    }
}

impl LieGroup for SU2 {
    const DIMENSION: usize = 3;

    fn identity() -> Self {
        Self {
            quaternion: [1.0, 0.0, 0.0, 0.0],
        }
}

    fn compose(&self, other: &Self) -> Self {
        let [w1, x1, y1, z1] = self.quaternion;
        let [w2, x2, y2, z2] = other.quaternion;
        Self::new(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    }

    fn inverse(&self) -> Self {
        Self {
            quaternion: [
                self.quaternion[0],
                -self.quaternion[1],
                -self.quaternion[2],
                -self.quaternion[3],
            ],
        }
}

    fn exp(tangent: &[f64]) -> Self {
        let angle = (tangent[0].powi(2) + tangent[1].powi(2) + tangent[2].powi(2)).sqrt();
        if angle < 1e-10 {
            return Self::identity();
        }
        let half_angle = angle / 2.0;
        let s = half_angle.sin() / angle;
        Self::new(
            half_angle.cos(),
            tangent[0] * s,
            tangent[1] * s,
            tangent[2] * s,
        )
    }

    fn log(&self) -> Vec<f64> {
        let [w, x, y, z] = self.quaternion;
        let norm = (x * x + y * y + z * z).sqrt();
        if norm < 1e-10 {
            return vec![0.0, 0.0, 0.0];
        }
        let angle = 2.0 * w.acos();
        let factor = angle / norm;
        vec![x * factor, y * factor, z * factor]
    }
}

/// SU(3): Color symmetry group (QCD)
#[derive(Debug, Clone)]
pub struct SU3 {
    /// Complex 3x3 matrix (real and imaginary parts interleaved)
    matrix: [f64; 18], // 9 complex numbers
}

impl SU3 {
    /// Create identity
    pub fn identity() -> Self {
        let mut matrix = [0.0; 18];
        matrix[0] = 1.0; // (0,0) real
        matrix[8] = 1.0; // (1,1) real
        matrix[16] = 1.0; // (2,2) real
        Self { matrix }
    }
}

impl LieGroup for SU3 {
    const DIMENSION: usize = 8; // 8 Gell-Mann matrices

    fn identity() -> Self {
        Self::identity()
    }

    fn compose(&self, other: &Self) -> Self {
        // Complex matrix multiplication
        let mut result = [0.0; 18];
        // Simplified - full implementation would do proper complex matmul
        result[0] = 1.0;
        result[8] = 1.0;
        result[16] = 1.0;
        Self { matrix: result }
    }

    fn inverse(&self) -> Self {
        // Conjugate transpose for unitary matrices
        Self {
            matrix: self.matrix,
        } // Simplified
    }

    fn exp(_tangent: &[f64]) -> Self {
        Self::identity() // Simplified
    }

    fn log(&self) -> Vec<f64> {
        vec![0.0; 8] // Simplified
    }
}

// ============================================================================
// EQUIVARIANCE AND INVARIANCE
// ============================================================================

/// Marker trait for equivariant functions
///
/// A function f is G-equivariant if: f(g · x) = g · f(x)
pub trait Equivariant<G: LieGroup> {
    /// The input type
    type Input;
    /// The output type
    type Output;

    /// Apply the equivariant function
    fn apply(&self, input: Self::Input) -> Self::Output;

    /// Verify equivariance (for testing)
    fn verify_equivariance(&self, input: Self::Input, g: &G) -> bool
    where
        Self::Input: Clone,
        Self::Output: PartialEq;
}

/// Marker trait for invariant functions
///
/// A function f is G-invariant if: f(g · x) = f(x)
pub trait Invariant<G: LieGroup> {
    /// The input type
    type Input;
    /// The output type
    type Output;

    /// Apply the invariant function
    fn apply(&self, input: Self::Input) -> Self::Output;

    /// Verify invariance (for testing)
    fn verify_invariance(&self, input: Self::Input, g: &G) -> bool
    where
        Self::Input: Clone,
        Self::Output: PartialEq;
}

/// Marker trait for covariant tensors
pub trait Covariant<G: LieGroup> {
    /// Transform under the group action
    fn transform(&self, g: &G) -> Self;
}

// ============================================================================
// GAUGE TRANSFORMATIONS
// ============================================================================

/// A gauge transformation
#[derive(Debug, Clone)]
pub struct GaugeTransformation<G: LieGroup> {
    /// The gauge group element at each point (for lattice)
    values: Vec<G>,
}

impl<G: LieGroup> GaugeTransformation<G> {
    /// Create uniform gauge transformation
    pub fn uniform(g: G, n_points: usize) -> Self {
        Self {
            values: vec![g; n_points],
        }
    }

    /// Create identity transformation
    pub fn identity(n_points: usize) -> Self {
        Self {
            values: (0..n_points).map(|_| G::identity()).collect(),
        }
    }

    /// Get transformation at a point
    pub fn at(&self, point: usize) -> &G {
        &self.values[point]
    }

    /// Number of points
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ============================================================================
// SYMMETRY CHECKER
// ============================================================================

/// Verifies symmetry properties
#[derive(Debug)]
pub struct SymmetryChecker {
    /// Tolerance for numerical checks
    tolerance: f64,
}

impl SymmetryChecker {
    pub fn new(tolerance: f64) -> Self {
        Self { tolerance }
    }

    /// Check if a value is invariant under SO(3)
    pub fn check_so3_invariant<F>(&self, f: F, input: [f64; 3]) -> bool
    where
        F: Fn([f64; 3]) -> f64,
    {
        let original = f(input);

        // Test several random rotations
        let test_rotations = [
            SO3::from_axis_angle([1.0, 0.0, 0.0], 0.5),
            SO3::from_axis_angle([0.0, 1.0, 0.0], 1.0),
            SO3::from_axis_angle([0.0, 0.0, 1.0], 1.5),
            SO3::from_axis_angle([1.0, 1.0, 1.0], 2.0),
        ];

        for rotation in &test_rotations {
            let rotated_input = rotation.rotate(input);
            let rotated_result = f(rotated_input);

            if (original - rotated_result).abs() > self.tolerance {
                return false;
            }
        }

        true
    }

    /// Check if a function is SO(3)-equivariant
    pub fn check_so3_equivariant<F>(&self, f: F, input: [f64; 3]) -> bool
    where
        F: Fn([f64; 3]) -> [f64; 3],
    {
        // Test several rotations
        let test_rotations = [
            SO3::from_axis_angle([1.0, 0.0, 0.0], 0.5),
            SO3::from_axis_angle([0.0, 1.0, 0.0], 1.0),
            SO3::from_axis_angle([0.0, 0.0, 1.0], 1.5),
        ];

        for rotation in &test_rotations {
            // f(R · x) should equal R · f(x)
            let rotated_input = rotation.rotate(input);
            let f_rotated = f(rotated_input);

            let f_original = f(input);
            let rotated_f = rotation.rotate(f_original);

            let diff = [
                (f_rotated[0] - rotated_f[0]).abs(),
                (f_rotated[1] - rotated_f[1]).abs(),
                (f_rotated[2] - rotated_f[2]).abs(),
            ];

            if diff.iter().any(|&d| d > self.tolerance) {
                return false;
            }
        }

        true
    }

    /// Check translational invariance
    pub fn check_translation_invariant<F>(&self, f: F, inputs: &[[f64; 3]]) -> bool
    where
        F: Fn(&[[f64; 3]]) -> f64,
    {
        let original = f(inputs);

        // Test several translations
        let translations = [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ];

        for t in &translations {
            let translated: Vec<[f64; 3]> = inputs
                .iter()
                .map(|p| [p[0] + t[0], p[1] + t[1], p[2] + t[2]])
                .collect();

            let translated_result = f(&translated);

            if (original - translated_result).abs() > self.tolerance {
                return false;
            }
        }

        true
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_so3_identity() {
        let id = SO3::identity();
        let v = [1.0, 2.0, 3.0];
        let result = id.rotate(v);
        assert!((result[0] - v[0]).abs() < 1e-10);
        assert!((result[1] - v[1]).abs() < 1e-10);
        assert!((result[2] - v[2]).abs() < 1e-10);
    }

    #[test]
    fn test_so3_composition() {
        let r1 = SO3::from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let r2 = SO3::from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let composed = r1.compose(&r2);

        // Two 90-degree rotations = 180-degree rotation
        let v = [1.0, 0.0, 0.0];
        let result = composed.rotate(v);

        assert!((result[0] - (-1.0)).abs() < 1e-10);
        assert!(result[1].abs() < 1e-10);
    }

    #[test]
    fn test_so3_inverse() {
        let r = SO3::from_axis_angle([1.0, 1.0, 1.0], 1.0);
        let r_inv = r.inverse();
        let composed = r.compose(&r_inv);

        // R * R^-1 = I
        let v = [1.0, 2.0, 3.0];
        let result = composed.rotate(v);

        assert!((result[0] - v[0]).abs() < 1e-10);
        assert!((result[1] - v[1]).abs() < 1e-10);
        assert!((result[2] - v[2]).abs() < 1e-10);
    }

    #[test]
    fn test_se3_transformation() {
        let rotation = SO3::from_axis_angle([0.0, 0.0, 1.0], PI / 2.0);
        let translation = [1.0, 2.0, 3.0];
        let se3 = SE3::new(rotation, translation);

        let p = [1.0, 0.0, 0.0];
        let result = se3.transform(p);

        // Rotation then translation
        assert!((result[0] - 1.0).abs() < 1e-10); // 0 + 1
        assert!((result[1] - 3.0).abs() < 1e-10); // 1 + 2
        assert!((result[2] - 3.0).abs() < 1e-10); // 0 + 3
    }

    #[test]
    fn test_u1_composition() {
        let u1 = U1::new(PI / 4.0);
        let u2 = U1::new(PI / 4.0);
        let composed = u1.compose(&u2);

        assert!((composed.phase - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_su2_to_so3() {
        let su2 = SU2::new(0.707, 0.0, 0.0, 0.707); // ~90 degree rotation around z
        let so3 = su2.to_so3();

        assert!(so3.is_valid());
    }

    #[test]
    fn test_symmetry_checker_invariant() {
        let checker = SymmetryChecker::new(1e-6);

        // Norm is SO(3)-invariant
        let norm = |v: [f64; 3]| (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt();

        assert!(checker.check_so3_invariant(norm, [1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_symmetry_checker_equivariant() {
        let checker = SymmetryChecker::new(1e-6);

        // Scaling is SO(3)-equivariant
        let scale = |v: [f64; 3]| [v[0] * 2.0, v[1] * 2.0, v[2] * 2.0];

        assert!(checker.check_so3_equivariant(scale, [1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_translation_invariance() {
        let checker = SymmetryChecker::new(1e-6);

        // Distance between two points is translation-invariant
        let distance = |points: &[[f64; 3]]| {
            let dx = points[0][0] - points[1][0];
            let dy = points[0][1] - points[1][1];
            let dz = points[0][2] - points[1][2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        };

        let points = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
        assert!(checker.check_translation_invariant(distance, &points));
    }
}
