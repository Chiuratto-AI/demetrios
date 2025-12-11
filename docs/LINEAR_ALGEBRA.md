# Native Linear Algebra in Demetrios

Demetrios provides first-class support for linear algebra primitives with SIMD-optimized code generation.

## Types

### Vectors

```d
vec2    // 2D vector (x, y) - 2x f32
vec3    // 3D vector (x, y, z) - 3x f32
vec4    // 4D vector (x, y, z, w) - 4x f32
```

### Matrices (Column-Major)

```d
mat2    // 2x2 matrix - 4x f32
mat3    // 3x3 matrix - 9x f32
mat4    // 4x4 matrix - 16x f32
```

### Quaternions

```d
quat    // Quaternion (x, y, z, w) - 4x f32
```

## Constructors

```d
// Vectors
let v2 = vec2(1.0, 2.0);
let v3 = vec3(1.0, 2.0, 3.0);
let v4 = vec4(1.0, 2.0, 3.0, 4.0);

// Matrices (column-major order)
let m2 = mat2(1.0, 0.0,   // column 0
              0.0, 1.0);  // column 1

let m4 = mat4(
    1.0, 0.0, 0.0, 0.0,  // column 0
    0.0, 1.0, 0.0, 0.0,  // column 1
    0.0, 0.0, 1.0, 0.0,  // column 2
    0.0, 0.0, 0.0, 1.0   // column 3
);

// Quaternions (x, y, z, w)
let q = quat(0.0, 0.0, 0.0, 1.0);  // identity
let qi = quat_identity();          // also identity
```

## Vector Operations

```d
// Dot product
let d = dot(v1, v2);           // f32

// Cross product (vec3 only)
let c = cross(v1, v2);         // vec3

// Normalization
let n = normalize(v);          // unit vector

// Length
let len = length(v);           // |v|
let len_sq = length_squared(v); // |v|^2
```

## Matrix Operations

```d
// Matrix multiplication
let m = mat_mul(m1, m2);       // m1 * m2

// Transpose
let mt = transpose(m);         // m^T

// Inverse
let mi = inverse(m);           // m^(-1)

// Determinant
let det = determinant(m);      // |m|
```

## Quaternion Operations

```d
// Multiplication (Hamilton product)
let q = quat_mul(q1, q2);      // q1 * q2

// Conjugate
let qc = quat_conj(q);         // q*

// Inverse
let qi = quat_inv(q);          // q^(-1)

// Normalize
let qn = quat_normalize(q);    // q / |q|

// Identity
let id = quat_identity();      // (0, 0, 0, 1)
```

## Interpolation

```d
// Linear interpolation (vectors)
let v = lerp(v1, v2, 0.5);     // v1 + t*(v2-v1)

// Spherical linear interpolation (quaternions)
let q = slerp(q1, q2, 0.5);    // smooth rotation interpolation
```

## Conversions

```d
// Quaternion <-> Euler angles
let euler = quat_to_euler(q);  // vec3 (pitch, yaw, roll)
let q = euler_to_quat(euler);  // quat

// Quaternion <-> Rotation matrix
let m3 = quat_to_mat3(q);      // 3x3 rotation matrix
let m4 = quat_to_mat4(q);      // 4x4 transformation matrix
let q = mat3_to_quat(m3);      // quaternion from rotation matrix
```

## SIMD Optimization

All linear algebra operations are compiled to SIMD instructions when using the Cranelift JIT backend:

- `vec2`, `vec3`, `vec4`, `quat` use 128-bit F32X4 SIMD registers
- Operations like `dot`, `cross`, `normalize` use vectorized instructions
- The Hamilton product for quaternions is fully SIMD-optimized

Enable with: `cargo build --features jit`

## Example: 3D Rotation

```d
fn rotate_point(point: vec3, axis: vec3, angle: f32) -> vec3 {
    // Create rotation quaternion from axis-angle
    let half_angle = angle * 0.5;
    let s = sin(half_angle);
    let c = cos(half_angle);
    
    let axis_n = normalize(axis);
    let qx = axis_n.x * s;
    let qy = axis_n.y * s;
    let qz = axis_n.z * s;
    
    let rotation = quat(qx, qy, qz, c);
    
    // Rotate point: q * p * q^(-1)
    return quat_rotate_vec(rotation, point);
}
```

## See Also

- [Quaternion Embeddings](QUATERNION_EMBEDDINGS.md) - Knowledge graph embeddings using quaternions
