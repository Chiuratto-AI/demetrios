// matrix.d - Fixed-size matrix types for Demetrios linear algebra
//
// Provides small matrix types (Mat2, Mat3, Mat4) and operations commonly
// needed in scientific computing: multiplication, transpose, determinant, inverse.
//
// These are stack-allocated, row-major storage for cache efficiency.

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn abs_val(x: f64) -> f64 {
    if x < 0.0 {
        return 0.0 - x
    }
    return x
}

fn sqrt_val(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    let mut y = x
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    return y
}

// ============================================================================
// MAT2 - 2x2 MATRIX
// ============================================================================
// Row-major: [m00 m01]
//            [m10 m11]

struct Mat2 {
    m00: f64, m01: f64,
    m10: f64, m11: f64
}

fn mat2_new(m00: f64, m01: f64, m10: f64, m11: f64) -> Mat2 {
    return Mat2 { m00: m00, m01: m01, m10: m10, m11: m11 }
}

fn mat2_zero() -> Mat2 {
    return Mat2 { m00: 0.0, m01: 0.0, m10: 0.0, m11: 0.0 }
}

fn mat2_identity() -> Mat2 {
    return Mat2 { m00: 1.0, m01: 0.0, m10: 0.0, m11: 1.0 }
}

fn mat2_diag(d0: f64, d1: f64) -> Mat2 {
    return Mat2 { m00: d0, m01: 0.0, m10: 0.0, m11: d1 }
}

fn mat2_add(a: Mat2, b: Mat2) -> Mat2 {
    return Mat2 {
        m00: a.m00 + b.m00, m01: a.m01 + b.m01,
        m10: a.m10 + b.m10, m11: a.m11 + b.m11
    }
}

fn mat2_sub(a: Mat2, b: Mat2) -> Mat2 {
    return Mat2 {
        m00: a.m00 - b.m00, m01: a.m01 - b.m01,
        m10: a.m10 - b.m10, m11: a.m11 - b.m11
    }
}

fn mat2_scale(m: Mat2, s: f64) -> Mat2 {
    return Mat2 {
        m00: m.m00 * s, m01: m.m01 * s,
        m10: m.m10 * s, m11: m.m11 * s
    }
}

fn mat2_neg(m: Mat2) -> Mat2 {
    return Mat2 {
        m00: 0.0 - m.m00, m01: 0.0 - m.m01,
        m10: 0.0 - m.m10, m11: 0.0 - m.m11
    }
}

fn mat2_transpose(m: Mat2) -> Mat2 {
    return Mat2 { m00: m.m00, m01: m.m10, m10: m.m01, m11: m.m11 }
}

fn mat2_mul(a: Mat2, b: Mat2) -> Mat2 {
    return Mat2 {
        m00: a.m00 * b.m00 + a.m01 * b.m10,
        m01: a.m00 * b.m01 + a.m01 * b.m11,
        m10: a.m10 * b.m00 + a.m11 * b.m10,
        m11: a.m10 * b.m01 + a.m11 * b.m11
    }
}

// Mat2 * Vec2 (using named struct fields)
struct Vec2 {
    x: f64,
    y: f64
}

fn mat2_vec_mul(m: Mat2, v: Vec2) -> Vec2 {
    return Vec2 {
        x: m.m00 * v.x + m.m01 * v.y,
        y: m.m10 * v.x + m.m11 * v.y
    }
}

fn mat2_det(m: Mat2) -> f64 {
    return m.m00 * m.m11 - m.m01 * m.m10
}

fn mat2_trace(m: Mat2) -> f64 {
    return m.m00 + m.m11
}

fn mat2_inverse(m: Mat2) -> Mat2 {
    let d = mat2_det(m)
    if abs_val(d) < 0.0000000001 {
        // Return identity for singular matrix (caller should check det)
        return mat2_identity()
    }
    let inv_d = 1.0 / d
    return Mat2 {
        m00: m.m11 * inv_d,
        m01: (0.0 - m.m01) * inv_d,
        m10: (0.0 - m.m10) * inv_d,
        m11: m.m00 * inv_d
    }
}

fn mat2_frobenius_norm(m: Mat2) -> f64 {
    return sqrt_val(m.m00*m.m00 + m.m01*m.m01 + m.m10*m.m10 + m.m11*m.m11)
}

// ============================================================================
// MAT3 - 3x3 MATRIX
// ============================================================================
// Row-major: [m00 m01 m02]
//            [m10 m11 m12]
//            [m20 m21 m22]

struct Mat3 {
    m00: f64, m01: f64, m02: f64,
    m10: f64, m11: f64, m12: f64,
    m20: f64, m21: f64, m22: f64
}

fn mat3_new(
    m00: f64, m01: f64, m02: f64,
    m10: f64, m11: f64, m12: f64,
    m20: f64, m21: f64, m22: f64
) -> Mat3 {
    return Mat3 {
        m00: m00, m01: m01, m02: m02,
        m10: m10, m11: m11, m12: m12,
        m20: m20, m21: m21, m22: m22
    }
}

fn mat3_zero() -> Mat3 {
    return Mat3 {
        m00: 0.0, m01: 0.0, m02: 0.0,
        m10: 0.0, m11: 0.0, m12: 0.0,
        m20: 0.0, m21: 0.0, m22: 0.0
    }
}

fn mat3_identity() -> Mat3 {
    return Mat3 {
        m00: 1.0, m01: 0.0, m02: 0.0,
        m10: 0.0, m11: 1.0, m12: 0.0,
        m20: 0.0, m21: 0.0, m22: 1.0
    }
}

fn mat3_diag(d0: f64, d1: f64, d2: f64) -> Mat3 {
    return Mat3 {
        m00: d0, m01: 0.0, m02: 0.0,
        m10: 0.0, m11: d1, m12: 0.0,
        m20: 0.0, m21: 0.0, m22: d2
    }
}

fn mat3_add(a: Mat3, b: Mat3) -> Mat3 {
    return Mat3 {
        m00: a.m00 + b.m00, m01: a.m01 + b.m01, m02: a.m02 + b.m02,
        m10: a.m10 + b.m10, m11: a.m11 + b.m11, m12: a.m12 + b.m12,
        m20: a.m20 + b.m20, m21: a.m21 + b.m21, m22: a.m22 + b.m22
    }
}

fn mat3_sub(a: Mat3, b: Mat3) -> Mat3 {
    return Mat3 {
        m00: a.m00 - b.m00, m01: a.m01 - b.m01, m02: a.m02 - b.m02,
        m10: a.m10 - b.m10, m11: a.m11 - b.m11, m12: a.m12 - b.m12,
        m20: a.m20 - b.m20, m21: a.m21 - b.m21, m22: a.m22 - b.m22
    }
}

fn mat3_scale(m: Mat3, s: f64) -> Mat3 {
    return Mat3 {
        m00: m.m00 * s, m01: m.m01 * s, m02: m.m02 * s,
        m10: m.m10 * s, m11: m.m11 * s, m12: m.m12 * s,
        m20: m.m20 * s, m21: m.m21 * s, m22: m.m22 * s
    }
}

fn mat3_neg(m: Mat3) -> Mat3 {
    return Mat3 {
        m00: 0.0 - m.m00, m01: 0.0 - m.m01, m02: 0.0 - m.m02,
        m10: 0.0 - m.m10, m11: 0.0 - m.m11, m12: 0.0 - m.m12,
        m20: 0.0 - m.m20, m21: 0.0 - m.m21, m22: 0.0 - m.m22
    }
}

fn mat3_transpose(m: Mat3) -> Mat3 {
    return Mat3 {
        m00: m.m00, m01: m.m10, m02: m.m20,
        m10: m.m01, m11: m.m11, m12: m.m21,
        m20: m.m02, m21: m.m12, m22: m.m22
    }
}

fn mat3_mul(a: Mat3, b: Mat3) -> Mat3 {
    return Mat3 {
        m00: a.m00*b.m00 + a.m01*b.m10 + a.m02*b.m20,
        m01: a.m00*b.m01 + a.m01*b.m11 + a.m02*b.m21,
        m02: a.m00*b.m02 + a.m01*b.m12 + a.m02*b.m22,
        m10: a.m10*b.m00 + a.m11*b.m10 + a.m12*b.m20,
        m11: a.m10*b.m01 + a.m11*b.m11 + a.m12*b.m21,
        m12: a.m10*b.m02 + a.m11*b.m12 + a.m12*b.m22,
        m20: a.m20*b.m00 + a.m21*b.m10 + a.m22*b.m20,
        m21: a.m20*b.m01 + a.m21*b.m11 + a.m22*b.m21,
        m22: a.m20*b.m02 + a.m21*b.m12 + a.m22*b.m22
    }
}

// Mat3 * Vec3
struct Vec3 {
    x: f64,
    y: f64,
    z: f64
}

fn mat3_vec_mul(m: Mat3, v: Vec3) -> Vec3 {
    return Vec3 {
        x: m.m00*v.x + m.m01*v.y + m.m02*v.z,
        y: m.m10*v.x + m.m11*v.y + m.m12*v.z,
        z: m.m20*v.x + m.m21*v.y + m.m22*v.z
    }
}

fn mat3_det(m: Mat3) -> f64 {
    // Sarrus rule / cofactor expansion
    return m.m00 * (m.m11*m.m22 - m.m12*m.m21)
         - m.m01 * (m.m10*m.m22 - m.m12*m.m20)
         + m.m02 * (m.m10*m.m21 - m.m11*m.m20)
}

fn mat3_trace(m: Mat3) -> f64 {
    return m.m00 + m.m11 + m.m22
}

fn mat3_inverse(m: Mat3) -> Mat3 {
    let d = mat3_det(m)
    if abs_val(d) < 0.0000000001 {
        return mat3_identity()
    }
    let inv_d = 1.0 / d

    // Cofactor matrix, transposed
    return Mat3 {
        m00: (m.m11*m.m22 - m.m12*m.m21) * inv_d,
        m01: (m.m02*m.m21 - m.m01*m.m22) * inv_d,
        m02: (m.m01*m.m12 - m.m02*m.m11) * inv_d,
        m10: (m.m12*m.m20 - m.m10*m.m22) * inv_d,
        m11: (m.m00*m.m22 - m.m02*m.m20) * inv_d,
        m12: (m.m02*m.m10 - m.m00*m.m12) * inv_d,
        m20: (m.m10*m.m21 - m.m11*m.m20) * inv_d,
        m21: (m.m01*m.m20 - m.m00*m.m21) * inv_d,
        m22: (m.m00*m.m11 - m.m01*m.m10) * inv_d
    }
}

fn mat3_frobenius_norm(m: Mat3) -> f64 {
    return sqrt_val(m.m00*m.m00 + m.m01*m.m01 + m.m02*m.m02
                  + m.m10*m.m10 + m.m11*m.m11 + m.m12*m.m12
                  + m.m20*m.m20 + m.m21*m.m21 + m.m22*m.m22)
}

// ============================================================================
// MAT4 - 4x4 MATRIX (for transformations, projections)
// ============================================================================

struct Mat4 {
    m00: f64, m01: f64, m02: f64, m03: f64,
    m10: f64, m11: f64, m12: f64, m13: f64,
    m20: f64, m21: f64, m22: f64, m23: f64,
    m30: f64, m31: f64, m32: f64, m33: f64
}

fn mat4_zero() -> Mat4 {
    return Mat4 {
        m00: 0.0, m01: 0.0, m02: 0.0, m03: 0.0,
        m10: 0.0, m11: 0.0, m12: 0.0, m13: 0.0,
        m20: 0.0, m21: 0.0, m22: 0.0, m23: 0.0,
        m30: 0.0, m31: 0.0, m32: 0.0, m33: 0.0
    }
}

fn mat4_identity() -> Mat4 {
    return Mat4 {
        m00: 1.0, m01: 0.0, m02: 0.0, m03: 0.0,
        m10: 0.0, m11: 1.0, m12: 0.0, m13: 0.0,
        m20: 0.0, m21: 0.0, m22: 1.0, m23: 0.0,
        m30: 0.0, m31: 0.0, m32: 0.0, m33: 1.0
    }
}

fn mat4_diag(d0: f64, d1: f64, d2: f64, d3: f64) -> Mat4 {
    return Mat4 {
        m00: d0, m01: 0.0, m02: 0.0, m03: 0.0,
        m10: 0.0, m11: d1, m12: 0.0, m13: 0.0,
        m20: 0.0, m21: 0.0, m22: d2, m23: 0.0,
        m30: 0.0, m31: 0.0, m32: 0.0, m33: d3
    }
}

fn mat4_add(a: Mat4, b: Mat4) -> Mat4 {
    return Mat4 {
        m00: a.m00+b.m00, m01: a.m01+b.m01, m02: a.m02+b.m02, m03: a.m03+b.m03,
        m10: a.m10+b.m10, m11: a.m11+b.m11, m12: a.m12+b.m12, m13: a.m13+b.m13,
        m20: a.m20+b.m20, m21: a.m21+b.m21, m22: a.m22+b.m22, m23: a.m23+b.m23,
        m30: a.m30+b.m30, m31: a.m31+b.m31, m32: a.m32+b.m32, m33: a.m33+b.m33
    }
}

fn mat4_scale(m: Mat4, s: f64) -> Mat4 {
    return Mat4 {
        m00: m.m00*s, m01: m.m01*s, m02: m.m02*s, m03: m.m03*s,
        m10: m.m10*s, m11: m.m11*s, m12: m.m12*s, m13: m.m13*s,
        m20: m.m20*s, m21: m.m21*s, m22: m.m22*s, m23: m.m23*s,
        m30: m.m30*s, m31: m.m31*s, m32: m.m32*s, m33: m.m33*s
    }
}

fn mat4_transpose(m: Mat4) -> Mat4 {
    return Mat4 {
        m00: m.m00, m01: m.m10, m02: m.m20, m03: m.m30,
        m10: m.m01, m11: m.m11, m12: m.m21, m13: m.m31,
        m20: m.m02, m21: m.m12, m22: m.m22, m23: m.m32,
        m30: m.m03, m31: m.m13, m32: m.m23, m33: m.m33
    }
}

fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    return Mat4 {
        m00: a.m00*b.m00 + a.m01*b.m10 + a.m02*b.m20 + a.m03*b.m30,
        m01: a.m00*b.m01 + a.m01*b.m11 + a.m02*b.m21 + a.m03*b.m31,
        m02: a.m00*b.m02 + a.m01*b.m12 + a.m02*b.m22 + a.m03*b.m32,
        m03: a.m00*b.m03 + a.m01*b.m13 + a.m02*b.m23 + a.m03*b.m33,
        m10: a.m10*b.m00 + a.m11*b.m10 + a.m12*b.m20 + a.m13*b.m30,
        m11: a.m10*b.m01 + a.m11*b.m11 + a.m12*b.m21 + a.m13*b.m31,
        m12: a.m10*b.m02 + a.m11*b.m12 + a.m12*b.m22 + a.m13*b.m32,
        m13: a.m10*b.m03 + a.m11*b.m13 + a.m12*b.m23 + a.m13*b.m33,
        m20: a.m20*b.m00 + a.m21*b.m10 + a.m22*b.m20 + a.m23*b.m30,
        m21: a.m20*b.m01 + a.m21*b.m11 + a.m22*b.m21 + a.m23*b.m31,
        m22: a.m20*b.m02 + a.m21*b.m12 + a.m22*b.m22 + a.m23*b.m32,
        m23: a.m20*b.m03 + a.m21*b.m13 + a.m22*b.m23 + a.m23*b.m33,
        m30: a.m30*b.m00 + a.m31*b.m10 + a.m32*b.m20 + a.m33*b.m30,
        m31: a.m30*b.m01 + a.m31*b.m11 + a.m32*b.m21 + a.m33*b.m31,
        m32: a.m30*b.m02 + a.m31*b.m12 + a.m32*b.m22 + a.m33*b.m32,
        m33: a.m30*b.m03 + a.m31*b.m13 + a.m32*b.m23 + a.m33*b.m33
    }
}

struct Vec4 {
    x: f64,
    y: f64,
    z: f64,
    w: f64
}

fn mat4_vec_mul(m: Mat4, v: Vec4) -> Vec4 {
    return Vec4 {
        x: m.m00*v.x + m.m01*v.y + m.m02*v.z + m.m03*v.w,
        y: m.m10*v.x + m.m11*v.y + m.m12*v.z + m.m13*v.w,
        z: m.m20*v.x + m.m21*v.y + m.m22*v.z + m.m23*v.w,
        w: m.m30*v.x + m.m31*v.y + m.m32*v.z + m.m33*v.w
    }
}

fn mat4_trace(m: Mat4) -> f64 {
    return m.m00 + m.m11 + m.m22 + m.m33
}

// ============================================================================
// TESTS
// ============================================================================

fn main() -> i32 {
    println("=== Demetrios Matrix Types Test ===")
    println("")

    // Test Mat2
    println("Testing Mat2:")
    let m2 = mat2_new(1.0, 2.0, 3.0, 4.0)
    let det2 = mat2_det(m2)
    println("  det(M2) = ")
    println(det2)

    let m2_inv = mat2_inverse(m2)
    let m2_check = mat2_mul(m2, m2_inv)
    println("  M2 * M2^-1 diag = ")
    println(m2_check.m00)
    println(m2_check.m11)
    println("")

    // Test Mat3
    println("Testing Mat3:")
    let m3 = mat3_new(
        1.0, 2.0, 3.0,
        0.0, 1.0, 4.0,
        5.0, 6.0, 0.0
    )
    let det3 = mat3_det(m3)
    println("  det(M3) = ")
    println(det3)

    let m3_inv = mat3_inverse(m3)
    let m3_check = mat3_mul(m3, m3_inv)
    println("  M3 * M3^-1 trace = ")
    println(mat3_trace(m3_check))
    println("")

    // Test Mat3 * Vec3
    println("Testing Mat3 * Vec3:")
    let v3 = Vec3 { x: 1.0, y: 2.0, z: 3.0 }
    let result = mat3_vec_mul(m3, v3)
    println("  result = ")
    println(result.x)
    println(result.y)
    println(result.z)

    // Expected: M3 * v3 = [1*1+2*2+3*3, 0*1+1*2+4*3, 5*1+6*2+0*3]
    //                   = [1+4+9, 0+2+12, 5+12+0] = [14, 14, 17]
    let expected_x = 14.0
    let expected_y = 14.0
    let expected_z = 17.0
    println("")

    // Test Mat4
    println("Testing Mat4:")
    let m4 = mat4_identity()
    let m4_scaled = mat4_scale(m4, 2.0)
    println("  2*I trace = ")
    println(mat4_trace(m4_scaled))
    println("")

    // Verify results
    // det(M2) = 1*4 - 2*3 = -2
    let det2_expected = 0.0 - 2.0
    let det2_err = abs_val(det2 - det2_expected)

    // det(M3) = 1*(1*0 - 4*6) - 2*(0*0 - 4*5) + 3*(0*6 - 1*5)
    //         = 1*(-24) - 2*(-20) + 3*(-5) = -24 + 40 - 15 = 1
    let det3_expected = 1.0
    let det3_err = abs_val(det3 - det3_expected)

    // M2 * M2^-1 should be identity (diag = 1, 1)
    let m2_diag_err = abs_val(m2_check.m00 - 1.0) + abs_val(m2_check.m11 - 1.0)

    // Mat-vec multiplication
    let mv_err = abs_val(result.x - expected_x) + abs_val(result.y - expected_y) + abs_val(result.z - expected_z)

    if det2_err < 0.0001 && det3_err < 0.0001 && m2_diag_err < 0.0001 && mv_err < 0.0001 {
        println("TEST PASSED: All matrix operations correct")
        return 0
    } else {
        println("TEST FAILED: Matrix operation errors")
        println("  det2_err = ")
        println(det2_err)
        println("  det3_err = ")
        println(det3_err)
        println("  m2_diag_err = ")
        println(m2_diag_err)
        println("  mv_err = ")
        println(mv_err)
        return 1
    }
}
