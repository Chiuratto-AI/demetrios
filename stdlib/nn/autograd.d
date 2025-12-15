// autograd.d - Reverse-Mode Automatic Differentiation (Backpropagation)
//
// Implements a Wengert tape for gradient computation.
// Uses struct creation (not mutation) to work with Demetrios semantics.

// ============================================================================
// MATH HELPERS
// ============================================================================

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    let mut y = x
    let mut i = 0
    while i < 15 { y = 0.5 * (y + x / y); i = i + 1 }
    return y
}

fn exp_f64(x: f64) -> f64 {
    if x > 20.0 { return exp_f64(x / 2.0) * exp_f64(x / 2.0) }
    if x < 0.0 - 20.0 { return 1.0 / exp_f64(0.0 - x) }
    let mut sum = 1.0
    let mut term = 1.0
    let mut i = 1
    while i <= 20 { term = term * x / i; sum = sum + term; i = i + 1 }
    return sum
}

fn cos_f64(x: f64) -> f64 {
    let pi = 3.141592653589793
    let mut y = x
    while y > pi { y = y - 2.0 * pi }
    while y < 0.0 - pi { y = y + 2.0 * pi }
    let y2 = y * y
    let mut sum = 1.0
    let mut term = 1.0
    term = term * (0.0 - y2) / 2.0; sum = sum + term
    term = term * (0.0 - y2) / 12.0; sum = sum + term
    term = term * (0.0 - y2) / 30.0; sum = sum + term
    term = term * (0.0 - y2) / 56.0; sum = sum + term
    term = term * (0.0 - y2) / 90.0; sum = sum + term
    return sum
}

fn sin_f64(x: f64) -> f64 {
    let pi = 3.141592653589793
    let mut y = x
    while y > pi { y = y - 2.0 * pi }
    while y < 0.0 - pi { y = y + 2.0 * pi }
    let y2 = y * y
    let mut sum = y
    let mut term = y
    term = term * (0.0 - y2) / 6.0; sum = sum + term
    term = term * (0.0 - y2) / 20.0; sum = sum + term
    term = term * (0.0 - y2) / 42.0; sum = sum + term
    term = term * (0.0 - y2) / 72.0; sum = sum + term
    return sum
}

// ============================================================================
// OPERATION CODES
// ============================================================================

fn OP_VAR() -> i64 { return 1 }
fn OP_ADD() -> i64 { return 2 }
fn OP_MUL() -> i64 { return 4 }
fn OP_DIV() -> i64 { return 5 }
fn OP_EXP() -> i64 { return 7 }
fn OP_SQRT() -> i64 { return 9 }
fn OP_SIN() -> i64 { return 11 }
fn OP_SIGMOID() -> i64 { return 15 }
fn OP_RELU() -> i64 { return 16 }
fn OP_TANH() -> i64 { return 17 }
fn OP_LEAKY_RELU() -> i64 { return 18 }

// Leaky ReLU slope for negative inputs (standard value)
fn LEAKY_ALPHA() -> f64 { return 0.01 }

// ============================================================================
// TAPE STRUCTURE - 6 slots for simplicity
// ============================================================================

struct Tape {
    // Slot 0
    op0: i64, a10: i64, a20: i64, v0: f64, g0: f64,
    // Slot 1
    op1: i64, a11: i64, a21: i64, v1: f64, g1: f64,
    // Slot 2
    op2: i64, a12: i64, a22: i64, v2: f64, g2: f64,
    // Slot 3
    op3: i64, a13: i64, a23: i64, v3: f64, g3: f64,
    // Slot 4
    op4: i64, a14: i64, a24: i64, v4: f64, g4: f64,
    // Slot 5
    op5: i64, a15: i64, a25: i64, v5: f64, g5: f64,
    len: i64
}

fn tape_new() -> Tape {
    return Tape {
        op0: 0, a10: 0, a20: 0, v0: 0.0, g0: 0.0,
        op1: 0, a11: 0, a21: 0, v1: 0.0, g1: 0.0,
        op2: 0, a12: 0, a22: 0, v2: 0.0, g2: 0.0,
        op3: 0, a13: 0, a23: 0, v3: 0.0, g3: 0.0,
        op4: 0, a14: 0, a24: 0, v4: 0.0, g4: 0.0,
        op5: 0, a15: 0, a25: 0, v5: 0.0, g5: 0.0,
        len: 0
    }
}

// ============================================================================
// GETTERS
// ============================================================================

fn get_v(t: Tape, i: i64) -> f64 {
    if i == 0 { return t.v0 }
    if i == 1 { return t.v1 }
    if i == 2 { return t.v2 }
    if i == 3 { return t.v3 }
    if i == 4 { return t.v4 }
    if i == 5 { return t.v5 }
    return 0.0
}

fn get_g(t: Tape, i: i64) -> f64 {
    if i == 0 { return t.g0 }
    if i == 1 { return t.g1 }
    if i == 2 { return t.g2 }
    if i == 3 { return t.g3 }
    if i == 4 { return t.g4 }
    if i == 5 { return t.g5 }
    return 0.0
}

fn get_op(t: Tape, i: i64) -> i64 {
    if i == 0 { return t.op0 }
    if i == 1 { return t.op1 }
    if i == 2 { return t.op2 }
    if i == 3 { return t.op3 }
    if i == 4 { return t.op4 }
    if i == 5 { return t.op5 }
    return 0
}

fn get_a1(t: Tape, i: i64) -> i64 {
    if i == 0 { return t.a10 }
    if i == 1 { return t.a11 }
    if i == 2 { return t.a12 }
    if i == 3 { return t.a13 }
    if i == 4 { return t.a14 }
    if i == 5 { return t.a15 }
    return 0
}

fn get_a2(t: Tape, i: i64) -> i64 {
    if i == 0 { return t.a20 }
    if i == 1 { return t.a21 }
    if i == 2 { return t.a22 }
    if i == 3 { return t.a23 }
    if i == 4 { return t.a24 }
    if i == 5 { return t.a25 }
    return 0
}

// ============================================================================
// SETTERS (create new struct with modified field)
// ============================================================================

fn set_g0(t: Tape, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: v,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: t.len }
}

fn set_g1(t: Tape, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: v,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: t.len }
}

fn set_g2(t: Tape, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: v,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: t.len }
}

fn set_g3(t: Tape, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: v,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: t.len }
}

fn set_g4(t: Tape, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: v,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: t.len }
}

fn set_g5(t: Tape, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: v, len: t.len }
}

fn set_g(t: Tape, i: i64, v: f64) -> Tape {
    if i == 0 { return set_g0(t, v) }
    if i == 1 { return set_g1(t, v) }
    if i == 2 { return set_g2(t, v) }
    if i == 3 { return set_g3(t, v) }
    if i == 4 { return set_g4(t, v) }
    if i == 5 { return set_g5(t, v) }
    return t
}

fn add_g(t: Tape, i: i64, v: f64) -> Tape {
    if i < 0 { return t }  // Skip for negative indices (no parent)
    let old = get_g(t, i)
    return set_g(t, i, old + v)
}

// ============================================================================
// PUSH (create new tape with slot filled)
// ============================================================================

fn push0(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape { op0: op, a10: a1, a20: a2, v0: v, g0: 0.0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: 1 }
}

fn push1(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: op, a11: a1, a21: a2, v1: v, g1: 0.0,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: 2 }
}

fn push2(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: op, a12: a1, a22: a2, v2: v, g2: 0.0,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: 3 }
}

fn push3(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: op, a13: a1, a23: a2, v3: v, g3: 0.0,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: 4 }
}

fn push4(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: op, a14: a1, a24: a2, v4: v, g4: 0.0,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: t.g5, len: 5 }
}

fn push5(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    return Tape { op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: t.g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: t.g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: t.g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: t.g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: t.g4,
        op5: op, a15: a1, a25: a2, v5: v, g5: 0.0, len: 6 }
}

fn push(t: Tape, op: i64, a1: i64, a2: i64, v: f64) -> Tape {
    let i = t.len
    if i == 0 { return push0(t, op, a1, a2, v) }
    if i == 1 { return push1(t, op, a1, a2, v) }
    if i == 2 { return push2(t, op, a1, a2, v) }
    if i == 3 { return push3(t, op, a1, a2, v) }
    if i == 4 { return push4(t, op, a1, a2, v) }
    if i == 5 { return push5(t, op, a1, a2, v) }
    return t
}

// ============================================================================
// OPERATIONS
// ============================================================================

fn tvar(t: Tape, v: f64) -> Tape { return push(t, OP_VAR(), 0 - 1, 0 - 1, v) }

fn tadd(t: Tape, a: i64, b: i64) -> Tape {
    return push(t, OP_ADD(), a, b, get_v(t, a) + get_v(t, b))
}

fn tmul(t: Tape, a: i64, b: i64) -> Tape {
    return push(t, OP_MUL(), a, b, get_v(t, a) * get_v(t, b))
}

fn tdiv(t: Tape, a: i64, b: i64) -> Tape {
    return push(t, OP_DIV(), a, b, get_v(t, a) / get_v(t, b))
}

fn texp(t: Tape, a: i64) -> Tape {
    return push(t, OP_EXP(), a, 0 - 1, exp_f64(get_v(t, a)))
}

fn tsqrt(t: Tape, a: i64) -> Tape {
    return push(t, OP_SQRT(), a, 0 - 1, sqrt_f64(get_v(t, a)))
}

fn tsin(t: Tape, a: i64) -> Tape {
    return push(t, OP_SIN(), a, 0 - 1, sin_f64(get_v(t, a)))
}

fn tsigmoid(t: Tape, a: i64) -> Tape {
    let av = get_v(t, a)
    return push(t, OP_SIGMOID(), a, 0 - 1, 1.0 / (1.0 + exp_f64(0.0 - av)))
}

fn trelu(t: Tape, a: i64) -> Tape {
    let av = get_v(t, a)
    let rv = if av > 0.0 { av } else { 0.0 }
    return push(t, OP_RELU(), a, 0 - 1, rv)
}

fn ttanh(t: Tape, a: i64) -> Tape {
    let av = get_v(t, a)
    // tanh(x) = (e^x - e^-x) / (e^x + e^-x)
    let ep = exp_f64(av)
    let en = exp_f64(0.0 - av)
    let tv = (ep - en) / (ep + en)
    return push(t, OP_TANH(), a, 0 - 1, tv)
}

fn tleaky_relu(t: Tape, a: i64) -> Tape {
    let av = get_v(t, a)
    let alpha = LEAKY_ALPHA()
    let rv = if av > 0.0 { av } else { alpha * av }
    return push(t, OP_LEAKY_RELU(), a, 0 - 1, rv)
}

// ============================================================================
// BACKWARD
// ============================================================================

// Process a single backward step and return new tape
// Note: Uses direct struct creation to avoid Demetrios compiler bug with
// function parameters inside while loops
fn backward_step(t: Tape, i: i64) -> Tape {
    let op = get_op(t, i)
    let a1 = get_a1(t, i)
    let a2 = get_a2(t, i)
    let v = get_v(t, i)
    let dout = get_g(t, i)

    if abs_f64(dout) < 0.0000000001 {
        return t
    }

    // Read current gradients
    let cur_g0 = t.g0
    let cur_g1 = t.g1
    let cur_g2 = t.g2
    let cur_g3 = t.g3
    let cur_g4 = t.g4
    let cur_g5 = t.g5

    // Compute new gradients
    let mut new_g0 = cur_g0
    let mut new_g1 = cur_g1
    let mut new_g2 = cur_g2
    let mut new_g3 = cur_g3
    let mut new_g4 = cur_g4
    let mut new_g5 = cur_g5

    if op == OP_ADD() {
        // d(a+b)/da = 1, d(a+b)/db = 1
        if a1 == 0 { new_g0 = new_g0 + dout }
        if a1 == 1 { new_g1 = new_g1 + dout }
        if a1 == 2 { new_g2 = new_g2 + dout }
        if a1 == 3 { new_g3 = new_g3 + dout }
        if a1 == 4 { new_g4 = new_g4 + dout }
        if a1 == 5 { new_g5 = new_g5 + dout }
        if a2 == 0 { new_g0 = new_g0 + dout }
        if a2 == 1 { new_g1 = new_g1 + dout }
        if a2 == 2 { new_g2 = new_g2 + dout }
        if a2 == 3 { new_g3 = new_g3 + dout }
        if a2 == 4 { new_g4 = new_g4 + dout }
        if a2 == 5 { new_g5 = new_g5 + dout }
    }
    if op == OP_MUL() {
        // d(a*b)/da = b, d(a*b)/db = a
        let av = get_v(t, a1)
        let bv = get_v(t, a2)
        let ga = dout * bv
        let gb = dout * av
        if a1 == 0 { new_g0 = new_g0 + ga }
        if a1 == 1 { new_g1 = new_g1 + ga }
        if a1 == 2 { new_g2 = new_g2 + ga }
        if a1 == 3 { new_g3 = new_g3 + ga }
        if a1 == 4 { new_g4 = new_g4 + ga }
        if a1 == 5 { new_g5 = new_g5 + ga }
        if a2 == 0 { new_g0 = new_g0 + gb }
        if a2 == 1 { new_g1 = new_g1 + gb }
        if a2 == 2 { new_g2 = new_g2 + gb }
        if a2 == 3 { new_g3 = new_g3 + gb }
        if a2 == 4 { new_g4 = new_g4 + gb }
        if a2 == 5 { new_g5 = new_g5 + gb }
    }
    if op == OP_DIV() {
        // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        let av = get_v(t, a1)
        let bv = get_v(t, a2)
        let ga = dout / bv
        let gb = 0.0 - dout * av / (bv * bv)
        if a1 == 0 { new_g0 = new_g0 + ga }
        if a1 == 1 { new_g1 = new_g1 + ga }
        if a1 == 2 { new_g2 = new_g2 + ga }
        if a1 == 3 { new_g3 = new_g3 + ga }
        if a1 == 4 { new_g4 = new_g4 + ga }
        if a1 == 5 { new_g5 = new_g5 + ga }
        if a2 == 0 { new_g0 = new_g0 + gb }
        if a2 == 1 { new_g1 = new_g1 + gb }
        if a2 == 2 { new_g2 = new_g2 + gb }
        if a2 == 3 { new_g3 = new_g3 + gb }
        if a2 == 4 { new_g4 = new_g4 + gb }
        if a2 == 5 { new_g5 = new_g5 + gb }
    }
    if op == OP_EXP() {
        // d(exp(a))/da = exp(a) = v
        let ga = dout * v
        if a1 == 0 { new_g0 = new_g0 + ga }
        if a1 == 1 { new_g1 = new_g1 + ga }
        if a1 == 2 { new_g2 = new_g2 + ga }
        if a1 == 3 { new_g3 = new_g3 + ga }
        if a1 == 4 { new_g4 = new_g4 + ga }
        if a1 == 5 { new_g5 = new_g5 + ga }
    }
    if op == OP_SQRT() {
        // d(sqrt(a))/da = 1/(2*sqrt(a)) = 1/(2v)
        let ga = dout / (2.0 * v)
        if a1 == 0 { new_g0 = new_g0 + ga }
        if a1 == 1 { new_g1 = new_g1 + ga }
        if a1 == 2 { new_g2 = new_g2 + ga }
        if a1 == 3 { new_g3 = new_g3 + ga }
        if a1 == 4 { new_g4 = new_g4 + ga }
        if a1 == 5 { new_g5 = new_g5 + ga }
    }
    if op == OP_SIN() {
        // d(sin(a))/da = cos(a)
        let av = get_v(t, a1)
        let ga = dout * cos_f64(av)
        if a1 == 0 { new_g0 = new_g0 + ga }
        if a1 == 1 { new_g1 = new_g1 + ga }
        if a1 == 2 { new_g2 = new_g2 + ga }
        if a1 == 3 { new_g3 = new_g3 + ga }
        if a1 == 4 { new_g4 = new_g4 + ga }
        if a1 == 5 { new_g5 = new_g5 + ga }
    }
    if op == OP_SIGMOID() {
        // d(sigmoid(a))/da = sigmoid(a) * (1 - sigmoid(a)) = v * (1-v)
        let ga = dout * v * (1.0 - v)
        if a1 == 0 { new_g0 = new_g0 + ga }
        if a1 == 1 { new_g1 = new_g1 + ga }
        if a1 == 2 { new_g2 = new_g2 + ga }
        if a1 == 3 { new_g3 = new_g3 + ga }
        if a1 == 4 { new_g4 = new_g4 + ga }
        if a1 == 5 { new_g5 = new_g5 + ga }
    }
    if op == OP_RELU() {
        // d(relu(a))/da = 1 if a > 0 else 0
        // Note: v = relu(input), so v > 0 iff input > 0
        let ga = if v > 0.0 { dout } else { 0.0 }
        if a1 == 0 { new_g0 = new_g0 + ga }
        if a1 == 1 { new_g1 = new_g1 + ga }
        if a1 == 2 { new_g2 = new_g2 + ga }
        if a1 == 3 { new_g3 = new_g3 + ga }
        if a1 == 4 { new_g4 = new_g4 + ga }
        if a1 == 5 { new_g5 = new_g5 + ga }
    }
    if op == OP_TANH() {
        // d(tanh(a))/da = 1 - tanh(a)^2 = 1 - v^2
        let ga = dout * (1.0 - v * v)
        if a1 == 0 { new_g0 = new_g0 + ga }
        if a1 == 1 { new_g1 = new_g1 + ga }
        if a1 == 2 { new_g2 = new_g2 + ga }
        if a1 == 3 { new_g3 = new_g3 + ga }
        if a1 == 4 { new_g4 = new_g4 + ga }
        if a1 == 5 { new_g5 = new_g5 + ga }
    }
    if op == OP_LEAKY_RELU() {
        // d(leaky_relu(a))/da = 1 if a > 0 else alpha
        // Note: v > 0 iff input > 0 (since alpha > 0)
        let alpha = LEAKY_ALPHA()
        let ga = if v > 0.0 { dout } else { dout * alpha }
        if a1 == 0 { new_g0 = new_g0 + ga }
        if a1 == 1 { new_g1 = new_g1 + ga }
        if a1 == 2 { new_g2 = new_g2 + ga }
        if a1 == 3 { new_g3 = new_g3 + ga }
        if a1 == 4 { new_g4 = new_g4 + ga }
        if a1 == 5 { new_g5 = new_g5 + ga }
    }

    // Create new tape with updated gradients
    return Tape {
        op0: t.op0, a10: t.a10, a20: t.a20, v0: t.v0, g0: new_g0,
        op1: t.op1, a11: t.a11, a21: t.a21, v1: t.v1, g1: new_g1,
        op2: t.op2, a12: t.a12, a22: t.a22, v2: t.v2, g2: new_g2,
        op3: t.op3, a13: t.a13, a23: t.a23, v3: t.v3, g3: new_g3,
        op4: t.op4, a14: t.a14, a24: t.a24, v4: t.v4, g4: new_g4,
        op5: t.op5, a15: t.a15, a25: t.a25, v5: t.v5, g5: new_g5,
        len: t.len
    }
}

fn backward(tape: Tape, out: i64) -> Tape {
    let mut t = set_g(tape, out, 1.0)

    // Unroll the loop to avoid while-loop struct assignment bug
    if t.len > 5 { t = backward_step(t, 5) }
    if t.len > 4 { t = backward_step(t, 4) }
    if t.len > 3 { t = backward_step(t, 3) }
    if t.len > 2 { t = backward_step(t, 2) }
    if t.len > 1 { t = backward_step(t, 1) }
    if t.len > 0 { t = backward_step(t, 0) }

    return t
}

// ============================================================================
// TESTS
// ============================================================================

fn main() -> i32 {
    println("=== Reverse-Mode AD Tests ===")
    println("")

    let mut ok = true
    let tol = 0.001

    // Test 1: d(x*y) at x=3, y=4 -> df/dx=4, df/dy=3
    println("Test 1: d(x*y) at x=3, y=4")
    let mut t1 = tape_new()
    t1 = tvar(t1, 3.0)    // 0
    t1 = tvar(t1, 4.0)    // 1
    t1 = tmul(t1, 0, 1)   // 2
    t1 = backward(t1, 2)
    let v1 = get_v(t1, 2)
    let g1x = get_g(t1, 0)
    let g1y = get_g(t1, 1)
    println("  f = ")
    println(v1)
    println("  df/dx = ")
    println(g1x)
    println("  df/dy = ")
    println(g1y)
    if abs_f64(v1 - 12.0) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g1x - 4.0) > tol { ok = false; println("  FAIL: gx") }
    if abs_f64(g1y - 3.0) > tol { ok = false; println("  FAIL: gy") }
    println("")

    // Test 2: d(x^2) at x=3 -> df/dx=6
    println("Test 2: d(x^2) at x=3")
    let mut t2 = tape_new()
    t2 = tvar(t2, 3.0)    // 0
    t2 = tmul(t2, 0, 0)   // 1
    t2 = backward(t2, 1)
    let v2 = get_v(t2, 1)
    let g2 = get_g(t2, 0)
    println("  f = ")
    println(v2)
    println("  df/dx = ")
    println(g2)
    if abs_f64(v2 - 9.0) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g2 - 6.0) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 3: d(exp(x^2)) at x=1 -> df/dx = 2*exp(1)
    println("Test 3: d(exp(x^2)) at x=1")
    let mut t3 = tape_new()
    t3 = tvar(t3, 1.0)    // 0
    t3 = tmul(t3, 0, 0)   // 1
    t3 = texp(t3, 1)      // 2
    t3 = backward(t3, 2)
    let v3 = get_v(t3, 2)
    let g3 = get_g(t3, 0)
    let ex3 = 2.0 * exp_f64(1.0)
    println("  f = ")
    println(v3)
    println("  df/dx = ")
    println(g3)
    println("  expected = ")
    println(ex3)
    if abs_f64(v3 - exp_f64(1.0)) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g3 - ex3) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 4: sigmoid at x=0 -> f=0.5, df=0.25
    println("Test 4: sigmoid at x=0")
    let mut t4 = tape_new()
    t4 = tvar(t4, 0.0)       // 0
    t4 = tsigmoid(t4, 0)     // 1
    t4 = backward(t4, 1)
    let v4 = get_v(t4, 1)
    let g4 = get_g(t4, 0)
    println("  f = ")
    println(v4)
    println("  df/dx = ")
    println(g4)
    if abs_f64(v4 - 0.5) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g4 - 0.25) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 5: f = x*y + y*z at (1,2,3) -> df/dx=2, df/dy=4, df/dz=2
    println("Test 5: f = x*y + y*z at (1,2,3)")
    let mut t5 = tape_new()
    t5 = tvar(t5, 1.0)    // 0: x
    t5 = tvar(t5, 2.0)    // 1: y
    t5 = tvar(t5, 3.0)    // 2: z
    t5 = tmul(t5, 0, 1)   // 3: x*y
    t5 = tmul(t5, 1, 2)   // 4: y*z
    t5 = tadd(t5, 3, 4)   // 5: x*y + y*z
    t5 = backward(t5, 5)
    let v5 = get_v(t5, 5)
    let gx = get_g(t5, 0)
    let gy = get_g(t5, 1)
    let gz = get_g(t5, 2)
    println("  f = ")
    println(v5)
    println("  df/dx = ")
    println(gx)
    println("  df/dy = ")
    println(gy)
    println("  df/dz = ")
    println(gz)
    if abs_f64(v5 - 8.0) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(gx - 2.0) > tol { ok = false; println("  FAIL: gx") }
    if abs_f64(gy - 4.0) > tol { ok = false; println("  FAIL: gy") }
    if abs_f64(gz - 2.0) > tol { ok = false; println("  FAIL: gz") }
    println("")

    // Test 6: ReLU at x=2 -> f=2, df=1
    println("Test 6: relu(x) at x=2")
    let mut t6a = tape_new()
    t6a = tvar(t6a, 2.0)      // 0
    t6a = trelu(t6a, 0)       // 1
    t6a = backward(t6a, 1)
    let v6a = get_v(t6a, 1)
    let g6a = get_g(t6a, 0)
    println("  f = ")
    println(v6a)
    println("  df/dx = ")
    println(g6a)
    if abs_f64(v6a - 2.0) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g6a - 1.0) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 7: ReLU at x=-3 -> f=0, df=0
    println("Test 7: relu(x) at x=-3")
    let mut t6b = tape_new()
    t6b = tvar(t6b, 0.0 - 3.0)  // 0
    t6b = trelu(t6b, 0)         // 1
    t6b = backward(t6b, 1)
    let v6b = get_v(t6b, 1)
    let g6b = get_g(t6b, 0)
    println("  f = ")
    println(v6b)
    println("  df/dx = ")
    println(g6b)
    if abs_f64(v6b - 0.0) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g6b - 0.0) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 8: Chain rule with ReLU: d(relu(x^2))/dx at x=2 -> f=4, df=4
    println("Test 8: relu(x^2) at x=2")
    let mut t7 = tape_new()
    t7 = tvar(t7, 2.0)        // 0
    t7 = tmul(t7, 0, 0)       // 1: x^2 = 4
    t7 = trelu(t7, 1)         // 2: relu(4) = 4
    t7 = backward(t7, 2)
    let v7 = get_v(t7, 2)
    let g7 = get_g(t7, 0)
    // Chain rule: d(relu(x^2))/dx = d(relu)/d(x^2) * d(x^2)/dx = 1 * 2x = 4
    println("  f = ")
    println(v7)
    println("  df/dx = ")
    println(g7)
    if abs_f64(v7 - 4.0) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g7 - 4.0) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 9: tanh at x=0 -> f=0, df=1
    println("Test 9: tanh(x) at x=0")
    let mut t8 = tape_new()
    t8 = tvar(t8, 0.0)        // 0
    t8 = ttanh(t8, 0)         // 1
    t8 = backward(t8, 1)
    let v8 = get_v(t8, 1)
    let g8 = get_g(t8, 0)
    println("  f = ")
    println(v8)
    println("  df/dx = ")
    println(g8)
    // tanh(0) = 0, d(tanh)/dx at 0 = 1 - 0^2 = 1
    if abs_f64(v8 - 0.0) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g8 - 1.0) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 10: tanh at x=1 -> f≈0.7616, df≈0.4200
    println("Test 10: tanh(x) at x=1")
    let mut t9 = tape_new()
    t9 = tvar(t9, 1.0)        // 0
    t9 = ttanh(t9, 0)         // 1
    t9 = backward(t9, 1)
    let v9 = get_v(t9, 1)
    let g9 = get_g(t9, 0)
    // tanh(1) = (e - 1/e) / (e + 1/e) ≈ 0.7616
    // d(tanh)/dx = 1 - tanh^2 ≈ 1 - 0.5800 ≈ 0.4200
    let expected_tanh1 = 0.7615941559557649
    let expected_grad1 = 1.0 - expected_tanh1 * expected_tanh1
    println("  f = ")
    println(v9)
    println("  expected = ")
    println(expected_tanh1)
    println("  df/dx = ")
    println(g9)
    println("  expected = ")
    println(expected_grad1)
    if abs_f64(v9 - expected_tanh1) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g9 - expected_grad1) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 11: Leaky ReLU at x=2 -> f=2, df=1
    println("Test 11: leaky_relu(x) at x=2")
    let mut t10 = tape_new()
    t10 = tvar(t10, 2.0)          // 0
    t10 = tleaky_relu(t10, 0)     // 1
    t10 = backward(t10, 1)
    let v10 = get_v(t10, 1)
    let g10 = get_g(t10, 0)
    println("  f = ")
    println(v10)
    println("  df/dx = ")
    println(g10)
    if abs_f64(v10 - 2.0) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g10 - 1.0) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 12: Leaky ReLU at x=-3 -> f=-0.03, df=0.01
    println("Test 12: leaky_relu(x) at x=-3")
    let mut t11 = tape_new()
    t11 = tvar(t11, 0.0 - 3.0)    // 0
    t11 = tleaky_relu(t11, 0)     // 1
    t11 = backward(t11, 1)
    let v11 = get_v(t11, 1)
    let g11 = get_g(t11, 0)
    let expected_v11 = 0.0 - 0.03  // -3 * 0.01 = -0.03
    let expected_g11 = 0.01        // alpha
    println("  f = ")
    println(v11)
    println("  expected = ")
    println(expected_v11)
    println("  df/dx = ")
    println(g11)
    println("  expected = ")
    println(expected_g11)
    if abs_f64(v11 - expected_v11) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g11 - expected_g11) > tol { ok = false; println("  FAIL: g") }
    println("")

    if ok {
        println("ALL TESTS PASSED")
        return 0
    } else {
        println("SOME TESTS FAILED")
        return 1
    }
}
