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

fn log_f64(x: f64) -> f64 {
    // Natural logarithm
    if x <= 0.0 { return 0.0 - 1000000.0 }  // Return large negative for invalid input
    if x == 1.0 { return 0.0 }

    // Use identity: log(x) = 2 * log(sqrt(x)) to bring x closer to 1
    // Trigger this for x > 1.5 or x < 0.7 for faster convergence
    if x > 1.5 { return 2.0 * log_f64(sqrt_f64(x)) }
    if x < 0.7 { return 0.0 - log_f64(1.0 / x) }

    // For x in [0.7, 1.5], use Taylor series around 1: log(1+u) = u - u^2/2 + u^3/3 - ...
    // This converges quickly for |u| < 0.5
    let u = x - 1.0
    let mut sum = 0.0
    let mut term = u
    let mut i = 1
    while i <= 30 {
        sum = sum + term / i
        term = 0.0 - term * u
        i = i + 1
    }
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
fn OP_SOFTMAX2() -> i64 { return 19 }
fn OP_LOG() -> i64 { return 20 }
fn OP_CROSS_ENTROPY() -> i64 { return 21 }

// Leaky ReLU slope for negative inputs (standard value)
fn LEAKY_ALPHA() -> f64 { return 0.01 }

// Small epsilon for numerical stability in log
fn LOG_EPSILON() -> f64 { return 0.0000001 }

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

// 2-class softmax: softmax_0(a, b) = exp(a) / (exp(a) + exp(b))
// Returns the probability for class 0 (first input)
// Note: softmax_1 = 1 - softmax_0 for 2-class case
fn tsoftmax2(t: Tape, a: i64, b: i64) -> Tape {
    let av = get_v(t, a)
    let bv = get_v(t, b)
    // For numerical stability, subtract max before exp
    let m = if av > bv { av } else { bv }
    let ea = exp_f64(av - m)
    let eb = exp_f64(bv - m)
    let sum = ea + eb
    let y0 = ea / sum
    return push(t, OP_SOFTMAX2(), a, b, y0)
}

// Natural logarithm: log(x)
fn tlog(t: Tape, a: i64) -> Tape {
    let av = get_v(t, a)
    // Add small epsilon for numerical stability
    let eps = LOG_EPSILON()
    let safe_v = if av > eps { av } else { eps }
    return push(t, OP_LOG(), a, 0 - 1, log_f64(safe_v))
}

// Clamp value to [eps, 1-eps] for numerical stability
fn clamp_prob(p: f64) -> f64 {
    let eps = LOG_EPSILON()
    let upper = 1.0 - eps
    if p < eps { return eps }
    if p > upper { return upper }
    return p
}

// Binary cross-entropy loss: -[target * log(pred) + (1-target) * log(1-pred)]
// Note: Due to Demetrios bug with large struct parameters, we pass values directly
fn cross_entropy_loss_debug(p: f64, y: f64, debug: bool) -> f64 {
    // Clamp prediction for numerical stability
    let p_safe = clamp_prob(p)
    let one_minus_p = clamp_prob(1.0 - p)

    // L = -[y * log(p) + (1-y) * log(1-p)]
    let log_p = log_f64(p_safe)
    let log_1mp = log_f64(one_minus_p)

    // Use weighted sum: loss = -y*log(p) - (1-y)*log(1-p)
    let neg_log_p = 0.0 - log_p
    let neg_log_1mp = 0.0 - log_1mp
    let term1 = y * neg_log_p
    let term2 = (1.0 - y) * neg_log_1mp

    if debug {
        println("    [CE] p_input = ")
        println(p)
        println("    [CE] y_input = ")
        println(y)
        println("    [CE] p_safe = ")
        println(p_safe)
        println("    [CE] 1-p = ")
        println(one_minus_p)
        println("    [CE] term1 = ")
        println(term1)
        println("    [CE] term2 = ")
        println(term2)
    }

    return term1 + term2
}

fn cross_entropy_loss(p: f64, y: f64) -> f64 {
    return cross_entropy_loss_debug(p, y, false)
}

// Build cross-entropy by reading values first, then calling with explicit values
// This is a workaround for Demetrios bug with large struct parameters
fn tcross_entropy_with_values_debug(t: Tape, pred_idx: i64, target_idx: i64, p: f64, y: f64, debug: bool) -> Tape {
    let loss = cross_entropy_loss_debug(p, y, debug)
    return push(t, OP_CROSS_ENTROPY(), pred_idx, target_idx, loss)
}

fn tcross_entropy_with_values(t: Tape, pred_idx: i64, target_idx: i64, p: f64, y: f64) -> Tape {
    return tcross_entropy_with_values_debug(t, pred_idx, target_idx, p, y, false)
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
    if op == OP_SOFTMAX2() {
        // 2-class softmax: y0 = softmax_0(x0, x1)
        // y1 = 1 - y0 (for 2-class)
        // ∂y0/∂x0 = y0 * (1 - y0) = y0 * y1
        // ∂y0/∂x1 = -y0 * y1
        let y0 = v
        let y1 = 1.0 - y0
        let ga = dout * y0 * y1         // gradient to first input (a1)
        let gb = 0.0 - dout * y0 * y1   // gradient to second input (a2)
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
    if op == OP_LOG() {
        // d(log(a))/da = 1/a
        let av = get_v(t, a1)
        let eps = LOG_EPSILON()
        let safe_a = if av > eps { av } else { eps }
        let ga = dout / safe_a
        if a1 == 0 { new_g0 = new_g0 + ga }
        if a1 == 1 { new_g1 = new_g1 + ga }
        if a1 == 2 { new_g2 = new_g2 + ga }
        if a1 == 3 { new_g3 = new_g3 + ga }
        if a1 == 4 { new_g4 = new_g4 + ga }
        if a1 == 5 { new_g5 = new_g5 + ga }
    }
    if op == OP_CROSS_ENTROPY() {
        // L = -[y * log(p) + (1-y) * log(1-p)]
        // dL/dp = -y/p + (1-y)/(1-p) = (p - y) / (p * (1-p))
        // dL/dy = -log(p) + log(1-p) = log((1-p)/p)
        let p = get_v(t, a1)  // predicted probability
        let y = get_v(t, a2)  // target label
        let eps = LOG_EPSILON()
        let p_safe = if p < eps { eps } else { if p > 1.0 - eps { 1.0 - eps } else { p } }

        // Gradient w.r.t. prediction: dL/dp = (p - y) / (p * (1 - p))
        let gp = dout * (p_safe - y) / (p_safe * (1.0 - p_safe))

        // Gradient w.r.t. target: dL/dy = log((1-p)/p)
        // Usually target is fixed (not learned), but include for completeness
        let gy = dout * (log_f64(1.0 - p_safe) - log_f64(p_safe))

        if a1 == 0 { new_g0 = new_g0 + gp }
        if a1 == 1 { new_g1 = new_g1 + gp }
        if a1 == 2 { new_g2 = new_g2 + gp }
        if a1 == 3 { new_g3 = new_g3 + gp }
        if a1 == 4 { new_g4 = new_g4 + gp }
        if a1 == 5 { new_g5 = new_g5 + gp }
        if a2 == 0 { new_g0 = new_g0 + gy }
        if a2 == 1 { new_g1 = new_g1 + gy }
        if a2 == 2 { new_g2 = new_g2 + gy }
        if a2 == 3 { new_g3 = new_g3 + gy }
        if a2 == 4 { new_g4 = new_g4 + gy }
        if a2 == 5 { new_g5 = new_g5 + gy }
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
// ADAM OPTIMIZER
// ============================================================================

// Adam hyperparameters
fn ADAM_BETA1() -> f64 { return 0.9 }
fn ADAM_BETA2() -> f64 { return 0.999 }
fn ADAM_EPSILON() -> f64 { return 0.00000001 }
fn ADAM_LR() -> f64 { return 0.001 }

// Adam state for 6 parameters (matches tape variable slots)
struct Adam {
    // First moment (momentum)
    m0: f64, m1: f64, m2: f64, m3: f64, m4: f64, m5: f64,
    // Second moment (squared gradient)
    v0: f64, v1: f64, v2: f64, v3: f64, v4: f64, v5: f64,
    // Timestep for bias correction
    t: f64
}

fn adam_new() -> Adam {
    return Adam {
        m0: 0.0, m1: 0.0, m2: 0.0, m3: 0.0, m4: 0.0, m5: 0.0,
        v0: 0.0, v1: 0.0, v2: 0.0, v3: 0.0, v4: 0.0, v5: 0.0,
        t: 0.0
    }
}

// Get first moment m for parameter i
fn adam_get_m(a: Adam, i: i64) -> f64 {
    if i == 0 { return a.m0 }
    if i == 1 { return a.m1 }
    if i == 2 { return a.m2 }
    if i == 3 { return a.m3 }
    if i == 4 { return a.m4 }
    return a.m5
}

// Get second moment v for parameter i
fn adam_get_v(a: Adam, i: i64) -> f64 {
    if i == 0 { return a.v0 }
    if i == 1 { return a.v1 }
    if i == 2 { return a.v2 }
    if i == 3 { return a.v3 }
    if i == 4 { return a.v4 }
    return a.v5
}

// Single parameter Adam update
// Returns (new_param, new_m, new_v)
fn adam_update_param(param: f64, g: f64, m: f64, v: f64, timestep: f64, lr: f64) -> f64 {
    let beta1 = ADAM_BETA1()
    let beta2 = ADAM_BETA2()
    let eps = ADAM_EPSILON()

    // Update biased first moment: m = β1*m + (1-β1)*g
    let new_m = beta1 * m + (1.0 - beta1) * g

    // Update biased second moment: v = β2*v + (1-β2)*g²
    let new_v = beta2 * v + (1.0 - beta2) * g * g

    // Bias correction
    let m_hat = new_m / (1.0 - pow_f64(beta1, timestep))
    let v_hat = new_v / (1.0 - pow_f64(beta2, timestep))

    // Parameter update: θ = θ - lr * m_hat / (√v_hat + ε)
    let new_param = param - lr * m_hat / (sqrt_f64(v_hat) + eps)

    return new_param
}

// Power function for bias correction
fn pow_f64(base: f64, exp: f64) -> f64 {
    // For small integer-like exponents, use multiplication
    // For Adam, exp is typically small (timesteps)
    if exp <= 0.0 { return 1.0 }
    if exp < 1.0 { return base }

    let mut result = 1.0
    let mut i = 0.0
    while i < exp {
        result = result * base
        i = i + 1.0
    }
    return result
}

// Adam step for single parameter - returns tuple-like struct
struct AdamResult {
    param: f64,
    m: f64,
    v: f64
}

fn adam_step_single(param: f64, g: f64, m: f64, v: f64, timestep: f64, lr: f64) -> AdamResult {
    let beta1 = ADAM_BETA1()
    let beta2 = ADAM_BETA2()
    let eps = ADAM_EPSILON()

    // Update biased first moment
    let new_m = beta1 * m + (1.0 - beta1) * g

    // Update biased second moment
    let new_v = beta2 * v + (1.0 - beta2) * g * g

    // Bias correction
    let m_hat = new_m / (1.0 - pow_f64(beta1, timestep))
    let v_hat = new_v / (1.0 - pow_f64(beta2, timestep))

    // Parameter update
    let new_param = param - lr * m_hat / (sqrt_f64(v_hat) + eps)

    return AdamResult { param: new_param, m: new_m, v: new_v }
}

// ============================================================================
// SGD WITH MOMENTUM
// ============================================================================

// SGD with momentum hyperparameters
fn SGD_MOMENTUM() -> f64 { return 0.9 }

// Result struct for SGD with momentum
struct SGDMomentumResult {
    param: f64,
    velocity: f64
}

// SGD with momentum update for single parameter
// Formula: v = momentum * v + gradient
//          param = param - lr * v
fn sgd_momentum_step(param: f64, g: f64, velocity: f64, lr: f64, momentum: f64) -> SGDMomentumResult {
    // Update velocity: v = momentum * v + g
    let new_velocity = momentum * velocity + g

    // Update parameter: θ = θ - lr * v
    let new_param = param - lr * new_velocity

    return SGDMomentumResult { param: new_param, velocity: new_velocity }
}

// Nesterov Accelerated Gradient (NAG) - a variant of momentum
// Formula: v = momentum * v + gradient(param - momentum * v)
//          param = param - lr * v
// Note: This simplified version computes gradient at current position
fn sgd_nesterov_step(param: f64, g: f64, velocity: f64, lr: f64, momentum: f64) -> SGDMomentumResult {
    // Nesterov update: v = momentum * v + g
    let new_velocity = momentum * velocity + g

    // Update with momentum correction: θ = θ - lr * (momentum * v + g)
    let new_param = param - lr * (momentum * new_velocity + g)

    return SGDMomentumResult { param: new_param, velocity: new_velocity }
}

// ============================================================================
// RMSPROP OPTIMIZER
// ============================================================================

// RMSprop hyperparameters (Hinton, 2012)
fn RMSPROP_DECAY() -> f64 { return 0.9 }
fn RMSPROP_EPS() -> f64 { return 0.00000001 }

// Result struct for RMSprop
struct RMSpropResult {
    param: f64,
    cache: f64
}

// RMSprop update for single parameter
// Formula: cache = decay * cache + (1 - decay) * gradient^2
//          param = param - lr * gradient / (sqrt(cache) + epsilon)
// RMSprop adapts learning rate per-parameter using moving average of squared gradients
fn rmsprop_step(param: f64, g: f64, cache: f64, lr: f64, decay: f64) -> RMSpropResult {
    let eps = RMSPROP_EPS()

    // Update cache: moving average of squared gradients
    let new_cache = decay * cache + (1.0 - decay) * g * g

    // Parameter update with adaptive learning rate
    let new_param = param - lr * g / (sqrt_f64(new_cache) + eps)

    return RMSpropResult { param: new_param, cache: new_cache }
}

// RMSprop with momentum (combines RMSprop adaptive lr with momentum)
struct RMSpropMomentumResult {
    param: f64,
    cache: f64,
    velocity: f64
}

fn rmsprop_momentum_step(param: f64, g: f64, cache: f64, velocity: f64, lr: f64, decay: f64, momentum: f64) -> RMSpropMomentumResult {
    let eps = RMSPROP_EPS()

    // Update cache
    let new_cache = decay * cache + (1.0 - decay) * g * g

    // Compute adaptive gradient
    let adaptive_g = g / (sqrt_f64(new_cache) + eps)

    // Apply momentum to adaptive gradient
    let new_velocity = momentum * velocity + adaptive_g
    let new_param = param - lr * new_velocity

    return RMSpropMomentumResult { param: new_param, cache: new_cache, velocity: new_velocity }
}

// ============================================================================
// ADAGRAD OPTIMIZER
// ============================================================================

// AdaGrad hyperparameters (Duchi et al., 2011)
fn ADAGRAD_EPS() -> f64 { return 0.00000001 }

// Result struct for AdaGrad
struct AdaGradResult {
    param: f64,
    sum_sq: f64
}

// AdaGrad update for single parameter
// Formula: sum_sq = sum_sq + gradient^2  (accumulates ALL past squared gradients)
//          param = param - lr * gradient / (sqrt(sum_sq) + epsilon)
// AdaGrad adapts learning rate per-parameter, but lr monotonically decreases
// Good for sparse gradients, but can stop learning too early in deep nets
fn adagrad_step(param: f64, g: f64, sum_sq: f64, lr: f64) -> AdaGradResult {
    let eps = ADAGRAD_EPS()

    // Accumulate squared gradient (no decay - key difference from RMSprop)
    let new_sum_sq = sum_sq + g * g

    // Parameter update with adaptive learning rate
    let new_param = param - lr * g / (sqrt_f64(new_sum_sq) + eps)

    return AdaGradResult { param: new_param, sum_sq: new_sum_sq }
}

// ============================================================================
// ADADELTA OPTIMIZER
// ============================================================================

// AdaDelta hyperparameters (Zeiler, 2012)
fn ADADELTA_RHO() -> f64 { return 0.95 }
fn ADADELTA_EPS() -> f64 { return 0.000001 }  // Typically larger eps than other optimizers

// Result struct for AdaDelta
struct AdaDeltaResult {
    param: f64,
    acc_grad: f64,   // E[g²] - accumulated squared gradients
    acc_delta: f64   // E[Δx²] - accumulated squared updates
}

// AdaDelta update for single parameter
// Key innovation: NO learning rate hyperparameter needed!
// Formula: E[g²]_t = ρ * E[g²]_{t-1} + (1-ρ) * g²
//          Δx = -RMS[Δx]_{t-1} / RMS[g]_t * g
//          E[Δx²]_t = ρ * E[Δx²]_{t-1} + (1-ρ) * Δx²
//          x_t = x_{t-1} + Δx
// where RMS[x] = sqrt(E[x²] + ε)
fn adadelta_step(param: f64, g: f64, acc_grad: f64, acc_delta: f64, rho: f64) -> AdaDeltaResult {
    let eps = ADADELTA_EPS()

    // Accumulate squared gradient with decay
    let new_acc_grad = rho * acc_grad + (1.0 - rho) * g * g

    // Compute RMS of gradients and previous updates
    let rms_grad = sqrt_f64(new_acc_grad + eps)
    let rms_delta = sqrt_f64(acc_delta + eps)

    // Compute update (note: no learning rate!)
    let delta_x = 0.0 - rms_delta / rms_grad * g

    // Accumulate squared updates
    let new_acc_delta = rho * acc_delta + (1.0 - rho) * delta_x * delta_x

    // Apply update
    let new_param = param + delta_x

    return AdaDeltaResult { param: new_param, acc_grad: new_acc_grad, acc_delta: new_acc_delta }
}

// ============================================================================
// ADAMW OPTIMIZER (DECOUPLED WEIGHT DECAY)
// ============================================================================

// AdamW hyperparameters (Loshchilov & Hutter, 2017)
fn ADAMW_BETA1() -> f64 { return 0.9 }
fn ADAMW_BETA2() -> f64 { return 0.999 }
fn ADAMW_EPS() -> f64 { return 0.00000001 }
fn ADAMW_WEIGHT_DECAY() -> f64 { return 0.01 }  // Common default

// Result struct for AdamW (same as Adam)
struct AdamWResult {
    param: f64,
    m: f64,
    v: f64
}

// AdamW update for single parameter
// Key difference from Adam: weight decay is DECOUPLED from gradient update
// Adam + L2: g = g + λ*w, then apply Adam (couples decay with adaptive lr)
// AdamW: apply Adam update, then subtract λ*w separately (proper regularization)
//
// Formula: m = β1*m + (1-β1)*g
//          v = β2*v + (1-β2)*g²
//          m_hat = m / (1-β1^t)
//          v_hat = v / (1-β2^t)
//          param = param - lr * (m_hat/(√v_hat+ε) + λ*param)
//
// This is equivalent to:
//          param = (1 - lr*λ)*param - lr*m_hat/(√v_hat+ε)
fn adamw_step(param: f64, g: f64, m: f64, v: f64, timestep: f64, lr: f64, weight_decay: f64) -> AdamWResult {
    let beta1 = ADAMW_BETA1()
    let beta2 = ADAMW_BETA2()
    let eps = ADAMW_EPS()

    // Update biased first moment estimate
    let new_m = beta1 * m + (1.0 - beta1) * g

    // Update biased second raw moment estimate
    let new_v = beta2 * v + (1.0 - beta2) * g * g

    // Compute bias-corrected estimates
    let m_hat = new_m / (1.0 - pow_f64(beta1, timestep))
    let v_hat = new_v / (1.0 - pow_f64(beta2, timestep))

    // AdamW update: decoupled weight decay
    // First apply Adam update, then apply weight decay separately
    let adam_update = lr * m_hat / (sqrt_f64(v_hat) + eps)
    let decay_update = lr * weight_decay * param
    let new_param = param - adam_update - decay_update

    return AdamWResult { param: new_param, m: new_m, v: new_v }
}

// AdamW with running powers (more efficient for training loops)
// Instead of computing pow_f64(beta, t) each step, caller tracks beta1^t and beta2^t
fn adamw_step_fast(param: f64, g: f64, m: f64, v: f64, beta1_t: f64, beta2_t: f64, lr: f64, weight_decay: f64) -> AdamWResult {
    let beta1 = ADAMW_BETA1()
    let beta2 = ADAMW_BETA2()
    let eps = ADAMW_EPS()

    // Update moments
    let new_m = beta1 * m + (1.0 - beta1) * g
    let new_v = beta2 * v + (1.0 - beta2) * g * g

    // Bias correction using pre-computed powers
    let m_hat = new_m / (1.0 - beta1_t)
    let v_hat = new_v / (1.0 - beta2_t)

    // Decoupled weight decay update
    let new_param = param - lr * m_hat / (sqrt_f64(v_hat) + eps) - lr * weight_decay * param

    return AdamWResult { param: new_param, m: new_m, v: new_v }
}

// ============================================================================
// NADAM OPTIMIZER (NESTEROV-ACCELERATED ADAM)
// ============================================================================

// NAdam hyperparameters (Dozat, 2016)
fn NADAM_BETA1() -> f64 { return 0.9 }
fn NADAM_BETA2() -> f64 { return 0.999 }
fn NADAM_EPS() -> f64 { return 0.00000001 }

// Result struct for NAdam (same as Adam)
struct NAdamResult {
    param: f64,
    m: f64,
    v: f64
}

// NAdam update for single parameter
// Combines Adam with Nesterov momentum for faster convergence
//
// Key insight: Instead of using m_hat directly, NAdam uses a "look-ahead":
//   nesterov_m = β1 * m_hat + (1 - β1) * g / (1 - β1^t)
//
// This applies Nesterov momentum to the bias-corrected first moment,
// giving the optimizer a "peek" at where the gradient is heading.
//
// Formula: m = β1*m + (1-β1)*g
//          v = β2*v + (1-β2)*g²
//          m_hat = m / (1-β1^t)
//          g_hat = g / (1-β1^t)
//          nesterov_m = β1*m_hat + (1-β1)*g_hat
//          v_hat = v / (1-β2^t)
//          param = param - lr * nesterov_m / (√v_hat + ε)
fn nadam_step(param: f64, g: f64, m: f64, v: f64, timestep: f64, lr: f64) -> NAdamResult {
    let beta1 = NADAM_BETA1()
    let beta2 = NADAM_BETA2()
    let eps = NADAM_EPS()

    // Update biased first moment estimate
    let new_m = beta1 * m + (1.0 - beta1) * g

    // Update biased second raw moment estimate
    let new_v = beta2 * v + (1.0 - beta2) * g * g

    // Compute bias correction terms
    let beta1_t = pow_f64(beta1, timestep)
    let beta2_t = pow_f64(beta2, timestep)

    // Bias-corrected estimates
    let m_hat = new_m / (1.0 - beta1_t)
    let g_hat = g / (1.0 - beta1_t)
    let v_hat = new_v / (1.0 - beta2_t)

    // Nesterov momentum: look-ahead on the first moment
    let nesterov_m = beta1 * m_hat + (1.0 - beta1) * g_hat

    // Parameter update
    let new_param = param - lr * nesterov_m / (sqrt_f64(v_hat) + eps)

    return NAdamResult { param: new_param, m: new_m, v: new_v }
}

// NAdam with running powers (more efficient for training loops)
fn nadam_step_fast(param: f64, g: f64, m: f64, v: f64, beta1_t: f64, beta2_t: f64, lr: f64) -> NAdamResult {
    let beta1 = NADAM_BETA1()
    let beta2 = NADAM_BETA2()
    let eps = NADAM_EPS()

    // Update moments
    let new_m = beta1 * m + (1.0 - beta1) * g
    let new_v = beta2 * v + (1.0 - beta2) * g * g

    // Bias-corrected estimates
    let m_hat = new_m / (1.0 - beta1_t)
    let g_hat = g / (1.0 - beta1_t)
    let v_hat = new_v / (1.0 - beta2_t)

    // Nesterov momentum
    let nesterov_m = beta1 * m_hat + (1.0 - beta1) * g_hat

    // Parameter update
    let new_param = param - lr * nesterov_m / (sqrt_f64(v_hat) + eps)

    return NAdamResult { param: new_param, m: new_m, v: new_v }
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

    // Test 13: softmax2(0, 0) -> f=0.5 (equal inputs = equal probabilities)
    println("Test 13: softmax2(0, 0)")
    let mut t12 = tape_new()
    t12 = tvar(t12, 0.0)          // 0: x0
    t12 = tvar(t12, 0.0)          // 1: x1
    t12 = tsoftmax2(t12, 0, 1)    // 2: softmax_0
    t12 = backward(t12, 2)
    let v12 = get_v(t12, 2)
    let g12_x0 = get_g(t12, 0)
    let g12_x1 = get_g(t12, 1)
    // softmax(0, 0) = 0.5
    // d(softmax_0)/dx0 = y0 * (1 - y0) = 0.5 * 0.5 = 0.25
    // d(softmax_0)/dx1 = -y0 * y1 = -0.5 * 0.5 = -0.25
    println("  f = ")
    println(v12)
    println("  df/dx0 = ")
    println(g12_x0)
    println("  df/dx1 = ")
    println(g12_x1)
    if abs_f64(v12 - 0.5) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g12_x0 - 0.25) > tol { ok = false; println("  FAIL: g_x0") }
    if abs_f64(g12_x1 - (0.0 - 0.25)) > tol { ok = false; println("  FAIL: g_x1") }
    println("")

    // Test 14: softmax2(2, 0) -> higher prob for first class
    println("Test 14: softmax2(2, 0)")
    let mut t13 = tape_new()
    t13 = tvar(t13, 2.0)          // 0: x0
    t13 = tvar(t13, 0.0)          // 1: x1
    t13 = tsoftmax2(t13, 0, 1)    // 2: softmax_0
    t13 = backward(t13, 2)
    let v13 = get_v(t13, 2)
    let g13_x0 = get_g(t13, 0)
    let g13_x1 = get_g(t13, 1)
    // softmax_0(2, 0) = exp(2) / (exp(2) + exp(0)) = e^2 / (e^2 + 1)
    let e2 = exp_f64(2.0)
    let expected_v13 = e2 / (e2 + 1.0)
    let y0_13 = expected_v13
    let y1_13 = 1.0 - y0_13
    let expected_g13_x0 = y0_13 * y1_13
    let expected_g13_x1 = 0.0 - y0_13 * y1_13
    println("  f = ")
    println(v13)
    println("  expected = ")
    println(expected_v13)
    println("  df/dx0 = ")
    println(g13_x0)
    println("  expected = ")
    println(expected_g13_x0)
    println("  df/dx1 = ")
    println(g13_x1)
    println("  expected = ")
    println(expected_g13_x1)
    if abs_f64(v13 - expected_v13) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g13_x0 - expected_g13_x0) > tol { ok = false; println("  FAIL: g_x0") }
    if abs_f64(g13_x1 - expected_g13_x1) > tol { ok = false; println("  FAIL: g_x1") }
    println("")

    // Test 15: log(e) = 1
    println("Test 15: log(e)")
    let mut t14 = tape_new()
    let e_val = exp_f64(1.0)  // e ≈ 2.718
    t14 = tvar(t14, e_val)    // 0: e
    t14 = tlog(t14, 0)        // 1: log(e) = 1
    t14 = backward(t14, 1)
    let v14 = get_v(t14, 1)
    let g14 = get_g(t14, 0)
    // log(e) = 1, d(log(x))/dx = 1/x = 1/e
    let expected_g14 = 1.0 / e_val
    println("  f = ")
    println(v14)
    println("  expected = 1.0")
    println("  df/dx = ")
    println(g14)
    println("  expected = ")
    println(expected_g14)
    if abs_f64(v14 - 1.0) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g14 - expected_g14) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 16: log(1) = 0
    println("Test 16: log(1)")
    let mut t15 = tape_new()
    t15 = tvar(t15, 1.0)      // 0: 1
    t15 = tlog(t15, 0)        // 1: log(1) = 0
    t15 = backward(t15, 1)
    let v15 = get_v(t15, 1)
    let g15 = get_g(t15, 0)
    // log(1) = 0, d(log(x))/dx = 1/x = 1
    println("  f = ")
    println(v15)
    println("  expected = 0.0")
    println("  df/dx = ")
    println(g15)
    println("  expected = 1.0")
    if abs_f64(v15 - 0.0) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g15 - 1.0) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 17: Verify log_f64(0.5) = -log(2)
    println("Test 17: log(0.5) and log(2)")
    let log_half = log_f64(0.5)
    let log_two = log_f64(2.0)
    println("  log(0.5) = ")
    println(log_half)
    println("  log(2.0) = ")
    println(log_two)
    println("  log(0.5) + log(2) should = 0: ")
    println(log_half + log_two)
    // log(0.5) should be about -0.693
    if abs_f64(log_half - (0.0 - 0.693)) > 0.01 { ok = false; println("  FAIL: log(0.5)") }
    if abs_f64(log_two - 0.693) > 0.01 { ok = false; println("  FAIL: log(2)") }
    println("")

    // Test 18: cross_entropy(pred=0.5, target=1) = -log(0.5) = log(2)
    println("Test 18: cross_entropy(0.5, 1)")
    let mut t16 = tape_new()
    t16 = tvar(t16, 0.5)      // 0: pred
    t16 = tvar(t16, 1.0)      // 1: target
    // Read values before passing tape to function (workaround for struct bug)
    let p16 = get_v(t16, 0)
    let y16 = get_v(t16, 1)
    t16 = tcross_entropy_with_values(t16, 0, 1, p16, y16)  // 2: loss
    t16 = backward(t16, 2)
    let v16 = get_v(t16, 2)
    let g16_pred = get_g(t16, 0)
    // L = -[1*log(0.5) + 0*log(0.5)] = -log(0.5) = log(2) ≈ 0.693
    // Use same log function for expected value
    let expected_v16 = 0.0 - log_f64(0.5)
    // dL/dp = (p - y) / (p * (1-p)) = (0.5 - 1) / (0.5 * 0.5) = -0.5 / 0.25 = -2
    let expected_g16 = 0.0 - 2.0
    println("  L = ")
    println(v16)
    println("  expected = ")
    println(expected_v16)
    println("  dL/dp = ")
    println(g16_pred)
    println("  expected = ")
    println(expected_g16)
    if abs_f64(v16 - expected_v16) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g16_pred - expected_g16) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 19: cross_entropy(pred=0.8, target=1) - higher prob = lower loss
    println("Test 19: cross_entropy(0.8, 1)")
    let mut t17 = tape_new()
    t17 = tvar(t17, 0.8)      // 0: pred
    t17 = tvar(t17, 1.0)      // 1: target
    // Compute loss directly in test to avoid function parameter corruption bug
    let p17 = 0.8
    let y17 = 1.0
    let log_p17 = log_f64(p17)
    let log_1mp17 = log_f64(1.0 - p17)
    let loss17 = y17 * (0.0 - log_p17) + (1.0 - y17) * (0.0 - log_1mp17)
    t17 = push(t17, OP_CROSS_ENTROPY(), 0, 1, loss17)  // 2: loss
    t17 = backward(t17, 2)
    let v17 = get_v(t17, 2)
    let g17_pred = get_g(t17, 0)
    // L = -log(0.8) ≈ 0.223
    let expected_v17 = 0.0 - log_f64(0.8)
    // dL/dp = (0.8 - 1) / (0.8 * 0.2) = -0.2 / 0.16 = -1.25
    let expected_g17 = (0.8 - 1.0) / (0.8 * 0.2)
    println("  L = ")
    println(v17)
    println("  expected = ")
    println(expected_v17)
    println("  dL/dp = ")
    println(g17_pred)
    println("  expected = ")
    println(expected_g17)
    if abs_f64(v17 - expected_v17) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g17_pred - expected_g17) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 20: cross_entropy(pred=0.2, target=0) - correct low prediction
    println("Test 20: cross_entropy(0.2, 0)")
    let mut t18 = tape_new()
    t18 = tvar(t18, 0.2)      // 0: pred
    t18 = tvar(t18, 0.0)      // 1: target
    // Compute loss directly to avoid function parameter corruption bug
    let p18 = 0.2
    let y18 = 0.0
    let log_p18 = log_f64(p18)
    let log_1mp18 = log_f64(1.0 - p18)
    let loss18 = y18 * (0.0 - log_p18) + (1.0 - y18) * (0.0 - log_1mp18)
    t18 = push(t18, OP_CROSS_ENTROPY(), 0, 1, loss18)  // 2: loss
    t18 = backward(t18, 2)
    let v18 = get_v(t18, 2)
    let g18_pred = get_g(t18, 0)
    // L = -[0*log(0.2) + 1*log(0.8)] = -log(0.8) ≈ 0.223
    let expected_v18 = 0.0 - log_f64(0.8)
    // dL/dp = (0.2 - 0) / (0.2 * 0.8) = 0.2 / 0.16 = 1.25
    let expected_g18 = 0.2 / (0.2 * 0.8)
    println("  L = ")
    println(v18)
    println("  expected = ")
    println(expected_v18)
    println("  dL/dp = ")
    println(g18_pred)
    println("  expected = ")
    println(expected_g18)
    if abs_f64(v18 - expected_v18) > tol { ok = false; println("  FAIL: v") }
    if abs_f64(g18_pred - expected_g18) > tol { ok = false; println("  FAIL: g") }
    println("")

    // Test 21: Adam optimizer - single step verification
    // Due to Demetrios struct-in-loop bug, we test Adam formula correctness
    // with a single step instead of iterative optimization
    println("Test 21: Adam single step correctness")
    let x21 = 5.0
    let m21 = 0.0
    let v21 = 0.0
    let lr21 = 0.1
    let dx21 = 2.0 * x21  // gradient = 10.0

    // Adam step 1: compute manually
    let beta1 = ADAM_BETA1()  // 0.9
    let beta2 = ADAM_BETA2()  // 0.999
    let eps21 = ADAM_EPSILON()

    // m = 0.9 * 0 + 0.1 * 10 = 1.0
    let new_m21 = beta1 * m21 + (1.0 - beta1) * dx21
    // v = 0.999 * 0 + 0.001 * 100 = 0.1
    let new_v21 = beta2 * v21 + (1.0 - beta2) * dx21 * dx21
    // m_hat = 1.0 / (1 - 0.9^1) = 1.0 / 0.1 = 10.0
    let m_hat21 = new_m21 / (1.0 - pow_f64(beta1, 1.0))
    // v_hat = 0.1 / (1 - 0.999^1) = 0.1 / 0.001 = 100.0
    let v_hat21 = new_v21 / (1.0 - pow_f64(beta2, 1.0))
    // x_new = 5 - 0.1 * 10 / (sqrt(100) + eps) = 5 - 1/10 = 4.9
    let x21_new = x21 - lr21 * m_hat21 / (sqrt_f64(v_hat21) + eps21)

    // Verify with adam_step_single
    let result21 = adam_step_single(x21, dx21, m21, v21, 1.0, lr21)

    println("  Manual calculation:")
    println("    new_m = ")
    println(new_m21)
    println("    new_v = ")
    println(new_v21)
    println("    x_new = ")
    println(x21_new)
    println("  adam_step_single result:")
    println("    result.m = ")
    println(result21.m)
    println("    result.v = ")
    println(result21.v)
    println("    result.param = ")
    println(result21.param)

    // Expected: new_m = 1.0, new_v = 0.1, x_new ≈ 4.9
    if abs_f64(new_m21 - 1.0) > tol { ok = false; println("  FAIL: new_m") }
    if abs_f64(new_v21 - 0.1) > tol { ok = false; println("  FAIL: new_v") }
    if abs_f64(x21_new - 4.9) > tol { ok = false; println("  FAIL: x_new") }
    if abs_f64(result21.m - new_m21) > tol { ok = false; println("  FAIL: result.m mismatch") }
    if abs_f64(result21.v - new_v21) > tol { ok = false; println("  FAIL: result.v mismatch") }
    if abs_f64(result21.param - x21_new) > tol { ok = false; println("  FAIL: result.param mismatch") }
    println("")

    // Test 22: Adam multi-step (unrolled) to verify convergence
    println("Test 22: Adam 5-step descent (unrolled)")
    // Start from x=5, minimize x^2
    // Due to struct-in-loop bug, we unroll 5 steps manually
    let x0 = 5.0
    let m0_22 = 0.0
    let v0_22 = 0.0
    let lr22 = 0.5  // Higher LR for faster convergence in 5 steps

    // Step 1
    let g1 = 2.0 * x0
    let r1 = adam_step_single(x0, g1, m0_22, v0_22, 1.0, lr22)
    let x1 = r1.param
    let m1_22 = r1.m
    let v1_22 = r1.v

    // Step 2
    let g2_22 = 2.0 * x1
    let r2 = adam_step_single(x1, g2_22, m1_22, v1_22, 2.0, lr22)
    let x2 = r2.param
    let m2_22 = r2.m
    let v2_22 = r2.v

    // Step 3
    let g3 = 2.0 * x2
    let r3 = adam_step_single(x2, g3, m2_22, v2_22, 3.0, lr22)
    let x3 = r3.param
    let m3_22 = r3.m
    let v3_22 = r3.v

    // Step 4
    let g4 = 2.0 * x3
    let r4 = adam_step_single(x3, g4, m3_22, v3_22, 4.0, lr22)
    let x4 = r4.param
    let m4_22 = r4.m
    let v4_22 = r4.v

    // Step 5
    let g5 = 2.0 * x4
    let r5 = adam_step_single(x4, g5, m4_22, v4_22, 5.0, lr22)
    let x5 = r5.param

    println("  Descent from x=5:")
    println("    x0 = 5.0")
    println("    x1 = ")
    println(x1)
    println("    x2 = ")
    println(x2)
    println("    x3 = ")
    println(x3)
    println("    x4 = ")
    println(x4)
    println("    x5 = ")
    println(x5)

    // x should decrease monotonically toward 0
    if x1 >= x0 { ok = false; println("  FAIL: x1 >= x0") }
    if x2 >= x1 { ok = false; println("  FAIL: x2 >= x1") }
    if x3 >= x2 { ok = false; println("  FAIL: x3 >= x2") }
    if x5 >= 3.0 { ok = false; println("  FAIL: x5 should be < 3 after 5 steps") }
    println("")

    // Test 23: SGD with momentum - single step verification
    println("Test 23: SGD with momentum single step")
    let x23 = 5.0
    let vel23 = 0.0
    let lr23 = 0.1
    let mom23 = 0.9
    let g23 = 2.0 * x23  // gradient = 10.0

    // Manual calculation:
    // new_velocity = 0.9 * 0 + 10 = 10
    // new_param = 5 - 0.1 * 10 = 4
    let expected_vel23 = mom23 * vel23 + g23
    let expected_x23 = x23 - lr23 * expected_vel23

    let result23 = sgd_momentum_step(x23, g23, vel23, lr23, mom23)

    println("  Manual calculation:")
    println("    new_velocity = ")
    println(expected_vel23)
    println("    new_param = ")
    println(expected_x23)
    println("  sgd_momentum_step result:")
    println("    result.velocity = ")
    println(result23.velocity)
    println("    result.param = ")
    println(result23.param)

    if abs_f64(expected_vel23 - 10.0) > tol { ok = false; println("  FAIL: expected_vel") }
    if abs_f64(expected_x23 - 4.0) > tol { ok = false; println("  FAIL: expected_x") }
    if abs_f64(result23.velocity - expected_vel23) > tol { ok = false; println("  FAIL: result.velocity") }
    if abs_f64(result23.param - expected_x23) > tol { ok = false; println("  FAIL: result.param") }
    println("")

    // Test 24: SGD with momentum 5-step descent (unrolled)
    println("Test 24: SGD momentum 5-step descent (unrolled)")
    let y0_24 = 5.0
    let v0_24 = 0.0
    let lr24 = 0.1
    let mom24 = 0.9

    // Step 1
    let gy1 = 2.0 * y0_24
    let s1 = sgd_momentum_step(y0_24, gy1, v0_24, lr24, mom24)
    let y1_24 = s1.param
    let v1_24 = s1.velocity

    // Step 2
    let gy2 = 2.0 * y1_24
    let s2 = sgd_momentum_step(y1_24, gy2, v1_24, lr24, mom24)
    let y2_24 = s2.param
    let v2_24 = s2.velocity

    // Step 3
    let gy3 = 2.0 * y2_24
    let s3 = sgd_momentum_step(y2_24, gy3, v2_24, lr24, mom24)
    let y3_24 = s3.param
    let v3_24 = s3.velocity

    // Step 4
    let gy4 = 2.0 * y3_24
    let s4 = sgd_momentum_step(y3_24, gy4, v3_24, lr24, mom24)
    let y4_24 = s4.param
    let v4_24 = s4.velocity

    // Step 5
    let gy5 = 2.0 * y4_24
    let s5 = sgd_momentum_step(y4_24, gy5, v4_24, lr24, mom24)
    let y5_24 = s5.param

    println("  Descent from y=5 with momentum=0.9:")
    println("    y0 = 5.0")
    println("    y1 = ")
    println(y1_24)
    println("    y2 = ")
    println(y2_24)
    println("    y3 = ")
    println(y3_24)
    println("    y4 = ")
    println(y4_24)
    println("    y5 = ")
    println(y5_24)

    // y should decrease toward 0
    if y1_24 >= y0_24 { ok = false; println("  FAIL: y1 >= y0") }
    if y2_24 >= y1_24 { ok = false; println("  FAIL: y2 >= y1") }
    if y5_24 >= 2.0 { ok = false; println("  FAIL: y5 should be < 2 after 5 steps") }
    println("")

    // Test 25: RMSprop single step verification
    println("Test 25: RMSprop single step")
    let x25 = 5.0
    let cache25 = 0.0
    let lr25 = 0.1
    let decay25 = 0.9
    let g25 = 2.0 * x25  // gradient = 10.0

    // Manual calculation:
    // new_cache = 0.9 * 0 + 0.1 * 10^2 = 10
    // new_param = 5 - 0.1 * 10 / (sqrt(10) + 1e-8) = 5 - 1/sqrt(10) ≈ 4.684
    let expected_cache25 = decay25 * cache25 + (1.0 - decay25) * g25 * g25
    let expected_x25 = x25 - lr25 * g25 / (sqrt_f64(expected_cache25) + 0.00000001)

    let result25 = rmsprop_step(x25, g25, cache25, lr25, decay25)

    println("  Manual calculation:")
    println("    new_cache = ")
    println(expected_cache25)
    println("    new_param = ")
    println(expected_x25)
    println("  rmsprop_step result:")
    println("    result.cache = ")
    println(result25.cache)
    println("    result.param = ")
    println(result25.param)

    if abs_f64(expected_cache25 - 10.0) > tol { ok = false; println("  FAIL: expected_cache") }
    if abs_f64(result25.cache - expected_cache25) > tol { ok = false; println("  FAIL: result.cache") }
    if abs_f64(result25.param - expected_x25) > tol { ok = false; println("  FAIL: result.param") }
    println("")

    // Test 26: RMSprop 5-step descent (unrolled)
    println("Test 26: RMSprop 5-step descent (unrolled)")
    let z0_26 = 5.0
    let c0_26 = 0.0
    let lr26 = 0.1
    let decay26 = 0.9

    // Step 1
    let gz1 = 2.0 * z0_26
    let r1 = rmsprop_step(z0_26, gz1, c0_26, lr26, decay26)
    let z1_26 = r1.param
    let c1_26 = r1.cache

    // Step 2
    let gz2 = 2.0 * z1_26
    let r2 = rmsprop_step(z1_26, gz2, c1_26, lr26, decay26)
    let z2_26 = r2.param
    let c2_26 = r2.cache

    // Step 3
    let gz3 = 2.0 * z2_26
    let r3 = rmsprop_step(z2_26, gz3, c2_26, lr26, decay26)
    let z3_26 = r3.param
    let c3_26 = r3.cache

    // Step 4
    let gz4 = 2.0 * z3_26
    let r4 = rmsprop_step(z3_26, gz4, c3_26, lr26, decay26)
    let z4_26 = r4.param
    let c4_26 = r4.cache

    // Step 5
    let gz5 = 2.0 * z4_26
    let r5 = rmsprop_step(z4_26, gz5, c4_26, lr26, decay26)
    let z5_26 = r5.param

    println("  Descent from z=5 with decay=0.9:")
    println("    z0 = 5.0")
    println("    z1 = ")
    println(z1_26)
    println("    z2 = ")
    println(z2_26)
    println("    z3 = ")
    println(z3_26)
    println("    z4 = ")
    println(z4_26)
    println("    z5 = ")
    println(z5_26)

    // z should decrease toward 0 (RMSprop converges slower than momentum due to adaptive lr)
    if z1_26 >= z0_26 { ok = false; println("  FAIL: z1 >= z0") }
    if z2_26 >= z1_26 { ok = false; println("  FAIL: z2 >= z1") }
    if z5_26 >= 4.5 { ok = false; println("  FAIL: z5 should be < 4.5 after 5 steps") }
    println("")

    // Test 27: AdaGrad single step verification
    println("Test 27: AdaGrad single step")
    let x27 = 5.0
    let sum_sq27 = 0.0
    let lr27 = 0.5
    let g27 = 2.0 * x27  // gradient = 10.0

    // Manual calculation:
    // new_sum_sq = 0 + 10^2 = 100
    // new_param = 5 - 0.5 * 10 / (sqrt(100) + 1e-8) = 5 - 5/10 = 4.5
    let expected_sum_sq27 = sum_sq27 + g27 * g27
    let expected_x27 = x27 - lr27 * g27 / (sqrt_f64(expected_sum_sq27) + 0.00000001)

    let result27 = adagrad_step(x27, g27, sum_sq27, lr27)

    println("  Manual calculation:")
    println("    new_sum_sq = ")
    println(expected_sum_sq27)
    println("    new_param = ")
    println(expected_x27)
    println("  adagrad_step result:")
    println("    result.sum_sq = ")
    println(result27.sum_sq)
    println("    result.param = ")
    println(result27.param)

    if abs_f64(expected_sum_sq27 - 100.0) > tol { ok = false; println("  FAIL: expected_sum_sq") }
    if abs_f64(result27.sum_sq - expected_sum_sq27) > tol { ok = false; println("  FAIL: result.sum_sq") }
    if abs_f64(result27.param - expected_x27) > tol { ok = false; println("  FAIL: result.param") }
    println("")

    // Test 28: AdaGrad 5-step descent (unrolled)
    println("Test 28: AdaGrad 5-step descent (unrolled)")
    let w0_28 = 5.0
    let sq0_28 = 0.0
    let lr28 = 0.5

    // Step 1
    let gw1 = 2.0 * w0_28
    let a1 = adagrad_step(w0_28, gw1, sq0_28, lr28)
    let w1_28 = a1.param
    let sq1_28 = a1.sum_sq

    // Step 2
    let gw2 = 2.0 * w1_28
    let a2 = adagrad_step(w1_28, gw2, sq1_28, lr28)
    let w2_28 = a2.param
    let sq2_28 = a2.sum_sq

    // Step 3
    let gw3 = 2.0 * w2_28
    let a3 = adagrad_step(w2_28, gw3, sq2_28, lr28)
    let w3_28 = a3.param
    let sq3_28 = a3.sum_sq

    // Step 4
    let gw4 = 2.0 * w3_28
    let a4 = adagrad_step(w3_28, gw4, sq3_28, lr28)
    let w4_28 = a4.param
    let sq4_28 = a4.sum_sq

    // Step 5
    let gw5 = 2.0 * w4_28
    let a5 = adagrad_step(w4_28, gw5, sq4_28, lr28)
    let w5_28 = a5.param

    println("  Descent from w=5 (AdaGrad lr decays over time):")
    println("    w0 = 5.0")
    println("    w1 = ")
    println(w1_28)
    println("    w2 = ")
    println(w2_28)
    println("    w3 = ")
    println(w3_28)
    println("    w4 = ")
    println(w4_28)
    println("    w5 = ")
    println(w5_28)

    // w should decrease toward 0 (AdaGrad converges even slower as sum_sq grows)
    if w1_28 >= w0_28 { ok = false; println("  FAIL: w1 >= w0") }
    if w2_28 >= w1_28 { ok = false; println("  FAIL: w2 >= w1") }
    if w5_28 >= 4.5 { ok = false; println("  FAIL: w5 should be < 4.5 after 5 steps") }
    println("")

    // Test 29: AdaDelta single step verification
    println("Test 29: AdaDelta single step")
    let x29 = 5.0
    let acc_g29 = 0.0
    let acc_d29 = 0.0
    let rho29 = 0.95
    let g29 = 2.0 * x29  // gradient = 10.0

    // Manual calculation:
    // new_acc_grad = 0.95 * 0 + 0.05 * 100 = 5
    // rms_grad = sqrt(5 + 1e-6) ≈ 2.236
    // rms_delta = sqrt(0 + 1e-6) = 0.001
    // delta_x = -0.001/2.236 * 10 ≈ -0.00447
    // new_param ≈ 4.9955
    let eps29 = 0.000001
    let expected_acc_g29 = rho29 * acc_g29 + (1.0 - rho29) * g29 * g29
    let rms_g29 = sqrt_f64(expected_acc_g29 + eps29)
    let rms_d29 = sqrt_f64(acc_d29 + eps29)
    let delta29 = 0.0 - rms_d29 / rms_g29 * g29
    let expected_x29 = x29 + delta29

    let result29 = adadelta_step(x29, g29, acc_g29, acc_d29, rho29)

    println("  Manual calculation:")
    println("    new_acc_grad = ")
    println(expected_acc_g29)
    println("    delta_x = ")
    println(delta29)
    println("    new_param = ")
    println(expected_x29)
    println("  adadelta_step result:")
    println("    result.acc_grad = ")
    println(result29.acc_grad)
    println("    result.param = ")
    println(result29.param)

    if abs_f64(result29.acc_grad - expected_acc_g29) > tol { ok = false; println("  FAIL: result.acc_grad") }
    if abs_f64(result29.param - expected_x29) > tol { ok = false; println("  FAIL: result.param") }
    // First step should decrease param (gradient points away from 0)
    if result29.param >= x29 { ok = false; println("  FAIL: param should decrease") }
    println("")

    // Test 30: AdaDelta 5-step descent (unrolled)
    println("Test 30: AdaDelta 5-step descent (unrolled)")
    let p0_30 = 5.0
    let ag0_30 = 0.0
    let ad0_30 = 0.0
    let rho30 = 0.95

    // Step 1
    let gp1 = 2.0 * p0_30
    let d1 = adadelta_step(p0_30, gp1, ag0_30, ad0_30, rho30)
    let p1_30 = d1.param
    let ag1_30 = d1.acc_grad
    let ad1_30 = d1.acc_delta

    // Step 2
    let gp2 = 2.0 * p1_30
    let d2 = adadelta_step(p1_30, gp2, ag1_30, ad1_30, rho30)
    let p2_30 = d2.param
    let ag2_30 = d2.acc_grad
    let ad2_30 = d2.acc_delta

    // Step 3
    let gp3 = 2.0 * p2_30
    let d3 = adadelta_step(p2_30, gp3, ag2_30, ad2_30, rho30)
    let p3_30 = d3.param
    let ag3_30 = d3.acc_grad
    let ad3_30 = d3.acc_delta

    // Step 4
    let gp4 = 2.0 * p3_30
    let d4 = adadelta_step(p3_30, gp4, ag3_30, ad3_30, rho30)
    let p4_30 = d4.param
    let ag4_30 = d4.acc_grad
    let ad4_30 = d4.acc_delta

    // Step 5
    let gp5 = 2.0 * p4_30
    let d5 = adadelta_step(p4_30, gp5, ag4_30, ad4_30, rho30)
    let p5_30 = d5.param

    println("  Descent from p=5 (AdaDelta - no learning rate!):")
    println("    p0 = 5.0")
    println("    p1 = ")
    println(p1_30)
    println("    p2 = ")
    println(p2_30)
    println("    p3 = ")
    println(p3_30)
    println("    p4 = ")
    println(p4_30)
    println("    p5 = ")
    println(p5_30)

    // p should decrease toward 0
    if p1_30 >= p0_30 { ok = false; println("  FAIL: p1 >= p0") }
    if p2_30 >= p1_30 { ok = false; println("  FAIL: p2 >= p1") }
    if p5_30 >= p0_30 { ok = false; println("  FAIL: p5 should be < p0 after 5 steps") }
    println("")

    // Test 31: AdamW single step - verify weight decay is applied
    println("Test 31: AdamW single step with weight decay")
    let x31 = 5.0
    let m31 = 0.0
    let v31 = 0.0
    let lr31 = 0.1
    let wd31 = 0.01  // weight decay
    let g31 = 2.0 * x31  // gradient = 10.0

    // Compare Adam vs AdamW at timestep 1
    // Adam: just gradient update
    // AdamW: gradient update + weight decay
    let adam_result31 = adam_step_single(x31, g31, m31, v31, 1.0, lr31)
    let adamw_result31 = adamw_step(x31, g31, m31, v31, 1.0, lr31, wd31)

    // Weight decay should make AdamW param smaller than Adam param
    // decay_term = lr * wd * param = 0.1 * 0.01 * 5 = 0.005
    let expected_decay = lr31 * wd31 * x31

    println("  Adam result (no weight decay):")
    println("    param = ")
    println(adam_result31.param)
    println("  AdamW result (with weight decay):")
    println("    param = ")
    println(adamw_result31.param)
    println("  Expected decay term = ")
    println(expected_decay)
    println("  Difference (Adam - AdamW) = ")
    println(adam_result31.param - adamw_result31.param)

    // AdamW should produce smaller param due to weight decay
    if adamw_result31.param >= adam_result31.param { ok = false; println("  FAIL: AdamW should be < Adam") }
    // The difference should be approximately the decay term
    if abs_f64((adam_result31.param - adamw_result31.param) - expected_decay) > tol {
        ok = false
        println("  FAIL: decay difference incorrect")
    }
    // Moments should be the same (weight decay doesn't affect moments)
    if abs_f64(adamw_result31.m - adam_result31.m) > tol { ok = false; println("  FAIL: m should match") }
    if abs_f64(adamw_result31.v - adam_result31.v) > tol { ok = false; println("  FAIL: v should match") }
    println("")

    // Test 32: AdamW 5-step descent with weight decay (unrolled)
    println("Test 32: AdamW 5-step descent (unrolled)")
    let q0_32 = 5.0
    let mq0_32 = 0.0
    let vq0_32 = 0.0
    let lr32 = 0.1
    let wd32 = 0.01

    // Step 1
    let gq1 = 2.0 * q0_32
    let w1 = adamw_step(q0_32, gq1, mq0_32, vq0_32, 1.0, lr32, wd32)
    let q1_32 = w1.param
    let mq1_32 = w1.m
    let vq1_32 = w1.v

    // Step 2
    let gq2 = 2.0 * q1_32
    let w2 = adamw_step(q1_32, gq2, mq1_32, vq1_32, 2.0, lr32, wd32)
    let q2_32 = w2.param
    let mq2_32 = w2.m
    let vq2_32 = w2.v

    // Step 3
    let gq3 = 2.0 * q2_32
    let w3 = adamw_step(q2_32, gq3, mq2_32, vq2_32, 3.0, lr32, wd32)
    let q3_32 = w3.param
    let mq3_32 = w3.m
    let vq3_32 = w3.v

    // Step 4
    let gq4 = 2.0 * q3_32
    let w4 = adamw_step(q3_32, gq4, mq3_32, vq3_32, 4.0, lr32, wd32)
    let q4_32 = w4.param
    let mq4_32 = w4.m
    let vq4_32 = w4.v

    // Step 5
    let gq5 = 2.0 * q4_32
    let w5 = adamw_step(q4_32, gq5, mq4_32, vq4_32, 5.0, lr32, wd32)
    let q5_32 = w5.param

    println("  Descent from q=5 with AdamW (lr=0.1, wd=0.01):")
    println("    q0 = 5.0")
    println("    q1 = ")
    println(q1_32)
    println("    q2 = ")
    println(q2_32)
    println("    q3 = ")
    println(q3_32)
    println("    q4 = ")
    println(q4_32)
    println("    q5 = ")
    println(q5_32)

    // q should decrease toward 0 (AdamW converges slightly slower due to weight decay)
    if q1_32 >= q0_32 { ok = false; println("  FAIL: q1 >= q0") }
    if q2_32 >= q1_32 { ok = false; println("  FAIL: q2 >= q1") }
    if q5_32 >= 4.5 { ok = false; println("  FAIL: q5 should be < 4.5 after 5 steps") }
    println("")

    // Test 33: NAdam single step - verify Nesterov acceleration
    println("Test 33: NAdam single step with Nesterov momentum")
    let x33 = 5.0
    let m33 = 0.0
    let v33 = 0.0
    let lr33 = 0.1
    let g33 = 2.0 * x33  // gradient = 10.0

    // Compare Adam vs NAdam at timestep 1
    // NAdam should converge faster due to Nesterov look-ahead
    let adam_result33 = adam_step_single(x33, g33, m33, v33, 1.0, lr33)
    let nadam_result33 = nadam_step(x33, g33, m33, v33, 1.0, lr33)

    println("  Adam result:")
    println("    param = ")
    println(adam_result33.param)
    println("  NAdam result (with Nesterov):")
    println("    param = ")
    println(nadam_result33.param)
    println("  NAdam converges faster (smaller param):")
    println("    difference = ")
    println(adam_result33.param - nadam_result33.param)

    // NAdam should produce smaller param (faster convergence toward 0)
    if nadam_result33.param >= adam_result33.param { ok = false; println("  FAIL: NAdam should be < Adam") }
    // Both should decrease from initial
    if nadam_result33.param >= x33 { ok = false; println("  FAIL: NAdam param should decrease") }
    if adam_result33.param >= x33 { ok = false; println("  FAIL: Adam param should decrease") }
    // Moments should be the same (Nesterov only affects the update, not moment storage)
    if abs_f64(nadam_result33.m - adam_result33.m) > tol { ok = false; println("  FAIL: m should match") }
    if abs_f64(nadam_result33.v - adam_result33.v) > tol { ok = false; println("  FAIL: v should match") }
    println("")

    // Test 34: NAdam 5-step descent (unrolled)
    println("Test 34: NAdam 5-step descent (unrolled)")
    let n0_34 = 5.0
    let mn0_34 = 0.0
    let vn0_34 = 0.0
    let lr34 = 0.1

    // Step 1
    let gn1 = 2.0 * n0_34
    let nd1 = nadam_step(n0_34, gn1, mn0_34, vn0_34, 1.0, lr34)
    let n1_34 = nd1.param
    let mn1_34 = nd1.m
    let vn1_34 = nd1.v

    // Step 2
    let gn2 = 2.0 * n1_34
    let nd2 = nadam_step(n1_34, gn2, mn1_34, vn1_34, 2.0, lr34)
    let n2_34 = nd2.param
    let mn2_34 = nd2.m
    let vn2_34 = nd2.v

    // Step 3
    let gn3 = 2.0 * n2_34
    let nd3 = nadam_step(n2_34, gn3, mn2_34, vn2_34, 3.0, lr34)
    let n3_34 = nd3.param
    let mn3_34 = nd3.m
    let vn3_34 = nd3.v

    // Step 4
    let gn4 = 2.0 * n3_34
    let nd4 = nadam_step(n3_34, gn4, mn3_34, vn3_34, 4.0, lr34)
    let n4_34 = nd4.param
    let mn4_34 = nd4.m
    let vn4_34 = nd4.v

    // Step 5
    let gn5 = 2.0 * n4_34
    let nd5 = nadam_step(n4_34, gn5, mn4_34, vn4_34, 5.0, lr34)
    let n5_34 = nd5.param

    println("  Descent from n=5 with NAdam (Nesterov-accelerated):")
    println("    n0 = 5.0")
    println("    n1 = ")
    println(n1_34)
    println("    n2 = ")
    println(n2_34)
    println("    n3 = ")
    println(n3_34)
    println("    n4 = ")
    println(n4_34)
    println("    n5 = ")
    println(n5_34)

    // n should decrease toward 0 (faster than Adam due to Nesterov)
    if n1_34 >= n0_34 { ok = false; println("  FAIL: n1 >= n0") }
    if n2_34 >= n1_34 { ok = false; println("  FAIL: n2 >= n1") }
    if n5_34 >= 4.5 { ok = false; println("  FAIL: n5 should be < 4.5 after 5 steps") }
    println("")

    if ok {
        println("ALL TESTS PASSED")
        return 0
    } else {
        println("SOME TESTS FAILED")
        return 1
    }
}
