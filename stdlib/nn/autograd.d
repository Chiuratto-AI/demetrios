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

// Alias log_f64 -> ln_f64 (defined below in loss functions section)
fn log_f64(x: f64) -> f64 {
    return ln_f64(x)
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
// RADAM OPTIMIZER (RECTIFIED ADAM)
// ============================================================================

// RAdam hyperparameters (Liu et al., 2019)
fn RADAM_BETA1() -> f64 { return 0.9 }
fn RADAM_BETA2() -> f64 { return 0.999 }
fn RADAM_EPS() -> f64 { return 0.00000001 }

// Result struct for RAdam
struct RAdamResult {
    param: f64,
    m: f64,
    v: f64
}

// RAdam update for single parameter
// Addresses variance issue in Adam during early training by computing
// the length of the approximated SMA (Simple Moving Average) and only
// using adaptive learning rate when variance is tractable.
//
// Key insight: Early in training, v has high variance due to few samples.
// RAdam detects this and falls back to SGD with momentum until variance stabilizes.
//
// Formula:
//   m = β1*m + (1-β1)*g
//   v = β2*v + (1-β2)*g²
//   m_hat = m / (1-β1^t)
//
//   ρ_inf = 2/(1-β2) - 1           (max SMA length ≈ 999 for β2=0.999)
//   ρ_t = ρ_inf - 2*t*β2^t/(1-β2^t) (SMA length at timestep t)
//
//   if ρ_t > 5 (variance tractable):
//     r_t = sqrt((ρ_t-4)(ρ_t-2)ρ_inf / ((ρ_inf-4)(ρ_inf-2)ρ_t))  (rectification)
//     v_hat = v / (1-β2^t)
//     param = param - lr * r_t * m_hat / (√v_hat + ε)
//   else (variance not tractable, use unadapted):
//     param = param - lr * m_hat
fn radam_step(param: f64, g: f64, m: f64, v: f64, timestep: f64, lr: f64) -> RAdamResult {
    let beta1 = RADAM_BETA1()
    let beta2 = RADAM_BETA2()
    let eps = RADAM_EPS()

    // Update biased first moment estimate
    let new_m = beta1 * m + (1.0 - beta1) * g

    // Update biased second raw moment estimate
    let new_v = beta2 * v + (1.0 - beta2) * g * g

    // Compute bias correction for first moment
    let beta1_t = pow_f64(beta1, timestep)
    let beta2_t = pow_f64(beta2, timestep)
    let m_hat = new_m / (1.0 - beta1_t)

    // Compute maximum length of the approximated SMA
    let rho_inf = 2.0 / (1.0 - beta2) - 1.0

    // Compute length of the approximated SMA at current timestep
    let rho_t = rho_inf - 2.0 * timestep * beta2_t / (1.0 - beta2_t)

    // Check if variance is tractable (ρ_t > 5)
    let new_param = if rho_t > 5.0 {
        // Variance is tractable - use adaptive learning rate with rectification
        let v_hat = new_v / (1.0 - beta2_t)

        // Compute variance rectification term
        let rect_num = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf
        let rect_den = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t
        let r_t = sqrt_f64(rect_num / rect_den)

        // Rectified adaptive update
        param - lr * r_t * m_hat / (sqrt_f64(v_hat) + eps)
    } else {
        // Variance not tractable - use unadapted update (like SGD with momentum)
        param - lr * m_hat
    }

    return RAdamResult { param: new_param, m: new_m, v: new_v }
}

// RAdam with running powers (more efficient for training loops)
fn radam_step_fast(param: f64, g: f64, m: f64, v: f64, timestep: f64, beta1_t: f64, beta2_t: f64, lr: f64) -> RAdamResult {
    let beta1 = RADAM_BETA1()
    let beta2 = RADAM_BETA2()
    let eps = RADAM_EPS()

    // Update moments
    let new_m = beta1 * m + (1.0 - beta1) * g
    let new_v = beta2 * v + (1.0 - beta2) * g * g

    // Bias-corrected first moment
    let m_hat = new_m / (1.0 - beta1_t)

    // SMA lengths
    let rho_inf = 2.0 / (1.0 - beta2) - 1.0
    let rho_t = rho_inf - 2.0 * timestep * beta2_t / (1.0 - beta2_t)

    // Adaptive or unadapted update
    let new_param = if rho_t > 5.0 {
        let v_hat = new_v / (1.0 - beta2_t)
        let rect_num = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf
        let rect_den = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t
        let r_t = sqrt_f64(rect_num / rect_den)
        param - lr * r_t * m_hat / (sqrt_f64(v_hat) + eps)
    } else {
        param - lr * m_hat
    }

    return RAdamResult { param: new_param, m: new_m, v: new_v }
}

// ============================================================================
// LAMB OPTIMIZER (LARGE BATCH TRAINING)
// ============================================================================

// LAMB hyperparameters (You et al., 2019)
fn LAMB_BETA1() -> f64 { return 0.9 }
fn LAMB_BETA2() -> f64 { return 0.999 }
fn LAMB_EPS() -> f64 { return 0.000001 }  // Larger eps for stability
fn LAMB_WEIGHT_DECAY() -> f64 { return 0.01 }

// Result struct for LAMB
struct LAMBResult {
    param: f64,
    m: f64,
    v: f64
}

// LAMB update for single parameter
// Designed for large batch training (batch sizes up to 32K+)
//
// Key innovation: Layer-wise adaptive learning rate via "trust ratio"
// The trust ratio scales updates based on ||param|| / ||update||,
// preventing updates from being too large relative to parameter magnitude.
//
// Formula:
//   m = β1*m + (1-β1)*g
//   v = β2*v + (1-β2)*g²
//   m_hat = m / (1-β1^t)
//   v_hat = v / (1-β2^t)
//   adam_update = m_hat / (√v_hat + ε) + λ*param   (with weight decay)
//
//   trust_ratio = ||param|| / ||adam_update||
//   (clamped to [0, 10] for stability, defaults to 1 if either norm is 0)
//
//   param = param - lr * trust_ratio * adam_update
//
// For single parameters, ||param|| = |param| and ||update|| = |update|
fn lamb_step(param: f64, g: f64, m: f64, v: f64, timestep: f64, lr: f64, weight_decay: f64) -> LAMBResult {
    let beta1 = LAMB_BETA1()
    let beta2 = LAMB_BETA2()
    let eps = LAMB_EPS()

    // Update biased first moment estimate
    let new_m = beta1 * m + (1.0 - beta1) * g

    // Update biased second raw moment estimate
    let new_v = beta2 * v + (1.0 - beta2) * g * g

    // Compute bias-corrected estimates
    let beta1_t = pow_f64(beta1, timestep)
    let beta2_t = pow_f64(beta2, timestep)
    let m_hat = new_m / (1.0 - beta1_t)
    let v_hat = new_v / (1.0 - beta2_t)

    // Compute Adam update with weight decay (AdamW style)
    let adam_update = m_hat / (sqrt_f64(v_hat) + eps) + weight_decay * param

    // Compute norms for trust ratio (for single param, norm = absolute value)
    let param_norm = abs_f64(param)
    let update_norm = abs_f64(adam_update)

    // Compute trust ratio with safety checks
    let trust_ratio = if param_norm > 0.0 {
        if update_norm > 0.0 {
            // Clamp trust ratio to [0, 10] for stability
            let ratio = param_norm / update_norm
            if ratio > 10.0 { 10.0 } else { ratio }
        } else {
            1.0  // Default if update is zero
        }
    } else {
        1.0  // Default if param is zero
    }

    // Apply update with trust ratio scaling
    let new_param = param - lr * trust_ratio * adam_update

    return LAMBResult { param: new_param, m: new_m, v: new_v }
}

// LAMB with running powers (more efficient for training loops)
fn lamb_step_fast(param: f64, g: f64, m: f64, v: f64, beta1_t: f64, beta2_t: f64, lr: f64, weight_decay: f64) -> LAMBResult {
    let beta1 = LAMB_BETA1()
    let beta2 = LAMB_BETA2()
    let eps = LAMB_EPS()

    // Update moments
    let new_m = beta1 * m + (1.0 - beta1) * g
    let new_v = beta2 * v + (1.0 - beta2) * g * g

    // Bias-corrected estimates
    let m_hat = new_m / (1.0 - beta1_t)
    let v_hat = new_v / (1.0 - beta2_t)

    // Adam update with weight decay
    let adam_update = m_hat / (sqrt_f64(v_hat) + eps) + weight_decay * param

    // Trust ratio computation
    let param_norm = abs_f64(param)
    let update_norm = abs_f64(adam_update)

    let trust_ratio = if param_norm > 0.0 {
        if update_norm > 0.0 {
            let ratio = param_norm / update_norm
            if ratio > 10.0 { 10.0 } else { ratio }
        } else {
            1.0
        }
    } else {
        1.0
    }

    let new_param = param - lr * trust_ratio * adam_update

    return LAMBResult { param: new_param, m: new_m, v: new_v }
}

// ============================================================================
// LION OPTIMIZER (EVOLVED SIGN MOMENTUM)
// ============================================================================

// Lion hyperparameters (Chen et al., Google 2023)
// Note: Lion uses different betas than Adam!
fn LION_BETA1() -> f64 { return 0.9 }   // For update interpolation
fn LION_BETA2() -> f64 { return 0.99 }  // For momentum update (not 0.999!)
fn LION_WEIGHT_DECAY() -> f64 { return 0.01 }

// Result struct for Lion (only needs momentum, no second moment!)
struct LionResult {
    param: f64,
    m: f64
}

// Sign function: returns -1, 0, or 1
fn sign_f64(x: f64) -> f64 {
    if x > 0.0 { return 1.0 }
    if x < 0.0 { return 0.0 - 1.0 }
    return 0.0
}

// Lion update for single parameter
// Discovered through program search, simpler and often better than Adam
//
// Key innovations:
// 1. Uses sign() of interpolated momentum (uniform magnitude updates)
// 2. Momentum updated AFTER parameter update (different from Adam)
// 3. No second moment tracking (memory efficient - only stores m, not v)
// 4. Typically needs 3-10x smaller learning rate than Adam
//
// Formula:
//   update = sign(β1 * m + (1 - β1) * g)        <- sign of interpolation
//   param = param - lr * (update + λ * param)   <- with weight decay
//   m = β2 * m + (1 - β2) * g                   <- momentum update AFTER
//
// Note: The order matters! Momentum is updated after using it for the update.
fn lion_step(param: f64, g: f64, m: f64, lr: f64, weight_decay: f64) -> LionResult {
    let beta1 = LION_BETA1()
    let beta2 = LION_BETA2()

    // Compute interpolation for update direction
    let interpolated = beta1 * m + (1.0 - beta1) * g

    // Take sign of interpolation (this is the key innovation!)
    let update = sign_f64(interpolated)

    // Apply update with decoupled weight decay
    let new_param = param - lr * update - lr * weight_decay * param

    // Update momentum AFTER using it (different from Adam!)
    let new_m = beta2 * m + (1.0 - beta2) * g

    return LionResult { param: new_param, m: new_m }
}

// Lion without weight decay
fn lion_step_no_wd(param: f64, g: f64, m: f64, lr: f64) -> LionResult {
    let beta1 = LION_BETA1()
    let beta2 = LION_BETA2()

    // Compute update direction
    let interpolated = beta1 * m + (1.0 - beta1) * g
    let update = sign_f64(interpolated)

    // Apply update (no weight decay)
    let new_param = param - lr * update

    // Update momentum after
    let new_m = beta2 * m + (1.0 - beta2) * g

    return LionResult { param: new_param, m: new_m }
}

// ============================================================================
// LEARNING RATE SCHEDULERS
// ============================================================================

// Constant learning rate (baseline)
fn lr_constant(initial_lr: f64, step: f64) -> f64 {
    return initial_lr
}

// Step decay: reduce LR by gamma every step_size steps
// lr = initial_lr * gamma^(floor(step / step_size))
fn lr_step_decay(initial_lr: f64, step: f64, step_size: f64, gamma: f64) -> f64 {
    let num_decays = floor_f64(step / step_size)
    return initial_lr * pow_f64(gamma, num_decays)
}

// Floor function - efficient non-recursive implementation
fn floor_f64(x: f64) -> f64 {
    if x >= 0.0 {
        // For small positive numbers, use simple digit extraction
        if x < 1.0 { return 0.0 }

        // Decompose using powers of 10 (fast for reasonable numbers)
        let mut result = 0.0
        let mut remaining = x

        // Handle up to 10^15 (well within f64 precision)
        let mut power = 1000000000000000.0  // 10^15
        while power >= 1.0 {
            while remaining >= power {
                remaining = remaining - power
                result = result + power
            }
            power = power / 10.0
        }
        return result
    } else {
        // For negative numbers: floor(-2.3) = -3
        let pos = 0.0 - x
        let pos_floor = floor_f64(pos)
        if pos > pos_floor {
            return 0.0 - pos_floor - 1.0
        }
        return 0.0 - pos_floor
    }
}

// Exponential decay: lr = initial_lr * decay_rate^step
fn lr_exponential_decay(initial_lr: f64, step: f64, decay_rate: f64) -> f64 {
    return initial_lr * pow_f64(decay_rate, step)
}

// Linear decay: lr decreases linearly from initial_lr to end_lr
// lr = initial_lr + (end_lr - initial_lr) * (step / total_steps)
fn lr_linear_decay(initial_lr: f64, end_lr: f64, step: f64, total_steps: f64) -> f64 {
    if step >= total_steps {
        return end_lr
    }
    let progress = step / total_steps
    return initial_lr + (end_lr - initial_lr) * progress
}

// Polynomial decay: lr = initial_lr * (1 - step/total_steps)^power
fn lr_polynomial_decay(initial_lr: f64, step: f64, total_steps: f64, power: f64) -> f64 {
    if step >= total_steps {
        return 0.0
    }
    let decay_factor = 1.0 - step / total_steps
    return initial_lr * pow_f64(decay_factor, power)
}

// Cosine annealing: lr follows cosine curve from initial_lr to min_lr
// lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(π * step / total_steps))
fn lr_cosine_annealing(initial_lr: f64, min_lr: f64, step: f64, total_steps: f64) -> f64 {
    if step >= total_steps {
        return min_lr
    }
    let pi = 3.14159265358979323846
    let progress = step / total_steps
    let cosine_value = cos_f64(pi * progress)
    return min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + cosine_value)
}

// Cosine function using Taylor series with proper range reduction
fn cos_f64(x: f64) -> f64 {
    let pi = 3.14159265358979323846
    let two_pi = 2.0 * pi
    let half_pi = pi / 2.0

    // Normalize to [0, 2π)
    let mut a = x
    while a >= two_pi { a = a - two_pi }
    while a < 0.0 { a = a + two_pi }

    // Reduce to [0, π/2] using symmetry
    // cos(x) = cos(2π - x) for x in [π, 2π]
    // cos(x) = -cos(π - x) for x in [π/2, π]
    let mut sign = 1.0
    if a > pi {
        a = two_pi - a
    }
    if a > half_pi {
        sign = 0.0 - 1.0
        a = pi - a
    }

    // Now a is in [0, π/2], Taylor series converges well
    let x2 = a * a
    let x4 = x2 * x2
    let x6 = x4 * x2
    let x8 = x4 * x4
    let x10 = x6 * x4

    return sign * (1.0 - x2/2.0 + x4/24.0 - x6/720.0 + x8/40320.0 - x10/3628800.0)
}

// Linear warmup: gradually increase LR from 0 to initial_lr
// lr = initial_lr * (step / warmup_steps) for step < warmup_steps
fn lr_linear_warmup(initial_lr: f64, step: f64, warmup_steps: f64) -> f64 {
    if step >= warmup_steps {
        return initial_lr
    }
    return initial_lr * (step / warmup_steps)
}

// Warmup + Cosine decay: linear warmup then cosine annealing
fn lr_warmup_cosine(initial_lr: f64, min_lr: f64, step: f64, warmup_steps: f64, total_steps: f64) -> f64 {
    if step < warmup_steps {
        // Linear warmup phase
        return initial_lr * (step / warmup_steps)
    } else {
        // Cosine annealing phase
        let decay_steps = total_steps - warmup_steps
        let decay_step = step - warmup_steps
        return lr_cosine_annealing(initial_lr, min_lr, decay_step, decay_steps)
    }
}

// Warmup + Linear decay
fn lr_warmup_linear(initial_lr: f64, end_lr: f64, step: f64, warmup_steps: f64, total_steps: f64) -> f64 {
    if step < warmup_steps {
        return initial_lr * (step / warmup_steps)
    } else {
        let decay_steps = total_steps - warmup_steps
        let decay_step = step - warmup_steps
        return lr_linear_decay(initial_lr, end_lr, decay_step, decay_steps)
    }
}

// One Cycle policy: LR increases then decreases (Smith, 2018)
// Popular for fast training with super-convergence
fn lr_one_cycle(initial_lr: f64, max_lr: f64, step: f64, total_steps: f64, pct_start: f64) -> f64 {
    let warmup_steps = total_steps * pct_start
    let pi = 3.14159265358979323846

    if step < warmup_steps {
        // Increasing phase: initial_lr to max_lr
        let progress = step / warmup_steps
        return initial_lr + (max_lr - initial_lr) * progress
    } else {
        // Decreasing phase: max_lr to ~0
        let decay_steps = total_steps - warmup_steps
        let decay_step = step - warmup_steps
        let progress = decay_step / decay_steps
        // Cosine decay from max_lr to near 0
        let cosine_value = cos_f64(pi * progress)
        return max_lr * 0.5 * (1.0 + cosine_value)
    }
}

// Inverse square root decay (commonly used in Transformers)
// lr = initial_lr * sqrt(warmup_steps) / sqrt(max(step, warmup_steps))
fn lr_inverse_sqrt(initial_lr: f64, step: f64, warmup_steps: f64) -> f64 {
    if step < warmup_steps {
        // Linear warmup
        return initial_lr * (step / warmup_steps)
    } else {
        // Inverse sqrt decay
        return initial_lr * sqrt_f64(warmup_steps) / sqrt_f64(step)
    }
}

// Cyclic learning rate: oscillates between min and max
// Useful for escaping local minima
fn lr_cyclic(min_lr: f64, max_lr: f64, step: f64, cycle_length: f64) -> f64 {
    let pi = 3.14159265358979323846
    // Use absolute value of cosine for triangular wave
    let cycle_pos = step / cycle_length
    let cosine_value = cos_f64(2.0 * pi * cycle_pos)
    // Map cos from [-1, 1] to [0, 1]
    let normalized = (cosine_value + 1.0) / 2.0
    return min_lr + (max_lr - min_lr) * normalized
}

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================
// Each loss function has:
// - loss_*: compute the loss value
// - loss_*_grad: compute gradient w.r.t. prediction
// For use with autograd, use tape operations instead

// Natural log function using Taylor series
fn ln_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 - 1000000.0 }  // -inf approximation
    if x == 1.0 { return 0.0 }

    // Range reduction: bring x to [0.5, 2] for better convergence
    let ln2 = 0.6931471805599453
    let mut val = x
    let mut adj = 0.0

    // Handle large values: divide by 2 repeatedly
    while val > 2.0 {
        val = val / 2.0
        adj = adj + ln2
    }

    // Handle small values: multiply by 2 repeatedly
    while val < 0.5 {
        val = val * 2.0
        adj = adj - ln2
    }

    // Now val is in [0.5, 2], use arctanh series
    // ln(x) = 2 * arctanh((x-1)/(x+1))
    let y = (val - 1.0) / (val + 1.0)
    let y2 = y * y
    let y3 = y2 * y
    let y5 = y3 * y2
    let y7 = y5 * y2
    let y9 = y7 * y2
    let y11 = y9 * y2
    let y13 = y11 * y2
    let y15 = y13 * y2

    let ln_val = 2.0 * (y + y3/3.0 + y5/5.0 + y7/7.0 + y9/9.0 + y11/11.0 + y13/13.0 + y15/15.0)
    return ln_val + adj
}

// ----------------------------------------------------------------------------
// MEAN SQUARED ERROR (MSE) - L2 Loss
// ----------------------------------------------------------------------------
// MSE = (1/n) * Σ(pred - target)²
// Used for: regression tasks

// MSE for single sample
fn loss_mse(pred: f64, target: f64) -> f64 {
    let diff = pred - target
    return diff * diff
}

// MSE gradient: d(MSE)/d(pred) = 2 * (pred - target)
fn loss_mse_grad(pred: f64, target: f64) -> f64 {
    return 2.0 * (pred - target)
}

// MSE for n samples (mean)
fn loss_mse_mean(preds: f64, targets: f64, n: f64) -> f64 {
    let diff = preds - targets
    return (diff * diff) / n
}

// ----------------------------------------------------------------------------
// MEAN ABSOLUTE ERROR (MAE) - L1 Loss
// ----------------------------------------------------------------------------
// MAE = (1/n) * Σ|pred - target|
// Used for: regression, more robust to outliers than MSE

fn loss_mae(pred: f64, target: f64) -> f64 {
    return abs_f64(pred - target)
}

// MAE gradient: d(MAE)/d(pred) = sign(pred - target)
fn loss_mae_grad(pred: f64, target: f64) -> f64 {
    let diff = pred - target
    if diff > 0.0 { return 1.0 }
    if diff < 0.0 { return 0.0 - 1.0 }
    return 0.0
}

// ----------------------------------------------------------------------------
// HUBER LOSS - Smooth L1
// ----------------------------------------------------------------------------
// Huber(x) = 0.5 * x² if |x| <= δ
//          = δ * (|x| - 0.5 * δ) otherwise
// Used for: regression, robust to outliers

fn HUBER_DELTA() -> f64 { return 1.0 }

fn loss_huber(pred: f64, target: f64, delta: f64) -> f64 {
    let diff = pred - target
    let abs_diff = abs_f64(diff)
    if abs_diff <= delta {
        return 0.5 * diff * diff
    } else {
        return delta * (abs_diff - 0.5 * delta)
    }
}

// Huber gradient
fn loss_huber_grad(pred: f64, target: f64, delta: f64) -> f64 {
    let diff = pred - target
    let abs_diff = abs_f64(diff)
    if abs_diff <= delta {
        return diff
    } else {
        if diff > 0.0 { return delta }
        return 0.0 - delta
    }
}

// Huber with default delta=1.0
fn loss_huber_default(pred: f64, target: f64) -> f64 {
    return loss_huber(pred, target, HUBER_DELTA())
}

fn loss_huber_grad_default(pred: f64, target: f64) -> f64 {
    return loss_huber_grad(pred, target, HUBER_DELTA())
}

// ----------------------------------------------------------------------------
// BINARY CROSS-ENTROPY (BCE)
// ----------------------------------------------------------------------------
// BCE = -[y * log(p) + (1-y) * log(1-p)]
// Used for: binary classification
// Note: pred should be in (0, 1), typically after sigmoid

fn loss_bce(pred: f64, target: f64) -> f64 {
    // Clamp pred to avoid log(0)
    let eps = 0.0000001
    let p = if pred < eps { eps } else { if pred > 1.0 - eps { 1.0 - eps } else { pred } }
    return 0.0 - (target * ln_f64(p) + (1.0 - target) * ln_f64(1.0 - p))
}

// BCE gradient: d(BCE)/d(pred) = (pred - target) / (pred * (1 - pred))
fn loss_bce_grad(pred: f64, target: f64) -> f64 {
    let eps = 0.0000001
    let p = if pred < eps { eps } else { if pred > 1.0 - eps { 1.0 - eps } else { pred } }
    return (p - target) / (p * (1.0 - p))
}

// BCE with logits (more numerically stable)
// BCE_logits = max(z, 0) - z*y + log(1 + exp(-|z|))
fn loss_bce_logits(logit: f64, target: f64) -> f64 {
    let abs_logit = abs_f64(logit)
    let max_val = if logit > 0.0 { logit } else { 0.0 }
    return max_val - logit * target + ln_f64(1.0 + exp_f64(0.0 - abs_logit))
}

// BCE with logits gradient: sigmoid(z) - y
fn loss_bce_logits_grad(logit: f64, target: f64) -> f64 {
    let sigmoid_z = 1.0 / (1.0 + exp_f64(0.0 - logit))
    return sigmoid_z - target
}

// ----------------------------------------------------------------------------
// CROSS-ENTROPY (CE) - for multi-class (single sample, single class)
// ----------------------------------------------------------------------------
// CE = -log(p_correct)
// Used for: multi-class classification
// Note: For full softmax CE, sum over all samples

// Cross-entropy for the correct class probability
fn loss_ce(pred_prob: f64) -> f64 {
    let eps = 0.0000001
    let p = if pred_prob < eps { eps } else { pred_prob }
    return 0.0 - ln_f64(p)
}

// CE gradient w.r.t. correct class probability: -1/p
fn loss_ce_grad(pred_prob: f64) -> f64 {
    let eps = 0.0000001
    let p = if pred_prob < eps { eps } else { pred_prob }
    return 0.0 - 1.0 / p
}

// Softmax cross-entropy gradient (after softmax): pred - one_hot
// For the correct class: pred - 1
// For other classes: pred - 0 = pred
fn loss_softmax_ce_grad(pred_prob: f64, is_correct: f64) -> f64 {
    return pred_prob - is_correct
}

// ----------------------------------------------------------------------------
// HINGE LOSS - SVM-style
// ----------------------------------------------------------------------------
// Hinge = max(0, 1 - y * pred)
// Used for: binary classification (y ∈ {-1, +1})

fn loss_hinge(pred: f64, target: f64) -> f64 {
    let margin = 1.0 - target * pred
    if margin > 0.0 { return margin }
    return 0.0
}

// Hinge gradient
fn loss_hinge_grad(pred: f64, target: f64) -> f64 {
    let margin = 1.0 - target * pred
    if margin > 0.0 { return 0.0 - target }
    return 0.0
}

// Squared hinge loss: max(0, 1 - y * pred)²
fn loss_hinge_squared(pred: f64, target: f64) -> f64 {
    let margin = 1.0 - target * pred
    if margin > 0.0 { return margin * margin }
    return 0.0
}

fn loss_hinge_squared_grad(pred: f64, target: f64) -> f64 {
    let margin = 1.0 - target * pred
    if margin > 0.0 { return 0.0 - 2.0 * margin * target }
    return 0.0
}

// ----------------------------------------------------------------------------
// KL DIVERGENCE
// ----------------------------------------------------------------------------
// KL(P||Q) = Σ p * log(p/q)
// Used for: comparing probability distributions, VAEs

// KL divergence for single probability pair
fn loss_kl_div(p: f64, q: f64) -> f64 {
    let eps = 0.0000001
    if p < eps { return 0.0 }  // 0 * log(0/q) = 0
    let q_safe = if q < eps { eps } else { q }
    return p * ln_f64(p / q_safe)
}

// KL gradient w.r.t. q: -p/q
fn loss_kl_div_grad_q(p: f64, q: f64) -> f64 {
    let eps = 0.0000001
    let q_safe = if q < eps { eps } else { q }
    return 0.0 - p / q_safe
}

// ----------------------------------------------------------------------------
// FOCAL LOSS - for imbalanced classification
// ----------------------------------------------------------------------------
// Focal = -α * (1-p)^γ * log(p) for positive class
// Used for: object detection, imbalanced datasets (Lin et al., 2017)

fn FOCAL_ALPHA() -> f64 { return 0.25 }
fn FOCAL_GAMMA() -> f64 { return 2.0 }

fn loss_focal(pred: f64, target: f64, alpha: f64, gamma: f64) -> f64 {
    let eps = 0.0000001
    let p = if pred < eps { eps } else { if pred > 1.0 - eps { 1.0 - eps } else { pred } }

    if target > 0.5 {
        // Positive class: -α * (1-p)^γ * log(p)
        let focal_weight = pow_f64(1.0 - p, gamma)
        return 0.0 - alpha * focal_weight * ln_f64(p)
    } else {
        // Negative class: -(1-α) * p^γ * log(1-p)
        let focal_weight = pow_f64(p, gamma)
        return 0.0 - (1.0 - alpha) * focal_weight * ln_f64(1.0 - p)
    }
}

// Focal loss with default parameters
fn loss_focal_default(pred: f64, target: f64) -> f64 {
    return loss_focal(pred, target, FOCAL_ALPHA(), FOCAL_GAMMA())
}

// ----------------------------------------------------------------------------
// SMOOTH L1 LOSS (same as Huber with delta=1)
// ----------------------------------------------------------------------------
// Used in: Faster R-CNN, object detection

fn loss_smooth_l1(pred: f64, target: f64) -> f64 {
    return loss_huber(pred, target, 1.0)
}

fn loss_smooth_l1_grad(pred: f64, target: f64) -> f64 {
    return loss_huber_grad(pred, target, 1.0)
}

// ----------------------------------------------------------------------------
// LOG COSH LOSS - smooth approximation to MAE
// ----------------------------------------------------------------------------
// LogCosh = log(cosh(pred - target))
// Used for: regression, smoother than Huber

fn cosh_f64(x: f64) -> f64 {
    return (exp_f64(x) + exp_f64(0.0 - x)) / 2.0
}

fn tanh_f64(x: f64) -> f64 {
    let e2x = exp_f64(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)
}

fn loss_log_cosh(pred: f64, target: f64) -> f64 {
    let diff = pred - target
    return ln_f64(cosh_f64(diff))
}

// LogCosh gradient: tanh(pred - target)
fn loss_log_cosh_grad(pred: f64, target: f64) -> f64 {
    return tanh_f64(pred - target)
}

// ----------------------------------------------------------------------------
// QUANTILE LOSS - for quantile regression
// ----------------------------------------------------------------------------
// Quantile(q) = q * max(y - pred, 0) + (1-q) * max(pred - y, 0)
// Used for: predicting confidence intervals

fn loss_quantile(pred: f64, target: f64, quantile: f64) -> f64 {
    let diff = target - pred
    if diff >= 0.0 {
        return quantile * diff
    } else {
        return (quantile - 1.0) * diff
    }
}

fn loss_quantile_grad(pred: f64, target: f64, quantile: f64) -> f64 {
    let diff = target - pred
    if diff >= 0.0 {
        return 0.0 - quantile
    } else {
        return 1.0 - quantile
    }
}

// ----------------------------------------------------------------------------
// COSINE SIMILARITY LOSS
// ----------------------------------------------------------------------------
// CosineLoss = 1 - cos_sim(a, b) = 1 - (a·b)/(|a||b|)
// For single values, this simplifies
// Used for: embedding similarity, contrastive learning

fn loss_cosine(pred: f64, target: f64) -> f64 {
    let eps = 0.0000001
    let pred_norm = abs_f64(pred) + eps
    let target_norm = abs_f64(target) + eps
    let cos_sim = (pred * target) / (pred_norm * target_norm)
    return 1.0 - cos_sim
}

// ----------------------------------------------------------------------------
// TRIPLET MARGIN LOSS
// ----------------------------------------------------------------------------
// Triplet = max(0, d(a,p) - d(a,n) + margin)
// Used for: metric learning, face recognition
// Note: This is a simplified version for scalar values

fn loss_triplet_margin(anchor: f64, positive: f64, negative: f64, margin: f64) -> f64 {
    let d_pos = abs_f64(anchor - positive)
    let d_neg = abs_f64(anchor - negative)
    let loss = d_pos - d_neg + margin
    if loss > 0.0 { return loss }
    return 0.0
}

// Default margin = 1.0
fn loss_triplet_default(anchor: f64, positive: f64, negative: f64) -> f64 {
    return loss_triplet_margin(anchor, positive, negative, 1.0)
}

// ============================================================================
// WEIGHT INITIALIZATION
// ============================================================================
// Proper weight initialization is critical for training deep networks.
// Different activation functions require different initialization strategies.

// ----------------------------------------------------------------------------
// PSEUDO-RANDOM NUMBER GENERATOR (LCG)
// ----------------------------------------------------------------------------
// Linear Congruential Generator for reproducible random weights
// State is passed through and returned for functional style

struct RngSt {
    seed: f64
}

fn rng_new(seed: f64) -> RngSt {
    // Ensure seed is in valid range [1, m-1]
    let m = 2147483647.0  // 2^31 - 1
    let mut s = seed
    if s <= 0.0 { s = 1.0 }
    if s >= m { s = m - 1.0 }
    return RngSt { seed: s }
}

// Generate next random number, returns (value in [0,1), new_state)
// Uses Parks-Miller MINSTD: seed' = (16807 * seed) mod (2^31-1)
// This keeps intermediate values within f64 precision (max ~3.6e13 < 2^53)
struct RngResult {
    value: f64,
    rng: RngSt
}

fn rng_next(st: RngSt) -> RngResult {
    // MINSTD parameters - safe for f64 arithmetic
    let a = 16807.0
    let m = 2147483647.0  // 2^31 - 1 (Mersenne prime)

    // Use Schrage's method to avoid overflow:
    // (a * seed) mod m = a * (seed mod q) - r * (seed / q)
    // where q = m / a, r = m mod a
    let q = 127773.0   // floor(m/a)
    let r = 2836.0     // m mod a

    let k = floor_f64(st.seed / q)
    let new_seed_raw = a * (st.seed - k * q) - r * k

    // Handle negative result
    let new_seed = if new_seed_raw < 0.0 { new_seed_raw + m } else { new_seed_raw }

    let value = new_seed / m
    return RngResult { value: value, rng: RngSt { seed: new_seed } }
}

// Floating point modulo (kept for other uses)
fn fmod_f64(x: f64, y: f64) -> f64 {
    if y == 0.0 { return 0.0 }
    let quotient = floor_f64(x / y)
    return x - quotient * y
}

// Generate uniform random in [low, high)
fn rng_uniform(st: RngSt, low: f64, high: f64) -> RngResult {
    let r = rng_next(st)
    let scaled = low + r.value * (high - low)
    return RngResult { value: scaled, rng: r.rng }
}

// Box-Muller transform for normal distribution
// Returns two independent normal samples
struct RngNormalResult {
    value1: f64,
    value2: f64,
    rng: RngSt
}

fn rng_normal_pair(st: RngSt, mean: f64, std: f64) -> RngNormalResult {
    // Generate two uniform samples
    let r1 = rng_next(st)
    let r2 = rng_next(r1.rng)

    // Avoid log(0)
    let u1 = if r1.value < 0.0000001 { 0.0000001 } else { r1.value }
    let u2 = r2.value

    // Box-Muller transform
    let pi = 3.14159265358979323846
    let mag = std * sqrt_f64(0.0 - 2.0 * ln_f64(u1))
    let z1 = mag * cos_f64(2.0 * pi * u2) + mean
    let z2 = mag * sin_f64(2.0 * pi * u2) + mean

    return RngNormalResult { value1: z1, value2: z2, rng: r2.rng }
}

// Sine function using Taylor series with proper range reduction
fn sin_f64(x: f64) -> f64 {
    let pi = 3.14159265358979323846
    let two_pi = 2.0 * pi
    let half_pi = pi / 2.0

    // Normalize to [0, 2π)
    let mut a = x
    while a >= two_pi { a = a - two_pi }
    while a < 0.0 { a = a + two_pi }

    // Reduce to [0, π/2] using symmetry
    let mut sign = 1.0
    if a > pi {
        sign = 0.0 - 1.0
        a = a - pi
    }
    if a > half_pi {
        a = pi - a
    }

    // Now a is in [0, π/2], Taylor series converges well
    let x2 = a * a
    let x3 = x2 * a
    let x5 = x3 * x2
    let x7 = x5 * x2
    let x9 = x7 * x2
    let x11 = x9 * x2

    return sign * (a - x3/6.0 + x5/120.0 - x7/5040.0 + x9/362880.0 - x11/39916800.0)
}

// Single normal sample (uses first of pair)
fn rng_normal(st: RngSt, mean: f64, std: f64) -> RngResult {
    let pair = rng_normal_pair(st, mean, std)
    return RngResult { value: pair.value1, rng: pair.rng }
}

// ----------------------------------------------------------------------------
// XAVIER/GLOROT INITIALIZATION
// ----------------------------------------------------------------------------
// For tanh and sigmoid activations
// Maintains variance across layers to prevent vanishing/exploding gradients
//
// Xavier Uniform: U[-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
// Xavier Normal: N(0, std) where std = sqrt(2 / (fan_in + fan_out))

// Calculate Xavier uniform bounds
fn xavier_uniform_bound(fan_in: f64, fan_out: f64) -> f64 {
    return sqrt_f64(6.0 / (fan_in + fan_out))
}

// Calculate Xavier normal std
fn xavier_normal_std(fan_in: f64, fan_out: f64) -> f64 {
    return sqrt_f64(2.0 / (fan_in + fan_out))
}

// Generate Xavier uniform weight
fn init_xavier_uniform(st: RngSt, fan_in: f64, fan_out: f64) -> RngResult {
    let bound = xavier_uniform_bound(fan_in, fan_out)
    return rng_uniform(st, 0.0 - bound, bound)
}

// Generate Xavier normal weight
fn init_xavier_normal(st: RngSt, fan_in: f64, fan_out: f64) -> RngResult {
    let std = xavier_normal_std(fan_in, fan_out)
    return rng_normal(st, 0.0, std)
}

// ----------------------------------------------------------------------------
// HE/KAIMING INITIALIZATION
// ----------------------------------------------------------------------------
// For ReLU and variants (designed for asymmetric activations)
// Accounts for the fact that ReLU zeros out negative values
//
// He Uniform: U[-limit, limit] where limit = sqrt(6 / fan_in)
// He Normal: N(0, std) where std = sqrt(2 / fan_in)
//
// For LeakyReLU with negative_slope a:
// std = sqrt(2 / ((1 + a²) * fan_in))

// Calculate He uniform bound
fn he_uniform_bound(fan_in: f64) -> f64 {
    return sqrt_f64(6.0 / fan_in)
}

// Calculate He normal std
fn he_normal_std(fan_in: f64) -> f64 {
    return sqrt_f64(2.0 / fan_in)
}

// He for LeakyReLU
fn he_normal_std_leaky(fan_in: f64, negative_slope: f64) -> f64 {
    return sqrt_f64(2.0 / ((1.0 + negative_slope * negative_slope) * fan_in))
}

// Generate He uniform weight
fn init_he_uniform(st: RngSt, fan_in: f64) -> RngResult {
    let bound = he_uniform_bound(fan_in)
    return rng_uniform(st, 0.0 - bound, bound)
}

// Generate He normal weight
fn init_he_normal(st: RngSt, fan_in: f64) -> RngResult {
    let std = he_normal_std(fan_in)
    return rng_normal(st, 0.0, std)
}

// He for LeakyReLU
fn init_he_leaky(st: RngSt, fan_in: f64, negative_slope: f64) -> RngResult {
    let std = he_normal_std_leaky(fan_in, negative_slope)
    return rng_normal(st, 0.0, std)
}

// Alias: Kaiming = He
fn init_kaiming_uniform(st: RngSt, fan_in: f64) -> RngResult {
    return init_he_uniform(st, fan_in)
}

fn init_kaiming_normal(st: RngSt, fan_in: f64) -> RngResult {
    return init_he_normal(st, fan_in)
}

// ----------------------------------------------------------------------------
// LECUN INITIALIZATION
// ----------------------------------------------------------------------------
// For SELU activation (self-normalizing networks)
// LeCun Normal: N(0, std) where std = sqrt(1 / fan_in)

fn lecun_normal_std(fan_in: f64) -> f64 {
    return sqrt_f64(1.0 / fan_in)
}

fn init_lecun_normal(st: RngSt, fan_in: f64) -> RngResult {
    let std = lecun_normal_std(fan_in)
    return rng_normal(st, 0.0, std)
}

fn init_lecun_uniform(st: RngSt, fan_in: f64) -> RngResult {
    let bound = sqrt_f64(3.0 / fan_in)
    return rng_uniform(st, 0.0 - bound, bound)
}

// ----------------------------------------------------------------------------
// BASIC INITIALIZATIONS
// ----------------------------------------------------------------------------

// Constant initialization
fn init_constant(value: f64) -> f64 {
    return value
}

// Zero initialization (use sparingly - can cause dead neurons)
fn init_zeros() -> f64 {
    return 0.0
}

// One initialization
fn init_ones() -> f64 {
    return 1.0
}

// Uniform initialization in [low, high)
fn init_uniform(st: RngSt, low: f64, high: f64) -> RngResult {
    return rng_uniform(st, low, high)
}

// Normal initialization with given mean and std
fn init_normal(st: RngSt, mean: f64, std: f64) -> RngResult {
    return rng_normal(st, mean, std)
}

// Standard normal N(0, 1)
fn init_standard_normal(st: RngSt) -> RngResult {
    return rng_normal(st, 0.0, 1.0)
}

// ----------------------------------------------------------------------------
// TRUNCATED NORMAL INITIALIZATION
// ----------------------------------------------------------------------------
// Normal distribution but values beyond 2*std are redrawn
// Used in TensorFlow's default initializers

fn init_truncated_normal(st: RngSt, mean: f64, std: f64) -> RngResult {
    let mut cur_rng = st
    let mut value = 0.0
    let mut found = false
    let mut iterations = 0

    // Rejection sampling (max 10 iterations to avoid infinite loop)
    while iterations < 10 {
        let r = rng_normal(cur_rng, mean, std)
        cur_rng = r.rng
        value = r.value

        // Accept if within 2 standard deviations
        if abs_f64(value - mean) <= 2.0 * std {
            found = true
            iterations = 10  // Exit loop
        }
        iterations = iterations + 1
    }

    // If not found after 10 tries, clamp to bounds
    if found == false {
        if value > mean + 2.0 * std {
            value = mean + 2.0 * std
        }
        if value < mean - 2.0 * std {
            value = mean - 2.0 * std
        }
    }

    return RngResult { value: value, rng: cur_rng }
}

// ----------------------------------------------------------------------------
// SPARSE INITIALIZATION
// ----------------------------------------------------------------------------
// Initialize with zeros except for a fraction of weights
// sparsity: fraction of weights to set to zero (0.0 = dense, 0.9 = 90% zeros)

fn init_sparse(st: RngSt, std: f64, sparsity: f64) -> RngResult {
    let r1 = rng_next(st)

    if r1.value < sparsity {
        // Zero with probability = sparsity
        return RngResult { value: 0.0, rng: r1.rng }
    } else {
        // Normal with probability = 1 - sparsity
        return rng_normal(r1.rng, 0.0, std)
    }
}

// ----------------------------------------------------------------------------
// ORTHOGONAL INITIALIZATION (simplified scalar version)
// ----------------------------------------------------------------------------
// For matrices, orthogonal init uses QR decomposition
// For scalars, we return ±1 scaled by gain
// gain: scaling factor (1.0 for linear, sqrt(2) for ReLU)

fn init_orthogonal_scalar(st: RngSt, gain: f64) -> RngResult {
    let r = rng_next(st)
    // Random sign
    if r.value < 0.5 {
        return RngResult { value: gain, rng: r.rng }
    } else {
        return RngResult { value: 0.0 - gain, rng: r.rng }
    }
}

// Orthogonal gain for different activations
fn orthogonal_gain_linear() -> f64 { return 1.0 }
fn orthogonal_gain_relu() -> f64 { return 1.4142135623730951 }  // sqrt(2)
fn orthogonal_gain_tanh() -> f64 { return 1.6666666666666667 }  // 5/3
fn orthogonal_gain_sigmoid() -> f64 { return 1.0 }

// ----------------------------------------------------------------------------
// CONVENIENCE FUNCTIONS (recommended defaults)
// ----------------------------------------------------------------------------

// Best for ReLU networks (most common)
fn init_default_relu(st: RngSt, fan_in: f64) -> RngResult {
    return init_he_normal(st, fan_in)
}

// Best for tanh/sigmoid networks
fn init_default_tanh(st: RngSt, fan_in: f64, fan_out: f64) -> RngResult {
    return init_xavier_normal(st, fan_in, fan_out)
}

// Best for SELU networks
fn init_default_selu(st: RngSt, fan_in: f64) -> RngResult {
    return init_lecun_normal(st, fan_in)
}

// Best for transformers (scaled normal)
fn init_default_transformer(st: RngSt, d_model: f64) -> RngResult {
    let std = 1.0 / sqrt_f64(d_model)
    return rng_normal(st, 0.0, std)
}

// Best for embeddings
fn init_default_embedding(st: RngSt) -> RngResult {
    return rng_normal(st, 0.0, 1.0)
}

// Best for biases (usually zeros or small constant)
fn init_default_bias() -> f64 {
    return 0.0
}

// Small constant bias (sometimes better for ReLU)
fn init_small_bias() -> f64 {
    return 0.01
}

// ============================================================================
// BATCH NORMALIZATION
// ============================================================================
// Normalizes activations to have zero mean and unit variance
// Then applies learnable scale (gamma) and shift (beta)
//
// Training: uses batch statistics, updates running mean/var
// Inference: uses running statistics
//
// Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta

// Batch normalization state
struct BatchNormState {
    gamma: f64,         // Scale parameter (learnable)
    beta: f64,          // Shift parameter (learnable)
    running_mean: f64,  // Running mean for inference
    running_var: f64,   // Running variance for inference
    momentum: f64,      // Momentum for running stats update (typically 0.1)
    eps: f64            // Epsilon for numerical stability
}

// Result of batch norm forward pass
struct BatchNormResult {
    output: f64,        // Normalized, scaled, shifted output
    mean: f64,          // Batch mean (for backward pass)
    variance: f64,      // Batch variance (for backward pass)
    x_norm: f64,        // Normalized input (for backward pass)
    bn_state: BatchNormState  // Updated state
}

// Create initial batch norm state
fn batchnorm_init(gamma: f64, beta: f64, momentum: f64, eps: f64) -> BatchNormState {
    return BatchNormState {
        gamma: gamma,
        beta: beta,
        running_mean: 0.0,
        running_var: 1.0,
        momentum: momentum,
        eps: eps
    }
}

// Default batch norm initialization
fn batchnorm_default() -> BatchNormState {
    return batchnorm_init(1.0, 0.0, 0.1, 0.00001)
}

// Batch norm forward pass for a single value (training mode)
// In practice, you'd compute mean/var over a batch; here we take them as inputs
fn batchnorm_forward_train(x: f64, batch_mean: f64, batch_var: f64, st: BatchNormState) -> BatchNormResult {
    // Normalize
    let x_norm = (x - batch_mean) / sqrt_f64(batch_var + st.eps)

    // Scale and shift
    let output = st.gamma * x_norm + st.beta

    // Update running statistics
    let new_running_mean = (1.0 - st.momentum) * st.running_mean + st.momentum * batch_mean
    let new_running_var = (1.0 - st.momentum) * st.running_var + st.momentum * batch_var

    let new_st = BatchNormState {
        gamma: st.gamma,
        beta: st.beta,
        running_mean: new_running_mean,
        running_var: new_running_var,
        momentum: st.momentum,
        eps: st.eps
    }

    return BatchNormResult {
        output: output,
        mean: batch_mean,
        variance: batch_var,
        x_norm: x_norm,
        bn_state: new_st
    }
}

// Batch norm forward pass (inference mode)
// Uses stored running statistics
fn batchnorm_forward_inference(x: f64, st: BatchNormState) -> f64 {
    let x_norm = (x - st.running_mean) / sqrt_f64(st.running_var + st.eps)
    return st.gamma * x_norm + st.beta
}

// Batch norm backward pass
// Returns gradients for gamma, beta, and input
struct BatchNormGrads {
    dx: f64,        // Gradient w.r.t. input
    dgamma: f64,    // Gradient w.r.t. gamma
    dbeta: f64      // Gradient w.r.t. beta
}

fn batchnorm_backward(dout: f64, x_norm: f64, gamma: f64) -> BatchNormGrads {
    // d_beta = sum(dout) - for single value, just dout
    let dbeta = dout

    // d_gamma = sum(dout * x_norm)
    let dgamma = dout * x_norm

    // d_x_norm = dout * gamma
    let dx_norm = dout * gamma

    // For single value, dx ≈ dx_norm / sqrt(var + eps)
    // Full formula involves batch size, which we don't have for scalar
    let dx = dx_norm

    return BatchNormGrads {
        dx: dx,
        dgamma: dgamma,
        dbeta: dbeta
    }
}

// Compute batch statistics (mean and variance) from array of values
// For a batch of N values: mean = sum(x)/N, var = sum((x-mean)^2)/N
struct BatchStats {
    mean: f64,
    variance: f64
}

fn compute_batch_stats_2(x1: f64, x2: f64) -> BatchStats {
    let mean = (x1 + x2) / 2.0
    let diff1 = x1 - mean
    let diff2 = x2 - mean
    let v = (diff1 * diff1 + diff2 * diff2) / 2.0
    return BatchStats { mean: mean, variance: v }
}

fn compute_batch_stats_3(x1: f64, x2: f64, x3: f64) -> BatchStats {
    let mean = (x1 + x2 + x3) / 3.0
    let d1 = x1 - mean
    let d2 = x2 - mean
    let d3 = x3 - mean
    let v = (d1*d1 + d2*d2 + d3*d3) / 3.0
    return BatchStats { mean: mean, variance: v }
}

fn compute_batch_stats_4(x1: f64, x2: f64, x3: f64, x4: f64) -> BatchStats {
    let mean = (x1 + x2 + x3 + x4) / 4.0
    let d1 = x1 - mean
    let d2 = x2 - mean
    let d3 = x3 - mean
    let d4 = x4 - mean
    let v = (d1*d1 + d2*d2 + d3*d3 + d4*d4) / 4.0
    return BatchStats { mean: mean, variance: v }
}

// ============================================================================
// LAYER NORMALIZATION
// ============================================================================
// Normalizes across features (not batch) - commonly used in transformers
// Unlike batch norm, doesn't need running statistics for inference
//
// Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
// where mean/var are computed across features for each sample

struct LayerNormState {
    gamma: f64,  // Scale parameter
    beta: f64,   // Shift parameter
    eps: f64     // Epsilon for numerical stability
}

struct LayerNormResult {
    output: f64,
    x_norm: f64
}

fn layernorm_init(gamma: f64, beta: f64, eps: f64) -> LayerNormState {
    return LayerNormState { gamma: gamma, beta: beta, eps: eps }
}

fn layernorm_default() -> LayerNormState {
    return layernorm_init(1.0, 0.0, 0.00001)
}

// Layer norm forward for single value with precomputed stats
fn layernorm_forward(x: f64, feature_mean: f64, feature_var: f64, st: LayerNormState) -> LayerNormResult {
    let x_norm = (x - feature_mean) / sqrt_f64(feature_var + st.eps)
    let output = st.gamma * x_norm + st.beta
    return LayerNormResult { output: output, x_norm: x_norm }
}

// Layer norm backward (same structure as batch norm)
fn layernorm_backward(dout: f64, x_norm: f64, gamma: f64) -> BatchNormGrads {
    let dbeta = dout
    let dgamma = dout * x_norm
    let dx = dout * gamma
    return BatchNormGrads { dx: dx, dgamma: dgamma, dbeta: dbeta }
}

// ============================================================================
// DROPOUT
// ============================================================================
// Randomly zeros out activations during training for regularization
// During inference, all activations are used (no dropout)
//
// Training: output = x * mask / (1 - p) where mask is Bernoulli(1-p)
// Inference: output = x (no scaling needed due to inverted dropout)

struct DropoutResult {
    output: f64,
    mask: f64,    // 1.0 if kept, 0.0 if dropped (for backward pass)
    rng: RngSt    // Updated RNG state
}

// Dropout forward (training mode)
// p = dropout probability (fraction of inputs to drop, typically 0.1-0.5)
fn dropout_forward_train(x: f64, p: f64, rng: RngSt) -> DropoutResult {
    if p <= 0.0 {
        // No dropout
        return DropoutResult { output: x, mask: 1.0, rng: rng }
    }
    if p >= 1.0 {
        // Drop everything
        return DropoutResult { output: 0.0, mask: 0.0, rng: rng }
    }

    let r = rng_next(rng)

    if r.value < p {
        // Drop this activation
        return DropoutResult { output: 0.0, mask: 0.0, rng: r.rng }
    } else {
        // Keep and scale by 1/(1-p) for inverted dropout
        let scale = 1.0 / (1.0 - p)
        return DropoutResult { output: x * scale, mask: scale, rng: r.rng }
    }
}

// Dropout forward (inference mode) - just pass through
fn dropout_forward_inference(x: f64) -> f64 {
    return x
}

// Dropout backward
// Gradient is scaled by the same mask used in forward pass
fn dropout_backward(dout: f64, mask: f64) -> f64 {
    return dout * mask
}

// Apply dropout to multiple values
struct Dropout2Result {
    out1: f64,
    out2: f64,
    mask1: f64,
    mask2: f64,
    rng: RngSt
}

fn dropout_forward_2(x1: f64, x2: f64, p: f64, rng: RngSt) -> Dropout2Result {
    let r1 = dropout_forward_train(x1, p, rng)
    let r2 = dropout_forward_train(x2, p, r1.rng)
    return Dropout2Result {
        out1: r1.output,
        out2: r2.output,
        mask1: r1.mask,
        mask2: r2.mask,
        rng: r2.rng
    }
}

struct Dropout3Result {
    out1: f64,
    out2: f64,
    out3: f64,
    mask1: f64,
    mask2: f64,
    mask3: f64,
    rng: RngSt
}

fn dropout_forward_3(x1: f64, x2: f64, x3: f64, p: f64, rng: RngSt) -> Dropout3Result {
    let r1 = dropout_forward_train(x1, p, rng)
    let r2 = dropout_forward_train(x2, p, r1.rng)
    let r3 = dropout_forward_train(x3, p, r2.rng)
    return Dropout3Result {
        out1: r1.output,
        out2: r2.output,
        out3: r3.output,
        mask1: r1.mask,
        mask2: r2.mask,
        mask3: r3.mask,
        rng: r3.rng
    }
}

// ============================================================================
// ALPHA DROPOUT (for SELU networks)
// ============================================================================
// Special dropout for Self-Normalizing Neural Networks (SNNs)
// Maintains self-normalizing property by using specific alpha and scale values

fn ALPHA_DROPOUT_ALPHA() -> f64 { return 1.6732632423543772 }
fn ALPHA_DROPOUT_SCALE() -> f64 { return 1.0507009873554805 }

fn alpha_dropout_forward_train(x: f64, p: f64, rng: RngSt) -> DropoutResult {
    if p <= 0.0 {
        return DropoutResult { output: x, mask: 1.0, rng: rng }
    }
    if p >= 1.0 {
        let alpha = ALPHA_DROPOUT_ALPHA()
        return DropoutResult { output: 0.0 - alpha, mask: 0.0, rng: rng }
    }

    let r = rng_next(rng)
    let alpha = ALPHA_DROPOUT_ALPHA()
    let scale = ALPHA_DROPOUT_SCALE()

    // Compute affine transformation parameters to maintain mean and variance
    let a = 1.0 / sqrt_f64((1.0 - p) * (1.0 + p * alpha * alpha))

    if r.value < p {
        // Set to -alpha * scale (not zero)
        let output = a * (0.0 - alpha)
        return DropoutResult { output: output, mask: 0.0, rng: r.rng }
    } else {
        let output = a * x
        return DropoutResult { output: output, mask: a, rng: r.rng }
    }
}

// ============================================================================
// SPATIAL DROPOUT (for convolutional networks)
// ============================================================================
// Drops entire feature channels instead of individual activations
// Simulated here by using the same mask for a group of values

struct SpatialDropoutResult {
    outputs: f64,  // Same scaling for entire channel
    channel_mask: f64,
    rng: RngSt
}

fn spatial_dropout_channel(x: f64, p: f64, channel_mask: f64) -> f64 {
    // Use precomputed channel mask
    if channel_mask == 0.0 {
        return 0.0
    }
    return x * channel_mask
}

// Generate channel mask (call once per channel, use for all spatial positions)
fn spatial_dropout_get_mask(p: f64, rng: RngSt) -> DropoutResult {
    let r = rng_next(rng)
    if r.value < p {
        return DropoutResult { output: 0.0, mask: 0.0, rng: r.rng }
    } else {
        let scale = 1.0 / (1.0 - p)
        return DropoutResult { output: scale, mask: scale, rng: r.rng }
    }
}

// ============================================================================
// DROPCONNECT (drops weights instead of activations)
// ============================================================================
// For a weight connecting input x to output: y = w * x
// DropConnect randomly zeros weights during training

struct DropConnectResult {
    output: f64,
    weight_mask: f64,
    rng: RngSt
}

fn dropconnect_forward(x: f64, w: f64, p: f64, rng: RngSt) -> DropConnectResult {
    let r = rng_next(rng)

    if r.value < p {
        // Drop this connection
        return DropConnectResult { output: 0.0, weight_mask: 0.0, rng: r.rng }
    } else {
        // Keep connection with scaling
        let scale = 1.0 / (1.0 - p)
        return DropConnectResult { output: w * x * scale, weight_mask: scale, rng: r.rng }
    }
}

// ============================================================================
// GROUP NORMALIZATION
// ============================================================================
// Divides channels into groups and normalizes within each group
// Works well with small batch sizes (unlike batch norm)

struct GroupNormState {
    gamma: f64,
    beta: f64,
    num_groups: f64,
    eps: f64
}

fn groupnorm_init(gamma: f64, beta: f64, num_groups: f64, eps: f64) -> GroupNormState {
    return GroupNormState { gamma: gamma, beta: beta, num_groups: num_groups, eps: eps }
}

fn groupnorm_default(num_groups: f64) -> GroupNormState {
    return groupnorm_init(1.0, 0.0, num_groups, 0.00001)
}

// Group norm forward for a value with precomputed group statistics
fn groupnorm_forward(x: f64, group_mean: f64, group_var: f64, st: GroupNormState) -> LayerNormResult {
    let x_norm = (x - group_mean) / sqrt_f64(group_var + st.eps)
    let output = st.gamma * x_norm + st.beta
    return LayerNormResult { output: output, x_norm: x_norm }
}

// ============================================================================
// INSTANCE NORMALIZATION
// ============================================================================
// Normalizes each sample independently (commonly used in style transfer)
// Like batch norm but with batch_size=1

struct InstanceNormState {
    gamma: f64,
    beta: f64,
    eps: f64
}

fn instancenorm_init(gamma: f64, beta: f64, eps: f64) -> InstanceNormState {
    return InstanceNormState { gamma: gamma, beta: beta, eps: eps }
}

fn instancenorm_default() -> InstanceNormState {
    return instancenorm_init(1.0, 0.0, 0.00001)
}

fn instancenorm_forward(x: f64, instance_mean: f64, instance_var: f64, st: InstanceNormState) -> LayerNormResult {
    let x_norm = (x - instance_mean) / sqrt_f64(instance_var + st.eps)
    let output = st.gamma * x_norm + st.beta
    return LayerNormResult { output: output, x_norm: x_norm }
}

// ============================================================================
// RMS NORMALIZATION (Root Mean Square Layer Normalization)
// ============================================================================
// Simplified layer norm without mean centering - used in LLaMA, etc.
// Formula: y = gamma * x / RMS(x) where RMS(x) = sqrt(mean(x^2))

struct RMSNormState {
    gamma: f64,
    eps: f64
}

fn rmsnorm_init(gamma: f64, eps: f64) -> RMSNormState {
    return RMSNormState { gamma: gamma, eps: eps }
}

fn rmsnorm_default() -> RMSNormState {
    return rmsnorm_init(1.0, 0.00001)
}

// RMS norm forward with precomputed RMS value
fn rmsnorm_forward(x: f64, rms: f64, st: RMSNormState) -> f64 {
    return st.gamma * x / (rms + st.eps)
}

// Compute RMS for 2 values
fn compute_rms_2(x1: f64, x2: f64) -> f64 {
    return sqrt_f64((x1*x1 + x2*x2) / 2.0)
}

// Compute RMS for 3 values
fn compute_rms_3(x1: f64, x2: f64, x3: f64) -> f64 {
    return sqrt_f64((x1*x1 + x2*x2 + x3*x3) / 3.0)
}

// Compute RMS for 4 values
fn compute_rms_4(x1: f64, x2: f64, x3: f64, x4: f64) -> f64 {
    return sqrt_f64((x1*x1 + x2*x2 + x3*x3 + x4*x4) / 4.0)
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

    // Test 35: RAdam - verify variance rectification behavior
    println("Test 35: RAdam variance rectification")
    let x35 = 5.0
    let m35 = 0.0
    let v35 = 0.0
    let lr35 = 0.1
    let g35 = 2.0 * x35  // gradient = 10.0

    // At timestep 1, RAdam should use unadapted update (ρ_t < 5)
    // At later timesteps, it should switch to adaptive update
    let radam_t1 = radam_step(x35, g35, m35, v35, 1.0, lr35)
    let radam_t5 = radam_step(x35, g35, m35, v35, 5.0, lr35)

    // Compute ρ values to verify behavior
    let beta2 = 0.999
    let rho_inf = 2.0 / (1.0 - beta2) - 1.0  // ≈ 1999
    let beta2_t1 = pow_f64(beta2, 1.0)
    let beta2_t5 = pow_f64(beta2, 5.0)
    let rho_t1 = rho_inf - 2.0 * 1.0 * beta2_t1 / (1.0 - beta2_t1)
    let rho_t5 = rho_inf - 2.0 * 5.0 * beta2_t5 / (1.0 - beta2_t5)

    println("  ρ_inf (max SMA length) = ")
    println(rho_inf)
    println("  ρ at t=1 = ")
    println(rho_t1)
    println("  ρ at t=5 = ")
    println(rho_t5)
    println("  RAdam at t=1 (unadapted if ρ<5):")
    println("    param = ")
    println(radam_t1.param)
    println("  RAdam at t=5 (adaptive if ρ>5):")
    println("    param = ")
    println(radam_t5.param)

    // Both should decrease from initial
    if radam_t1.param >= x35 { ok = false; println("  FAIL: RAdam t=1 should decrease") }
    if radam_t5.param >= x35 { ok = false; println("  FAIL: RAdam t=5 should decrease") }
    // ρ_inf should be approximately 1999 for β2=0.999
    if abs_f64(rho_inf - 1999.0) > 1.0 { ok = false; println("  FAIL: rho_inf should be ~1999") }
    println("")

    // Test 36: RAdam 5-step descent (unrolled)
    println("Test 36: RAdam 5-step descent (unrolled)")
    let r0_36 = 5.0
    let mr0_36 = 0.0
    let vr0_36 = 0.0
    let lr36 = 0.1

    // Step 1
    let gr1 = 2.0 * r0_36
    let rd1 = radam_step(r0_36, gr1, mr0_36, vr0_36, 1.0, lr36)
    let r1_36 = rd1.param
    let mr1_36 = rd1.m
    let vr1_36 = rd1.v

    // Step 2
    let gr2 = 2.0 * r1_36
    let rd2 = radam_step(r1_36, gr2, mr1_36, vr1_36, 2.0, lr36)
    let r2_36 = rd2.param
    let mr2_36 = rd2.m
    let vr2_36 = rd2.v

    // Step 3
    let gr3 = 2.0 * r2_36
    let rd3 = radam_step(r2_36, gr3, mr2_36, vr2_36, 3.0, lr36)
    let r3_36 = rd3.param
    let mr3_36 = rd3.m
    let vr3_36 = rd3.v

    // Step 4
    let gr4 = 2.0 * r3_36
    let rd4 = radam_step(r3_36, gr4, mr3_36, vr3_36, 4.0, lr36)
    let r4_36 = rd4.param
    let mr4_36 = rd4.m
    let vr4_36 = rd4.v

    // Step 5
    let gr5 = 2.0 * r4_36
    let rd5 = radam_step(r4_36, gr5, mr4_36, vr4_36, 5.0, lr36)
    let r5_36 = rd5.param

    println("  Descent from r=5 with RAdam (rectified variance):")
    println("    r0 = 5.0")
    println("    r1 = ")
    println(r1_36)
    println("    r2 = ")
    println(r2_36)
    println("    r3 = ")
    println(r3_36)
    println("    r4 = ")
    println(r4_36)
    println("    r5 = ")
    println(r5_36)

    // r should decrease toward 0
    if r1_36 >= r0_36 { ok = false; println("  FAIL: r1 >= r0") }
    if r2_36 >= r1_36 { ok = false; println("  FAIL: r2 >= r1") }
    if r5_36 >= 4.5 { ok = false; println("  FAIL: r5 should be < 4.5 after 5 steps") }
    println("")

    // Test 37: LAMB - verify trust ratio behavior
    println("Test 37: LAMB trust ratio computation")
    let x37 = 5.0
    let m37 = 0.0
    let v37 = 0.0
    let lr37 = 0.1
    let wd37 = 0.01
    let g37 = 2.0 * x37  // gradient = 10.0

    // Compare AdamW vs LAMB at timestep 1
    let adamw_result37 = adamw_step(x37, g37, m37, v37, 1.0, lr37, wd37)
    let lamb_result37 = lamb_step(x37, g37, m37, v37, 1.0, lr37, wd37)

    // Compute expected trust ratio manually
    // adam_update = m_hat / (sqrt(v_hat) + eps) + wd * param
    let beta1 = 0.9
    let beta2 = 0.999
    let eps37 = 0.000001
    let m_hat37 = (beta1 * m37 + (1.0 - beta1) * g37) / (1.0 - beta1)  // = g37
    let v_hat37 = (beta2 * v37 + (1.0 - beta2) * g37 * g37) / (1.0 - beta2)  // = g37^2
    let adam_update37 = m_hat37 / (sqrt_f64(v_hat37) + eps37) + wd37 * x37
    let param_norm37 = abs_f64(x37)
    let update_norm37 = abs_f64(adam_update37)
    let trust_ratio37 = param_norm37 / update_norm37

    println("  AdamW result:")
    println("    param = ")
    println(adamw_result37.param)
    println("  LAMB result (with trust ratio):")
    println("    param = ")
    println(lamb_result37.param)
    println("  Trust ratio = ||param|| / ||update|| = ")
    println(trust_ratio37)
    println("  param_norm = ")
    println(param_norm37)
    println("  update_norm = ")
    println(update_norm37)

    // Both should decrease from initial
    if lamb_result37.param >= x37 { ok = false; println("  FAIL: LAMB should decrease") }
    if adamw_result37.param >= x37 { ok = false; println("  FAIL: AdamW should decrease") }
    // Trust ratio should be positive and reasonable
    if trust_ratio37 <= 0.0 { ok = false; println("  FAIL: trust ratio should be > 0") }
    if trust_ratio37 > 10.0 { ok = false; println("  FAIL: trust ratio should be clamped to 10") }
    // Moments should be the same
    if abs_f64(lamb_result37.m - adamw_result37.m) > tol { ok = false; println("  FAIL: m should match") }
    if abs_f64(lamb_result37.v - adamw_result37.v) > tol { ok = false; println("  FAIL: v should match") }
    println("")

    // Test 38: LAMB 5-step descent (unrolled)
    println("Test 38: LAMB 5-step descent (unrolled)")
    let l0_38 = 5.0
    let ml0_38 = 0.0
    let vl0_38 = 0.0
    let lr38 = 0.1
    let wd38 = 0.01

    // Step 1
    let gl1 = 2.0 * l0_38
    let lb1 = lamb_step(l0_38, gl1, ml0_38, vl0_38, 1.0, lr38, wd38)
    let l1_38 = lb1.param
    let ml1_38 = lb1.m
    let vl1_38 = lb1.v

    // Step 2
    let gl2 = 2.0 * l1_38
    let lb2 = lamb_step(l1_38, gl2, ml1_38, vl1_38, 2.0, lr38, wd38)
    let l2_38 = lb2.param
    let ml2_38 = lb2.m
    let vl2_38 = lb2.v

    // Step 3
    let gl3 = 2.0 * l2_38
    let lb3 = lamb_step(l2_38, gl3, ml2_38, vl2_38, 3.0, lr38, wd38)
    let l3_38 = lb3.param
    let ml3_38 = lb3.m
    let vl3_38 = lb3.v

    // Step 4
    let gl4 = 2.0 * l3_38
    let lb4 = lamb_step(l3_38, gl4, ml3_38, vl3_38, 4.0, lr38, wd38)
    let l4_38 = lb4.param
    let ml4_38 = lb4.m
    let vl4_38 = lb4.v

    // Step 5
    let gl5 = 2.0 * l4_38
    let lb5 = lamb_step(l4_38, gl5, ml4_38, vl4_38, 5.0, lr38, wd38)
    let l5_38 = lb5.param

    println("  Descent from l=5 with LAMB (large batch optimizer):")
    println("    l0 = 5.0")
    println("    l1 = ")
    println(l1_38)
    println("    l2 = ")
    println(l2_38)
    println("    l3 = ")
    println(l3_38)
    println("    l4 = ")
    println(l4_38)
    println("    l5 = ")
    println(l5_38)

    // l should decrease toward 0
    if l1_38 >= l0_38 { ok = false; println("  FAIL: l1 >= l0") }
    if l2_38 >= l1_38 { ok = false; println("  FAIL: l2 >= l1") }
    if l5_38 >= 4.5 { ok = false; println("  FAIL: l5 should be < 4.5 after 5 steps") }
    println("")

    // Test 39: Lion - verify sign-based update
    println("Test 39: Lion sign-based update")
    let x39 = 5.0
    let m39 = 0.0
    let lr39 = 0.1
    let wd39 = 0.01
    let g39 = 2.0 * x39  // gradient = 10.0

    // Lion uses sign of interpolated momentum
    // interpolated = β1 * m + (1 - β1) * g = 0.9 * 0 + 0.1 * 10 = 1.0
    // sign(1.0) = 1.0
    // update = 1.0, so param moves by -lr * 1 = -0.1
    let lion_result39 = lion_step(x39, g39, m39, lr39, wd39)

    // Expected: param = 5 - 0.1 * 1 - 0.1 * 0.01 * 5 = 5 - 0.1 - 0.005 = 4.895
    let expected_param39 = x39 - lr39 * 1.0 - lr39 * wd39 * x39

    println("  Lion result:")
    println("    param = ")
    println(lion_result39.param)
    println("    m = ")
    println(lion_result39.m)
    println("  Expected param = ")
    println(expected_param39)
    println("  Sign of interpolated momentum = 1.0 (positive gradient)")

    // Verify sign-based update
    if abs_f64(lion_result39.param - expected_param39) > tol { ok = false; println("  FAIL: param mismatch") }
    // Momentum should be updated
    let expected_m39 = 0.99 * m39 + 0.01 * g39  // β2 * m + (1-β2) * g
    if abs_f64(lion_result39.m - expected_m39) > tol { ok = false; println("  FAIL: m mismatch") }
    // Should decrease from initial
    if lion_result39.param >= x39 { ok = false; println("  FAIL: Lion should decrease") }
    println("")

    // Test 40: Lion 5-step descent (unrolled)
    println("Test 40: Lion 5-step descent (unrolled)")
    let li0_40 = 5.0
    let mli0_40 = 0.0
    let lr40 = 0.5  // Lion often uses larger lr since updates are ±1
    let wd40 = 0.0  // No weight decay for cleaner test

    // Step 1
    let gli1 = 2.0 * li0_40
    let lio1 = lion_step_no_wd(li0_40, gli1, mli0_40, lr40)
    let li1_40 = lio1.param
    let mli1_40 = lio1.m

    // Step 2
    let gli2 = 2.0 * li1_40
    let lio2 = lion_step_no_wd(li1_40, gli2, mli1_40, lr40)
    let li2_40 = lio2.param
    let mli2_40 = lio2.m

    // Step 3
    let gli3 = 2.0 * li2_40
    let lio3 = lion_step_no_wd(li2_40, gli3, mli2_40, lr40)
    let li3_40 = lio3.param
    let mli3_40 = lio3.m

    // Step 4
    let gli4 = 2.0 * li3_40
    let lio4 = lion_step_no_wd(li3_40, gli4, mli3_40, lr40)
    let li4_40 = lio4.param
    let mli4_40 = lio4.m

    // Step 5
    let gli5 = 2.0 * li4_40
    let lio5 = lion_step_no_wd(li4_40, gli5, mli4_40, lr40)
    let li5_40 = lio5.param

    println("  Descent from li=5 with Lion (sign momentum, lr=0.5):")
    println("    li0 = 5.0")
    println("    li1 = ")
    println(li1_40)
    println("    li2 = ")
    println(li2_40)
    println("    li3 = ")
    println(li3_40)
    println("    li4 = ")
    println(li4_40)
    println("    li5 = ")
    println(li5_40)

    // li should decrease toward 0 (uniform steps of ±lr due to sign)
    if li1_40 >= li0_40 { ok = false; println("  FAIL: li1 >= li0") }
    if li2_40 >= li1_40 { ok = false; println("  FAIL: li2 >= li1") }
    if li5_40 >= 3.0 { ok = false; println("  FAIL: li5 should be < 3 after 5 steps") }
    println("")

    // ========================================================================
    // LEARNING RATE SCHEDULER TESTS
    // ========================================================================

    // Test 41: Cosine annealing scheduler
    println("Test 41: Cosine annealing scheduler")
    let lr_init41 = 0.1
    let lr_min41 = 0.001
    let total_steps41 = 100.0

    // At step 0, should be at initial_lr
    let lr_s0 = lr_cosine_annealing(lr_init41, lr_min41, 0.0, total_steps41)
    println("  lr at step 0 = ")
    println(lr_s0)

    // At step 50 (midpoint), should be halfway between
    let lr_s50 = lr_cosine_annealing(lr_init41, lr_min41, 50.0, total_steps41)
    println("  lr at step 50 = ")
    println(lr_s50)

    // At step 100, should be at min_lr
    let lr_s100 = lr_cosine_annealing(lr_init41, lr_min41, 100.0, total_steps41)
    println("  lr at step 100 = ")
    println(lr_s100)

    // Verify: start high, end low, midpoint in between
    if abs_f64(lr_s0 - lr_init41) > tol { ok = false; println("  FAIL: s0 should be init_lr") }
    if abs_f64(lr_s100 - lr_min41) > tol { ok = false; println("  FAIL: s100 should be min_lr") }
    // Midpoint of cosine: min + 0.5 * (init - min) * (1 + cos(π/2)) = min + 0.5*(init-min)
    let expected_mid = lr_min41 + 0.5 * (lr_init41 - lr_min41)
    if abs_f64(lr_s50 - expected_mid) > 0.01 { ok = false; println("  FAIL: s50 should be midpoint") }
    if lr_s0 <= lr_s100 { ok = false; println("  FAIL: start should be > end") }
    println("")

    // Test 42: Linear warmup scheduler
    println("Test 42: Linear warmup scheduler")
    let lr_init42 = 0.01
    let warmup_steps42 = 10.0

    // At step 0, lr should be 0
    let lr_w0 = lr_linear_warmup(lr_init42, 0.0, warmup_steps42)
    println("  lr at step 0 = ")
    println(lr_w0)

    // At step 5 (halfway), lr should be 0.5 * initial
    let lr_w5 = lr_linear_warmup(lr_init42, 5.0, warmup_steps42)
    println("  lr at step 5 = ")
    println(lr_w5)

    // At step 10 (end of warmup), lr should be initial
    let lr_w10 = lr_linear_warmup(lr_init42, 10.0, warmup_steps42)
    println("  lr at step 10 = ")
    println(lr_w10)

    // After warmup, lr stays constant
    let lr_w20 = lr_linear_warmup(lr_init42, 20.0, warmup_steps42)
    println("  lr at step 20 = ")
    println(lr_w20)

    // Verify warmup behavior
    if abs_f64(lr_w0 - 0.0) > tol { ok = false; println("  FAIL: w0 should be 0") }
    if abs_f64(lr_w5 - 0.005) > tol { ok = false; println("  FAIL: w5 should be 0.005") }
    if abs_f64(lr_w10 - lr_init42) > tol { ok = false; println("  FAIL: w10 should be init") }
    if abs_f64(lr_w20 - lr_init42) > tol { ok = false; println("  FAIL: w20 should be init") }
    if lr_w0 >= lr_w5 { ok = false; println("  FAIL: should increase during warmup") }
    if lr_w5 >= lr_w10 { ok = false; println("  FAIL: should increase during warmup") }
    println("")

    // Test 43: One cycle policy scheduler
    println("Test 43: One cycle policy scheduler")
    let lr_init43 = 0.001
    let lr_max43 = 0.01
    let total_steps43 = 100.0
    let pct_start43 = 0.3  // 30% increasing, 70% decreasing

    // At step 0, should be at initial_lr
    let lr_oc0 = lr_one_cycle(lr_init43, lr_max43, 0.0, total_steps43, pct_start43)
    println("  lr at step 0 = ")
    println(lr_oc0)

    // At step 30 (peak), should be at max_lr
    let lr_oc30 = lr_one_cycle(lr_init43, lr_max43, 30.0, total_steps43, pct_start43)
    println("  lr at step 30 (peak) = ")
    println(lr_oc30)

    // At step 65 (midway through decay), should be decreasing
    let lr_oc65 = lr_one_cycle(lr_init43, lr_max43, 65.0, total_steps43, pct_start43)
    println("  lr at step 65 = ")
    println(lr_oc65)

    // At step 100, should be near 0
    let lr_oc100 = lr_one_cycle(lr_init43, lr_max43, 100.0, total_steps43, pct_start43)
    println("  lr at step 100 = ")
    println(lr_oc100)

    // Verify one cycle: start low -> peak at pct_start -> decay to ~0
    if abs_f64(lr_oc0 - lr_init43) > tol { ok = false; println("  FAIL: oc0 should be init") }
    if abs_f64(lr_oc30 - lr_max43) > 0.001 { ok = false; println("  FAIL: oc30 should be max") }
    if lr_oc30 <= lr_oc0 { ok = false; println("  FAIL: peak should be > start") }
    if lr_oc65 >= lr_oc30 { ok = false; println("  FAIL: decay phase should decrease from peak") }
    if lr_oc100 >= lr_oc65 { ok = false; println("  FAIL: should continue decreasing") }
    println("")

    // Test 44: Step decay scheduler
    println("Test 44: Step decay scheduler")
    let lr_init44 = 0.1
    let step_size44 = 10.0
    let gamma44 = 0.5

    // At step 0, lr = 0.1
    let lr_sd0 = lr_step_decay(lr_init44, 0.0, step_size44, gamma44)
    println("  lr at step 0 = ")
    println(lr_sd0)

    // At step 5, still lr = 0.1 (no decay yet)
    let lr_sd5 = lr_step_decay(lr_init44, 5.0, step_size44, gamma44)
    println("  lr at step 5 = ")
    println(lr_sd5)

    // At step 10, lr = 0.1 * 0.5 = 0.05
    let lr_sd10 = lr_step_decay(lr_init44, 10.0, step_size44, gamma44)
    println("  lr at step 10 = ")
    println(lr_sd10)

    // At step 20, lr = 0.1 * 0.5^2 = 0.025
    let lr_sd20 = lr_step_decay(lr_init44, 20.0, step_size44, gamma44)
    println("  lr at step 20 = ")
    println(lr_sd20)

    // At step 30, lr = 0.1 * 0.5^3 = 0.0125
    let lr_sd30 = lr_step_decay(lr_init44, 30.0, step_size44, gamma44)
    println("  lr at step 30 = ")
    println(lr_sd30)

    // Verify step decay
    if abs_f64(lr_sd0 - 0.1) > tol { ok = false; println("  FAIL: sd0 should be 0.1") }
    if abs_f64(lr_sd5 - 0.1) > tol { ok = false; println("  FAIL: sd5 should be 0.1") }
    if abs_f64(lr_sd10 - 0.05) > tol { ok = false; println("  FAIL: sd10 should be 0.05") }
    if abs_f64(lr_sd20 - 0.025) > tol { ok = false; println("  FAIL: sd20 should be 0.025") }
    if abs_f64(lr_sd30 - 0.0125) > tol { ok = false; println("  FAIL: sd30 should be 0.0125") }
    println("")

    // Test 45: Exponential decay scheduler
    println("Test 45: Exponential decay scheduler")
    let lr_init45 = 0.1
    let decay_rate45 = 0.95

    // lr = initial * decay^step
    let lr_exp0 = lr_exponential_decay(lr_init45, 0.0, decay_rate45)
    let lr_exp10 = lr_exponential_decay(lr_init45, 10.0, decay_rate45)
    let lr_exp50 = lr_exponential_decay(lr_init45, 50.0, decay_rate45)

    println("  lr at step 0 = ")
    println(lr_exp0)
    println("  lr at step 10 = ")
    println(lr_exp10)
    println("  lr at step 50 = ")
    println(lr_exp50)

    // Expected: 0.1 * 0.95^10 ≈ 0.0598, 0.1 * 0.95^50 ≈ 0.00769
    let expected_exp10 = lr_init45 * pow_f64(decay_rate45, 10.0)
    let expected_exp50 = lr_init45 * pow_f64(decay_rate45, 50.0)

    if abs_f64(lr_exp0 - 0.1) > tol { ok = false; println("  FAIL: exp0 should be 0.1") }
    if abs_f64(lr_exp10 - expected_exp10) > tol { ok = false; println("  FAIL: exp10 mismatch") }
    if abs_f64(lr_exp50 - expected_exp50) > tol { ok = false; println("  FAIL: exp50 mismatch") }
    if lr_exp0 <= lr_exp10 { ok = false; println("  FAIL: should decrease") }
    if lr_exp10 <= lr_exp50 { ok = false; println("  FAIL: should decrease") }
    println("")

    // Test 46: Warmup + Cosine annealing (Transformer-style)
    println("Test 46: Warmup + Cosine annealing scheduler")
    let lr_init46 = 0.0001
    let lr_min46 = 0.00001
    let warmup_steps46 = 10.0
    let total_steps46 = 100.0

    // During warmup: linear increase
    let lr_wc0 = lr_warmup_cosine(lr_init46, lr_min46, 0.0, warmup_steps46, total_steps46)
    let lr_wc5 = lr_warmup_cosine(lr_init46, lr_min46, 5.0, warmup_steps46, total_steps46)
    let lr_wc10 = lr_warmup_cosine(lr_init46, lr_min46, 10.0, warmup_steps46, total_steps46)

    // After warmup: cosine annealing
    let lr_wc50 = lr_warmup_cosine(lr_init46, lr_min46, 50.0, warmup_steps46, total_steps46)
    let lr_wc100 = lr_warmup_cosine(lr_init46, lr_min46, 100.0, warmup_steps46, total_steps46)

    println("  lr at step 0 (warmup start) = ")
    println(lr_wc0)
    println("  lr at step 5 (warmup mid) = ")
    println(lr_wc5)
    println("  lr at step 10 (warmup end) = ")
    println(lr_wc10)
    println("  lr at step 50 (decay mid) = ")
    println(lr_wc50)
    println("  lr at step 100 (decay end) = ")
    println(lr_wc100)

    // Verify warmup phase
    if abs_f64(lr_wc0 - 0.0) > tol { ok = false; println("  FAIL: wc0 should be ~0") }
    if lr_wc0 >= lr_wc5 { ok = false; println("  FAIL: warmup should increase") }
    if lr_wc5 >= lr_wc10 { ok = false; println("  FAIL: warmup should increase") }
    // Verify decay phase
    if lr_wc10 <= lr_wc50 { ok = false; println("  FAIL: should decay after warmup") }
    if lr_wc50 <= lr_wc100 { ok = false; println("  FAIL: should continue decaying") }
    // End should be near min_lr
    if abs_f64(lr_wc100 - lr_min46) > 0.001 { ok = false; println("  FAIL: wc100 should be near min") }
    println("")

    // Test 47: Cyclic learning rate
    println("Test 47: Cyclic learning rate scheduler")
    let lr_min47 = 0.001
    let lr_max47 = 0.01
    let cycle_len47 = 20.0

    // At step 0, should be at max (cosine = 1)
    let lr_cyc0 = lr_cyclic(lr_min47, lr_max47, 0.0, cycle_len47)
    // At step 5, should be decreasing
    let lr_cyc5 = lr_cyclic(lr_min47, lr_max47, 5.0, cycle_len47)
    // At step 10 (half cycle), should be at min
    let lr_cyc10 = lr_cyclic(lr_min47, lr_max47, 10.0, cycle_len47)
    // At step 20 (full cycle), should be back at max
    let lr_cyc20 = lr_cyclic(lr_min47, lr_max47, 20.0, cycle_len47)
    // At step 30 (1.5 cycles), should be at min again
    let lr_cyc30 = lr_cyclic(lr_min47, lr_max47, 30.0, cycle_len47)

    println("  lr at step 0 = ")
    println(lr_cyc0)
    println("  lr at step 5 = ")
    println(lr_cyc5)
    println("  lr at step 10 (min) = ")
    println(lr_cyc10)
    println("  lr at step 20 (max) = ")
    println(lr_cyc20)
    println("  lr at step 30 (min) = ")
    println(lr_cyc30)

    // Verify cyclic behavior
    if abs_f64(lr_cyc0 - lr_max47) > tol { ok = false; println("  FAIL: cyc0 should be max") }
    if abs_f64(lr_cyc10 - lr_min47) > tol { ok = false; println("  FAIL: cyc10 should be min") }
    if abs_f64(lr_cyc20 - lr_max47) > tol { ok = false; println("  FAIL: cyc20 should be max") }
    if abs_f64(lr_cyc30 - lr_min47) > tol { ok = false; println("  FAIL: cyc30 should be min") }
    if lr_cyc5 >= lr_cyc0 { ok = false; println("  FAIL: should decrease from 0 to 10") }
    if lr_cyc5 <= lr_cyc10 { ok = false; println("  FAIL: should continue decreasing") }
    println("")

    // Test 48: Inverse sqrt scheduler (Transformer-style)
    println("Test 48: Inverse sqrt scheduler")
    let lr_init48 = 0.001
    let warmup_steps48 = 100.0

    // During warmup
    let lr_isq10 = lr_inverse_sqrt(lr_init48, 10.0, warmup_steps48)
    let lr_isq50 = lr_inverse_sqrt(lr_init48, 50.0, warmup_steps48)
    let lr_isq100 = lr_inverse_sqrt(lr_init48, 100.0, warmup_steps48)

    // After warmup: inverse sqrt decay
    let lr_isq200 = lr_inverse_sqrt(lr_init48, 200.0, warmup_steps48)
    let lr_isq400 = lr_inverse_sqrt(lr_init48, 400.0, warmup_steps48)

    println("  lr at step 10 (warmup) = ")
    println(lr_isq10)
    println("  lr at step 50 (warmup) = ")
    println(lr_isq50)
    println("  lr at step 100 (peak) = ")
    println(lr_isq100)
    println("  lr at step 200 (decay) = ")
    println(lr_isq200)
    println("  lr at step 400 (decay) = ")
    println(lr_isq400)

    // Verify warmup phase: linear increase
    if lr_isq10 >= lr_isq50 { ok = false; println("  FAIL: warmup should increase") }
    if lr_isq50 >= lr_isq100 { ok = false; println("  FAIL: warmup should increase") }
    // At warmup end, should be near initial
    if abs_f64(lr_isq100 - lr_init48) > tol { ok = false; println("  FAIL: isq100 should be init") }
    // Verify decay phase: inverse sqrt
    if lr_isq100 <= lr_isq200 { ok = false; println("  FAIL: should decay after warmup") }
    if lr_isq200 <= lr_isq400 { ok = false; println("  FAIL: should continue decaying") }
    // Check inverse sqrt: lr(200) = lr(100) * sqrt(100/200) = 0.001 * sqrt(0.5) ≈ 0.000707
    let expected_isq200 = lr_init48 * sqrt_f64(warmup_steps48) / sqrt_f64(200.0)
    if abs_f64(lr_isq200 - expected_isq200) > tol { ok = false; println("  FAIL: isq200 mismatch") }
    println("")

    // ========================================================================
    // LOSS FUNCTION TESTS
    // ========================================================================

    // Test 49: MSE loss and gradient
    println("Test 49: MSE loss and gradient")
    let pred49 = 3.0
    let target49 = 1.0
    let mse49 = loss_mse(pred49, target49)
    let mse_grad49 = loss_mse_grad(pred49, target49)

    println("  pred=3, target=1")
    println("  MSE = ")
    println(mse49)
    println("  MSE grad = ")
    println(mse_grad49)

    // MSE = (3-1)² = 4, grad = 2*(3-1) = 4
    if abs_f64(mse49 - 4.0) > tol { ok = false; println("  FAIL: MSE should be 4") }
    if abs_f64(mse_grad49 - 4.0) > tol { ok = false; println("  FAIL: MSE grad should be 4") }
    println("")

    // Test 50: MAE loss and gradient
    println("Test 50: MAE loss and gradient")
    let pred50 = 5.0
    let target50 = 2.0
    let mae50 = loss_mae(pred50, target50)
    let mae_grad50 = loss_mae_grad(pred50, target50)

    println("  pred=5, target=2")
    println("  MAE = ")
    println(mae50)
    println("  MAE grad = ")
    println(mae_grad50)

    // MAE = |5-2| = 3, grad = sign(5-2) = 1
    if abs_f64(mae50 - 3.0) > tol { ok = false; println("  FAIL: MAE should be 3") }
    if abs_f64(mae_grad50 - 1.0) > tol { ok = false; println("  FAIL: MAE grad should be 1") }
    println("")

    // Test 51: Huber loss (smooth L1)
    println("Test 51: Huber loss")
    let delta51 = 1.0

    // Small error (quadratic region): pred=1.5, target=1.0, diff=0.5
    let huber_small = loss_huber(1.5, 1.0, delta51)
    let huber_small_grad = loss_huber_grad(1.5, 1.0, delta51)

    // Large error (linear region): pred=5.0, target=1.0, diff=4.0
    let huber_large = loss_huber(5.0, 1.0, delta51)
    let huber_large_grad = loss_huber_grad(5.0, 1.0, delta51)

    println("  Small diff (0.5): loss = ")
    println(huber_small)
    println("  Small diff grad = ")
    println(huber_small_grad)
    println("  Large diff (4.0): loss = ")
    println(huber_large)
    println("  Large diff grad = ")
    println(huber_large_grad)

    // Small: 0.5 * 0.5² = 0.125, grad = 0.5
    if abs_f64(huber_small - 0.125) > tol { ok = false; println("  FAIL: Huber small should be 0.125") }
    if abs_f64(huber_small_grad - 0.5) > tol { ok = false; println("  FAIL: Huber small grad should be 0.5") }
    // Large: 1.0 * (4.0 - 0.5) = 3.5, grad = 1.0 (delta)
    if abs_f64(huber_large - 3.5) > tol { ok = false; println("  FAIL: Huber large should be 3.5") }
    if abs_f64(huber_large_grad - 1.0) > tol { ok = false; println("  FAIL: Huber large grad should be 1.0") }
    println("")

    // Test 52: Binary cross-entropy
    println("Test 52: Binary cross-entropy")
    // BCE for pred=0.8, target=1.0: -log(0.8) ≈ 0.223
    let bce52 = loss_bce(0.8, 1.0)
    let bce_grad52 = loss_bce_grad(0.8, 1.0)

    println("  pred=0.8, target=1.0")
    println("  BCE = ")
    println(bce52)
    println("  BCE grad = ")
    println(bce_grad52)

    // -log(0.8) ≈ 0.223
    let expected_bce = 0.0 - ln_f64(0.8)
    if abs_f64(bce52 - expected_bce) > 0.01 { ok = false; println("  FAIL: BCE mismatch") }
    // grad = (0.8 - 1) / (0.8 * 0.2) = -0.2 / 0.16 = -1.25
    let expected_bce_grad = (0.8 - 1.0) / (0.8 * 0.2)
    if abs_f64(bce_grad52 - expected_bce_grad) > 0.01 { ok = false; println("  FAIL: BCE grad mismatch") }
    println("")

    // Test 53: Hinge loss (SVM)
    println("Test 53: Hinge loss (SVM)")
    // Correct classification with margin: pred=2.0, target=1.0
    let hinge_correct = loss_hinge(2.0, 1.0)
    // margin = 1 - 1*2 = -1, so loss = max(0, -1) = 0

    // Misclassification: pred=-0.5, target=1.0
    let hinge_wrong = loss_hinge(0.0 - 0.5, 1.0)
    // margin = 1 - 1*(-0.5) = 1.5, so loss = max(0, 1.5) = 1.5

    println("  Correct (pred=2, y=1): loss = ")
    println(hinge_correct)
    println("  Wrong (pred=-0.5, y=1): loss = ")
    println(hinge_wrong)

    if abs_f64(hinge_correct - 0.0) > tol { ok = false; println("  FAIL: Hinge correct should be 0") }
    if abs_f64(hinge_wrong - 1.5) > tol { ok = false; println("  FAIL: Hinge wrong should be 1.5") }
    println("")

    // Test 54: Log-cosh loss
    println("Test 54: Log-cosh loss")
    let logcosh54 = loss_log_cosh(3.0, 1.0)
    let logcosh_grad54 = loss_log_cosh_grad(3.0, 1.0)

    println("  pred=3, target=1")
    println("  LogCosh = ")
    println(logcosh54)
    println("  LogCosh grad = ")
    println(logcosh_grad54)

    // log(cosh(2)) ≈ 1.325, tanh(2) ≈ 0.964
    let expected_logcosh = ln_f64(cosh_f64(2.0))
    let expected_logcosh_grad = tanh_f64(2.0)
    if abs_f64(logcosh54 - expected_logcosh) > 0.01 { ok = false; println("  FAIL: LogCosh mismatch") }
    if abs_f64(logcosh_grad54 - expected_logcosh_grad) > 0.01 { ok = false; println("  FAIL: LogCosh grad mismatch") }
    println("")

    // Test 55: Focal loss (for imbalanced classification)
    println("Test 55: Focal loss")
    // High confidence correct: pred=0.9, target=1.0
    let focal_high = loss_focal_default(0.9, 1.0)
    // Low confidence correct: pred=0.6, target=1.0
    let focal_low = loss_focal_default(0.6, 1.0)

    println("  High confidence (p=0.9, y=1): loss = ")
    println(focal_high)
    println("  Low confidence (p=0.6, y=1): loss = ")
    println(focal_low)

    // Focal loss should be lower for high confidence (downweights easy examples)
    if focal_high >= focal_low { ok = false; println("  FAIL: Focal should be lower for high conf") }
    // Both should be positive
    if focal_high < 0.0 { ok = false; println("  FAIL: Focal should be >= 0") }
    if focal_low < 0.0 { ok = false; println("  FAIL: Focal should be >= 0") }
    println("")

    // Test 56: KL divergence
    println("Test 56: KL divergence")
    // KL(p=0.3 || q=0.5) = 0.3 * log(0.3/0.5)
    let kl56 = loss_kl_div(0.3, 0.5)

    println("  KL(p=0.3 || q=0.5) = ")
    println(kl56)

    let expected_kl = 0.3 * ln_f64(0.3 / 0.5)
    if abs_f64(kl56 - expected_kl) > 0.01 { ok = false; println("  FAIL: KL mismatch") }
    // KL should be negative when p < q (for this single term)
    println("")

    // Test 57: Quantile loss
    println("Test 57: Quantile loss")
    // Median (q=0.5): symmetric
    let quant_under = loss_quantile(1.0, 3.0, 0.5)  // pred=1, target=3, underprediction
    let quant_over = loss_quantile(5.0, 3.0, 0.5)   // pred=5, target=3, overprediction

    // 90th percentile (q=0.9): penalizes underprediction more
    let quant90_under = loss_quantile(1.0, 3.0, 0.9)
    let quant90_over = loss_quantile(5.0, 3.0, 0.9)

    println("  Median (q=0.5), under: ")
    println(quant_under)
    println("  Median (q=0.5), over: ")
    println(quant_over)
    println("  q=0.9, under: ")
    println(quant90_under)
    println("  q=0.9, over: ")
    println(quant90_over)

    // Median should be symmetric: 0.5 * |3-1| = 1.0, 0.5 * |3-5| = 1.0
    if abs_f64(quant_under - 1.0) > tol { ok = false; println("  FAIL: Quant median under") }
    if abs_f64(quant_over - 1.0) > tol { ok = false; println("  FAIL: Quant median over") }
    // 90th: underprediction = 0.9 * 2 = 1.8, overprediction = 0.1 * 2 = 0.2
    if abs_f64(quant90_under - 1.8) > tol { ok = false; println("  FAIL: Quant 90 under") }
    if abs_f64(quant90_over - 0.2) > tol { ok = false; println("  FAIL: Quant 90 over") }
    println("")

    // Test 58: Triplet margin loss
    println("Test 58: Triplet margin loss")
    // Good embedding: anchor closer to positive than negative
    let trip_good = loss_triplet_default(0.0, 0.1, 2.0)  // d_pos=0.1, d_neg=2.0
    // Bad embedding: anchor closer to negative
    let trip_bad = loss_triplet_default(0.0, 2.0, 0.1)   // d_pos=2.0, d_neg=0.1

    println("  Good (d_pos < d_neg): loss = ")
    println(trip_good)
    println("  Bad (d_pos > d_neg): loss = ")
    println(trip_bad)

    // Good: max(0, 0.1 - 2.0 + 1.0) = max(0, -0.9) = 0
    if abs_f64(trip_good - 0.0) > tol { ok = false; println("  FAIL: Triplet good should be 0") }
    // Bad: max(0, 2.0 - 0.1 + 1.0) = max(0, 2.9) = 2.9
    if abs_f64(trip_bad - 2.9) > tol { ok = false; println("  FAIL: Triplet bad should be 2.9") }
    println("")

    // ========================================================================
    // WEIGHT INITIALIZATION TESTS
    // ========================================================================

    // Test 59: RNG basic functionality
    println("Test 59: RNG basic functionality")
    let rng59 = rng_new(42.0)  // Seed with 42

    // Generate several random numbers
    let r1_59 = rng_next(rng59)
    let r2_59 = rng_next(r1_59.rng)
    let r3_59 = rng_next(r2_59.rng)

    println("  seed=42, r1 = ")
    println(r1_59.value)
    println("  r2 = ")
    println(r2_59.value)
    println("  r3 = ")
    println(r3_59.value)

    // All values should be in [0, 1)
    if r1_59.value < 0.0 { ok = false; println("  FAIL: r1 < 0") }
    if r1_59.value >= 1.0 { ok = false; println("  FAIL: r1 >= 1") }
    if r2_59.value < 0.0 { ok = false; println("  FAIL: r2 < 0") }
    if r2_59.value >= 1.0 { ok = false; println("  FAIL: r2 >= 1") }
    // Values should be different
    if abs_f64(r1_59.value - r2_59.value) < 0.0001 { ok = false; println("  FAIL: r1 == r2") }
    if abs_f64(r2_59.value - r3_59.value) < 0.0001 { ok = false; println("  FAIL: r2 == r3") }
    println("")

    // Test 60: Xavier initialization bounds
    println("Test 60: Xavier initialization")
    let fan_in60 = 256.0
    let fan_out60 = 128.0

    // Xavier uniform bound = sqrt(6 / (256 + 128)) = sqrt(6/384) ≈ 0.125
    let xavier_bound = xavier_uniform_bound(fan_in60, fan_out60)
    // Xavier normal std = sqrt(2 / (256 + 128)) = sqrt(2/384) ≈ 0.0722
    let xavier_std = xavier_normal_std(fan_in60, fan_out60)

    println("  fan_in=256, fan_out=128")
    println("  Xavier uniform bound = ")
    println(xavier_bound)
    println("  Xavier normal std = ")
    println(xavier_std)

    let expected_xavier_bound = sqrt_f64(6.0 / 384.0)
    let expected_xavier_std = sqrt_f64(2.0 / 384.0)
    if abs_f64(xavier_bound - expected_xavier_bound) > tol { ok = false; println("  FAIL: Xavier bound") }
    if abs_f64(xavier_std - expected_xavier_std) > tol { ok = false; println("  FAIL: Xavier std") }

    // Generate a few Xavier uniform weights
    let rng60 = rng_new(123.0)
    let xu1 = init_xavier_uniform(rng60, fan_in60, fan_out60)
    let xu2 = init_xavier_uniform(xu1.rng, fan_in60, fan_out60)

    println("  Xavier uniform w1 = ")
    println(xu1.value)
    println("  Xavier uniform w2 = ")
    println(xu2.value)

    // Weights should be within bounds
    if abs_f64(xu1.value) > xavier_bound + 0.001 { ok = false; println("  FAIL: xu1 out of bounds") }
    if abs_f64(xu2.value) > xavier_bound + 0.001 { ok = false; println("  FAIL: xu2 out of bounds") }
    println("")

    // Test 61: He initialization bounds
    println("Test 61: He/Kaiming initialization")
    let fan_in61 = 512.0

    // He uniform bound = sqrt(6 / 512) ≈ 0.108
    let he_bound = he_uniform_bound(fan_in61)
    // He normal std = sqrt(2 / 512) ≈ 0.0625
    let he_std = he_normal_std(fan_in61)

    println("  fan_in=512")
    println("  He uniform bound = ")
    println(he_bound)
    println("  He normal std = ")
    println(he_std)

    let expected_he_bound = sqrt_f64(6.0 / 512.0)
    let expected_he_std = sqrt_f64(2.0 / 512.0)
    if abs_f64(he_bound - expected_he_bound) > tol { ok = false; println("  FAIL: He bound") }
    if abs_f64(he_std - expected_he_std) > tol { ok = false; println("  FAIL: He std") }

    // Generate He normal weights
    let rng61 = rng_new(456.0)
    let he1 = init_he_normal(rng61, fan_in61)
    let he2 = init_he_normal(he1.rng, fan_in61)

    println("  He normal w1 = ")
    println(he1.value)
    println("  He normal w2 = ")
    println(he2.value)

    // He normal should have reasonable magnitude (within 4 std)
    if abs_f64(he1.value) > 4.0 * he_std { ok = false; println("  FAIL: he1 too large") }
    if abs_f64(he2.value) > 4.0 * he_std { ok = false; println("  FAIL: he2 too large") }
    println("")

    // Test 62: LeCun initialization
    println("Test 62: LeCun initialization")
    let fan_in62 = 1024.0

    // LeCun std = sqrt(1 / 1024) ≈ 0.03125
    let lecun_std = lecun_normal_std(fan_in62)

    println("  fan_in=1024")
    println("  LeCun normal std = ")
    println(lecun_std)

    let expected_lecun_std = sqrt_f64(1.0 / 1024.0)
    if abs_f64(lecun_std - expected_lecun_std) > tol { ok = false; println("  FAIL: LeCun std") }

    // Generate LeCun weight
    let rng62 = rng_new(789.0)
    let lc1 = init_lecun_normal(rng62, fan_in62)

    println("  LeCun normal w1 = ")
    println(lc1.value)

    if abs_f64(lc1.value) > 4.0 * lecun_std { ok = false; println("  FAIL: lc1 too large") }
    println("")

    // Test 63: Normal distribution via Box-Muller
    println("Test 63: Box-Muller normal distribution")
    let rng63 = rng_new(111.0)
    let mean63 = 5.0
    let std63 = 2.0

    // Generate several normal samples
    let n1 = rng_normal(rng63, mean63, std63)
    let n2 = rng_normal(n1.rng, mean63, std63)
    let n3 = rng_normal(n2.rng, mean63, std63)
    let n4 = rng_normal(n3.rng, mean63, std63)
    let n5 = rng_normal(n4.rng, mean63, std63)

    println("  N(5, 2) samples:")
    println("    n1 = ")
    println(n1.value)
    println("    n2 = ")
    println(n2.value)
    println("    n3 = ")
    println(n3.value)

    // Compute sample mean
    let sample_mean = (n1.value + n2.value + n3.value + n4.value + n5.value) / 5.0
    println("  Sample mean (5 samples) = ")
    println(sample_mean)

    // Sample mean should be roughly near 5.0 (within 2 std of mean = 4 std errors)
    // With 5 samples, std error = 2/sqrt(5) ≈ 0.894, so 4*0.894 ≈ 3.58
    if abs_f64(sample_mean - mean63) > 4.0 { ok = false; println("  FAIL: sample mean too far") }

    // All values should be finite (not NaN or inf)
    if n1.value != n1.value { ok = false; println("  FAIL: n1 is NaN") }
    if n2.value != n2.value { ok = false; println("  FAIL: n2 is NaN") }
    println("")

    // Test 64: Sparse initialization
    println("Test 64: Sparse initialization")
    let rng64 = rng_new(222.0)
    let sparsity64 = 0.7  // 70% zeros

    // Generate several sparse weights
    let mut zero_count = 0.0
    let mut s_rng = rng64

    let s1 = init_sparse(s_rng, 1.0, sparsity64)
    s_rng = s1.rng
    if abs_f64(s1.value) < 0.0001 { zero_count = zero_count + 1.0 }

    let s2 = init_sparse(s_rng, 1.0, sparsity64)
    s_rng = s2.rng
    if abs_f64(s2.value) < 0.0001 { zero_count = zero_count + 1.0 }

    let s3 = init_sparse(s_rng, 1.0, sparsity64)
    s_rng = s3.rng
    if abs_f64(s3.value) < 0.0001 { zero_count = zero_count + 1.0 }

    let s4 = init_sparse(s_rng, 1.0, sparsity64)
    s_rng = s4.rng
    if abs_f64(s4.value) < 0.0001 { zero_count = zero_count + 1.0 }

    let s5 = init_sparse(s_rng, 1.0, sparsity64)
    if abs_f64(s5.value) < 0.0001 { zero_count = zero_count + 1.0 }

    println("  Sparse(std=1, sparsity=0.7):")
    println("    s1 = ")
    println(s1.value)
    println("    s2 = ")
    println(s2.value)
    println("    s3 = ")
    println(s3.value)
    println("  Zero count (of 5) = ")
    println(zero_count)

    // With 70% sparsity, expect some zeros (but random, so just check it works)
    // Non-zero values should be reasonable
    println("")

    // Test 65: Truncated normal
    println("Test 65: Truncated normal initialization")
    let rng65 = rng_new(333.0)
    let mean65 = 0.0
    let std65 = 1.0

    // Generate truncated normal samples
    let tn1 = init_truncated_normal(rng65, mean65, std65)
    let tn2 = init_truncated_normal(tn1.rng, mean65, std65)
    let tn3 = init_truncated_normal(tn2.rng, mean65, std65)

    println("  Truncated N(0, 1) samples:")
    println("    tn1 = ")
    println(tn1.value)
    println("    tn2 = ")
    println(tn2.value)
    println("    tn3 = ")
    println(tn3.value)

    // All values should be within [-2, 2] (2 std from mean)
    if abs_f64(tn1.value) > 2.0 + 0.001 { ok = false; println("  FAIL: tn1 outside [-2, 2]") }
    if abs_f64(tn2.value) > 2.0 + 0.001 { ok = false; println("  FAIL: tn2 outside [-2, 2]") }
    if abs_f64(tn3.value) > 2.0 + 0.001 { ok = false; println("  FAIL: tn3 outside [-2, 2]") }
    println("")

    // Test 66: Convenience initialization functions
    println("Test 66: Convenience initialization functions")
    let rng66 = rng_new(444.0)

    // ReLU default (He)
    let relu_w = init_default_relu(rng66, 256.0)
    // Tanh default (Xavier)
    let tanh_w = init_default_tanh(relu_w.rng, 256.0, 128.0)
    // Transformer default
    let trans_w = init_default_transformer(tanh_w.rng, 512.0)
    // Bias default
    let bias = init_default_bias()

    println("  ReLU default (fan_in=256) = ")
    println(relu_w.value)
    println("  Tanh default (256->128) = ")
    println(tanh_w.value)
    println("  Transformer default (d=512) = ")
    println(trans_w.value)
    println("  Bias default = ")
    println(bias)

    // Bias should be 0
    if abs_f64(bias - 0.0) > tol { ok = false; println("  FAIL: bias should be 0") }
    // All weights should be finite
    if relu_w.value != relu_w.value { ok = false; println("  FAIL: relu_w is NaN") }
    if tanh_w.value != tanh_w.value { ok = false; println("  FAIL: tanh_w is NaN") }
    if trans_w.value != trans_w.value { ok = false; println("  FAIL: trans_w is NaN") }
    println("")

    // Test 67: Batch normalization
    println("Test 67: Batch normalization")

    // Create a batch of values: [2, 4, 6, 8]
    let bn_x1 = 2.0
    let bn_x2 = 4.0
    let bn_x3 = 6.0
    let bn_x4 = 8.0

    // Compute batch statistics
    let bn_stats = compute_batch_stats_4(bn_x1, bn_x2, bn_x3, bn_x4)
    // mean = (2+4+6+8)/4 = 5, var = ((−3)² + (−1)² + 1² + 3²)/4 = (9+1+1+9)/4 = 5

    println("  Batch mean = ")
    println(bn_stats.mean)
    println("  Batch var = ")
    println(bn_stats.variance)

    if abs_f64(bn_stats.mean - 5.0) > tol { ok = false; println("  FAIL: mean != 5") }
    if abs_f64(bn_stats.variance - 5.0) > tol { ok = false; println("  FAIL: var != 5") }

    // Apply batch norm with gamma=1, beta=0
    let bn_state = batchnorm_default()
    let bn_result = batchnorm_forward_train(bn_x1, bn_stats.mean, bn_stats.variance, bn_state)

    // x_norm = (2 - 5) / sqrt(5 + eps) ≈ -3 / 2.236 ≈ -1.342
    println("  Normalized x1 = ")
    println(bn_result.x_norm)
    println("  Output (gamma=1, beta=0) = ")
    println(bn_result.output)

    let expected_bn_norm = (bn_x1 - bn_stats.mean) / sqrt_f64(bn_stats.variance + 0.00001)
    if abs_f64(bn_result.x_norm - expected_bn_norm) > tol { ok = false; println("  FAIL: x_norm wrong") }
    if abs_f64(bn_result.output - expected_bn_norm) > tol { ok = false; println("  FAIL: output wrong") }

    // Check running mean update (momentum=0.1)
    println("  Running mean after update = ")
    println(bn_result.bn_state.running_mean)
    // new_running_mean = 0.9 * 0 + 0.1 * 5 = 0.5
    if abs_f64(bn_result.bn_state.running_mean - 0.5) > tol { ok = false; println("  FAIL: running mean") }
    println("")

    // Test 68: Layer normalization
    println("Test 68: Layer normalization")

    // Normalize across features (use same batch stats as feature stats for simplicity)
    let ln_state = layernorm_default()
    let ln_result = layernorm_forward(bn_x1, bn_stats.mean, bn_stats.variance, ln_state)

    println("  LayerNorm output = ")
    println(ln_result.output)
    println("  LayerNorm x_norm = ")
    println(ln_result.x_norm)

    // Should match batch norm result (same formula)
    if abs_f64(ln_result.x_norm - expected_bn_norm) > tol { ok = false; println("  FAIL: LN x_norm") }
    println("")

    // Test 69: Dropout forward (training)
    println("Test 69: Dropout forward")
    let rng69 = rng_new(555.0)
    let drop_p = 0.5  // 50% dropout

    // Apply dropout to several values
    let d1 = dropout_forward_train(1.0, drop_p, rng69)
    let d2 = dropout_forward_train(1.0, drop_p, d1.rng)
    let d3 = dropout_forward_train(1.0, drop_p, d2.rng)
    let d4 = dropout_forward_train(1.0, drop_p, d3.rng)
    let d5 = dropout_forward_train(1.0, drop_p, d4.rng)
    let d6 = dropout_forward_train(1.0, drop_p, d5.rng)

    println("  p=0.5 dropout outputs (input=1.0):")
    println("    d1 = ")
    println(d1.output)
    println("    d2 = ")
    println(d2.output)
    println("    d3 = ")
    println(d3.output)
    println("    d4 = ")
    println(d4.output)

    // Count how many were kept vs dropped
    let mut kept_count = 0.0
    if d1.mask > 0.0 { kept_count = kept_count + 1.0 }
    if d2.mask > 0.0 { kept_count = kept_count + 1.0 }
    if d3.mask > 0.0 { kept_count = kept_count + 1.0 }
    if d4.mask > 0.0 { kept_count = kept_count + 1.0 }
    if d5.mask > 0.0 { kept_count = kept_count + 1.0 }
    if d6.mask > 0.0 { kept_count = kept_count + 1.0 }

    println("  Kept count (of 6) = ")
    println(kept_count)

    // Output should be either 0 (dropped) or 2 (kept with scale 1/(1-0.5)=2)
    if d1.output != 0.0 {
        if abs_f64(d1.output - 2.0) > tol { ok = false; println("  FAIL: d1 scale wrong") }
    }

    // Test inference mode (no dropout)
    let d_inf = dropout_forward_inference(5.0)
    println("  Inference output (input=5.0) = ")
    println(d_inf)
    if abs_f64(d_inf - 5.0) > tol { ok = false; println("  FAIL: inference should pass through") }
    println("")

    // Test 70: Dropout backward
    println("Test 70: Dropout backward")

    // Gradient should be scaled by the same mask
    let grad_out = 1.0
    let grad_d1 = dropout_backward(grad_out, d1.mask)
    let grad_d2 = dropout_backward(grad_out, d2.mask)

    println("  grad_d1 (mask=")
    println(d1.mask)
    println(") = ")
    println(grad_d1)
    println("  grad_d2 (mask=")
    println(d2.mask)
    println(") = ")
    println(grad_d2)

    // Gradient should match mask
    if abs_f64(grad_d1 - d1.mask) > tol { ok = false; println("  FAIL: grad_d1") }
    if abs_f64(grad_d2 - d2.mask) > tol { ok = false; println("  FAIL: grad_d2") }
    println("")

    // Test 71: RMS Normalization
    println("Test 71: RMS Normalization")

    // RMS of [3, 4] = sqrt((9 + 16)/2) = sqrt(12.5) ≈ 3.536
    let rms_val = compute_rms_2(3.0, 4.0)
    println("  RMS([3, 4]) = ")
    println(rms_val)

    let expected_rms = sqrt_f64((9.0 + 16.0) / 2.0)
    if abs_f64(rms_val - expected_rms) > tol { ok = false; println("  FAIL: RMS value") }

    // Apply RMS norm
    let rms_state = rmsnorm_default()
    let rms_out = rmsnorm_forward(3.0, rms_val, rms_state)

    println("  RMSNorm(3.0) = ")
    println(rms_out)

    // Expected: gamma * 3 / rms = 1 * 3 / 3.536 ≈ 0.848
    let expected_rms_out = 3.0 / rms_val
    if abs_f64(rms_out - expected_rms_out) > tol { ok = false; println("  FAIL: RMSNorm output") }
    println("")

    // Test 72: Batch norm backward
    println("Test 72: Batch norm backward")

    // Backward pass with dout=1.0, x_norm from test 67
    let bn_grads = batchnorm_backward(1.0, bn_result.x_norm, bn_state.gamma)

    println("  dout=1.0, x_norm=")
    println(bn_result.x_norm)
    println("  dgamma = ")
    println(bn_grads.dgamma)
    println("  dbeta = ")
    println(bn_grads.dbeta)
    println("  dx = ")
    println(bn_grads.dx)

    // dgamma should equal x_norm * dout = x_norm
    if abs_f64(bn_grads.dgamma - bn_result.x_norm) > tol { ok = false; println("  FAIL: dgamma") }
    // dbeta should equal dout = 1.0
    if abs_f64(bn_grads.dbeta - 1.0) > tol { ok = false; println("  FAIL: dbeta") }
    println("")

    // Test 73: Alpha Dropout (for SELU)
    println("Test 73: Alpha Dropout (SELU)")
    let rng73 = rng_new(777.0)
    let alpha_p = 0.3

    let ad1 = alpha_dropout_forward_train(1.0, alpha_p, rng73)
    let ad2 = alpha_dropout_forward_train(1.0, alpha_p, ad1.rng)
    let ad3 = alpha_dropout_forward_train(1.0, alpha_p, ad2.rng)

    println("  Alpha dropout outputs (p=0.3, input=1.0):")
    println("    ad1 = ")
    println(ad1.output)
    println("    ad2 = ")
    println(ad2.output)
    println("    ad3 = ")
    println(ad3.output)

    // Alpha dropout keeps self-normalizing property
    // Output is not 0 when dropped, but -alpha * scale
    // All outputs should be finite
    if ad1.output != ad1.output { ok = false; println("  FAIL: ad1 is NaN") }
    if ad2.output != ad2.output { ok = false; println("  FAIL: ad2 is NaN") }
    println("")

    // Test 74: Group Normalization
    println("Test 74: Group Normalization")
    let gn_state = groupnorm_default(4.0)  // 4 groups

    // Use same stats as batch norm for simplicity
    let gn_result = groupnorm_forward(bn_x1, bn_stats.mean, bn_stats.variance, gn_state)

    println("  GroupNorm output = ")
    println(gn_result.output)

    // Should match batch/layer norm (same formula)
    if abs_f64(gn_result.x_norm - expected_bn_norm) > tol { ok = false; println("  FAIL: GN x_norm") }
    println("")

    // Test 75: Instance Normalization
    println("Test 75: Instance Normalization")
    let in_state = instancenorm_default()

    let in_result = instancenorm_forward(bn_x1, bn_stats.mean, bn_stats.variance, in_state)

    println("  InstanceNorm output = ")
    println(in_result.output)

    // Should match batch/layer norm (same formula)
    if abs_f64(in_result.x_norm - expected_bn_norm) > tol { ok = false; println("  FAIL: IN x_norm") }
    println("")

    // Test 76: DropConnect
    println("Test 76: DropConnect")
    let rng76 = rng_new(888.0)

    // DropConnect drops weights, not activations
    let dc1 = dropconnect_forward(2.0, 3.0, 0.5, rng76)  // x=2, w=3, p=0.5
    let dc2 = dropconnect_forward(2.0, 3.0, 0.5, dc1.rng)

    println("  DropConnect (x=2, w=3, p=0.5):")
    println("    dc1 output = ")
    println(dc1.output)
    println("    dc2 output = ")
    println(dc2.output)

    // Output should be 0 (dropped) or 12 (2*3*2 with scale)
    if dc1.output != 0.0 {
        if abs_f64(dc1.output - 12.0) > tol { ok = false; println("  FAIL: dc1 scale wrong") }
    }
    println("")

    if ok {
        println("ALL TESTS PASSED")
        return 0
    } else {
        println("SOME TESTS FAILED")
        return 1
    }
}
