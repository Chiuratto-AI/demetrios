// Test unit error detection - this SHOULD fail

fn main() -> i32 {
    let dose: f64@mg = 500.0
    let time: f64@h = 2.0
    let bad = dose + time
    return 0
}
