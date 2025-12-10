fn main() -> f64 {
    let k = 0.39
    let dt = 0.001
    let factor = 1.0 - k * dt
    let mut c = 0.026
    let mut i: i32 = 0
    while i < 100 {
        c = c * factor
        i = i + 1
    }
    return c
}
