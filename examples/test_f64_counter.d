fn main() -> f64 {
    let mut x = 10.0
    let mut t = 0.0
    while t < 1.0 {
        x = x * 0.9
        t = t + 0.1
    }
    return x
}
