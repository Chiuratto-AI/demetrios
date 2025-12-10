fn half(x: f64) -> f64 {
    return x * 0.5
}

fn main() -> f64 {
    let mut x = 10.0
    x = half(x)
    x = half(x)
    x = half(x)
    return x
}
