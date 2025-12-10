fn half(x: f64) -> f64 {
    return x * 0.5
}

fn main() -> f64 {
    let mut x = 10.0
    let mut i: i32 = 0
    
    while i < 3 {
        x = half(x)
        i = i + 1
    }
    
    return x
}
