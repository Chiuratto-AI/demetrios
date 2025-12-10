fn step(x: f64) -> f64 {
    return x * 0.9
}

fn main() -> f64 {
    let mut x = 100.0
    let mut i = 0.0
    
    while i < 3.0 {
        x = step(x)
        i = i + 1.0
    }
    
    return x
}
