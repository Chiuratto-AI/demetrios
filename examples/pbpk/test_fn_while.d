fn test(a: f64, b: f64) -> f64 {
    let c = a / b
    let mut x = 0.0
    
    while x < 1.0 {
        x = x + 0.1
    }
    
    return c
}

fn main() -> i32 {
    let result = test(30.0, 77.0)
    return 1
}
