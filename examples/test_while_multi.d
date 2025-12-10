fn add(a: f64, b: f64) -> f64 {
    return a + b
}

fn main() -> i32 {
    let mut x = 0.0
    let mut i: i32 = 0
    
    while i < 5 {
        x = add(x, 1.0)
        i = i + 1
    }
    
    return i
}
