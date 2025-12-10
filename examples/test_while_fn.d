fn step(x: f64) -> f64 {
    return x * 0.9
}

fn main() -> i32 {
    let mut x = 100.0
    let mut i: i32 = 0
    
    while i < 10 {
        x = step(x)
        i = i + 1
    }
    
    return i
}
