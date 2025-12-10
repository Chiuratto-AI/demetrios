fn f4(a: f64, b: f64, c: f64, d: f64) -> f64 {
    return a + b + c + d
}

fn main() -> i32 {
    let mut x = 0.0
    let mut i: i32 = 0
    
    while i < 3 {
        x = f4(1.0, 2.0, 3.0, 4.0)
        i = i + 1
    }
    
    return i
}
