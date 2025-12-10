fn f4(a: f64, b: f64, c: f64, d: f64) -> f64 {
    return a + b + c + d
}

fn main() -> i32 {
    let c = 1.0
    let cl = 2.0
    let v = 3.0
    let dt = 4.0
    
    let mut x = 0.0
    let mut i: i32 = 0
    
    while i < 3 {
        x = f4(c, cl, v, dt)
        i = i + 1
    }
    
    return i
}
