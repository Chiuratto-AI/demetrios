fn main() -> f64 {
    let mut x = 10.0
    let mut i: i32 = 0
    let factor = 0.5
    
    while i < 3 {
        x = x * factor
        i = i + 1
    }
    
    return x
}
