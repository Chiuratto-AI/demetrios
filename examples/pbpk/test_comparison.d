fn main() -> i32 {
    let x = 1.5;
    let y = 0.5;
    let z = 2.0;
    
    let pass1 = x >= y && x <= z;
    
    return if pass1 { 0 } else { 1 }
}
