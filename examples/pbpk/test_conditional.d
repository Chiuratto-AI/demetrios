fn main() -> i32 {
    let x = 1.5;
    let pass = x >= 0.5;
    
    let result = if pass {
        0
    } else {
        1
    };
    
    return result
}
