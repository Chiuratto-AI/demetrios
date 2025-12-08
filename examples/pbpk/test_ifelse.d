// Test if-else with binary operators

fn test_multiply() -> f64 {
    let x = 2.0
    let result = if x > 1.0 {
        x * 1.5
    } else {
        x
    }
    return result
}

fn main() -> i32 {
    let r = test_multiply()
    return 0
}
