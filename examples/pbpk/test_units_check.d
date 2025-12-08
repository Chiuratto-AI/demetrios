// Test unit compatibility checking

fn main() -> i32 {
    let dose: f64@mg = 500.0
    let volume: f64@L = 10.0
    let time: f64@h = 2.0
    
    // This should work: mg / L = mg/L
    let conc = dose / volume
    
    // This should fail: mg + h incompatible
    let bad = dose + time
    
    return 0
}
