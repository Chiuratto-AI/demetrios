// Darwin PBPK ODE Solver - Working Version
// Single compartment IV bolus

fn main() -> f64 {
    // Midazolam pharmacokinetics
    let dose = 2.0      // mg
    let v = 77.0        // L (Vd)
    let cl = 30.0       // L/h (clearance)
    
    // Derived parameters
    let k = cl / v      // elimination rate constant (1/h)
    
    // Simulation
    let dt = 0.001      // hour
    let mut c = dose / v
    let mut t = 0.0
    
    while t < 2.0 {
        c = c * (1.0 - k * dt)
        t = t + dt
    }
    
    return c
}
