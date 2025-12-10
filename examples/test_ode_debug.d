fn euler(c: f64, k: f64, dt: f64) -> f64 {
    let dc = 0.0 - k * c
    return c + dc * dt
}

fn main() -> f64 {
    let k = 0.39
    let dt = 0.01
    
    let mut c = 0.026
    let mut t = 0.0
    
    while t < 2.0 {
        c = euler(c, k, dt)
        t = t + dt
    }
    
    return c
}
