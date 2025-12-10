fn sim(dose: f64, vd: f64, cl: f64, t: f64) -> f64 {
    let k = cl / vd
    let dt = 0.001
    let c_init = dose / vd
    let mut c = c_init
    let mut time = 0.0
    
    while time < t {
        c = c * (1.0 - k * dt)
        time = time + dt
    }
    
    return c
}

fn main() -> i32 {
    return 1
}
