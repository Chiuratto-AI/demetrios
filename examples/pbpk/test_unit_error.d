// This should FAIL - adding volume to time

fn bad_calculation(vd: f64@L, time: f64@h) -> f64 {
    return vd + time
}

fn main() -> i32 {
    let result = bad_calculation(77.0, 2.0)
    return 0
}
