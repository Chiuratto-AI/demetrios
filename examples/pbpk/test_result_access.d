struct PKResults {
    cmax: f64@mg_per_L,
    t_half: f64@h
}

fn calc_pk() -> PKResults {
    return PKResults {
        cmax: 1.0,
        t_half: 5.0
    }
}

fn main() -> i32 {
    let results = calc_pk();
    let x = results.t_half;
    let y = results.cmax;
    return 0
}
