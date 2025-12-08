struct Data {
    value: f64
}

fn test(d: Data) -> f64 {
    return d.value
}

fn main() -> i32 {
    let d = Data {
        value: 1.5
    };
    let x = test(d);
    return 0
}
