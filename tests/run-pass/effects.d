//@ run-pass
// Algebraic effects test

fn pure_computation(x: i32) -> i32 {
    x * 2
}

fn effectful_read() -> string with IO {
    "test data"
}

fn main() with IO {
    let result = pure_computation(21)
    assert(result == 42)

    let data = effectful_read()
    println(data)
}
