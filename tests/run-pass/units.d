//@ run-pass
// Units of measure test

use units::{mg, mL}

fn main() {
    let dose: mg = 500.0
    let volume: mL = 10.0
    let concentration: mg/mL = dose / volume

    assert(concentration == 50.0 : mg/mL)
}
