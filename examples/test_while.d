// Test while loop in Demetrios

fn main() -> i32 {
    let mut count: i32 = 0
    let mut sum: i32 = 0
    
    while count < 10 {
        sum = sum + count
        count = count + 1
    }
    
    return sum
}
