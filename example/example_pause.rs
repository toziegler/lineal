use std::{arch::x86_64::_mm_pause, collections::HashMap, hint::black_box};

fn main() -> Result<(), std::io::Error> {
    let operations = 10_000_000;
    let mut benchmark_params = Vec::new();
    benchmark_params.push(("Test".to_owned(), "Running".to_owned()));
    {
        let lineal = lineal::PerfEventBlock::new(benchmark_params, operations as f64)?;
        for _ in 0..operations {
            unsafe { _mm_pause() };
        }
    }

    Ok(())
}
