# Quick-Noise

WIP High performance SIMD batched processing noise library. Made in Rust.

# Disclaimer

Very early work in progress! Only 2D Perlin uniform grid and 3D Perlin uniform grid is completed. Feel free to add to or improve on the library!

Rust nightly is **required** to use this library. It utilizes std::simd, which is only available on nightly.

# Performance

## Uniform Grid

Uniform grid computes a batch of noise results positioned on a uniform grid.
These batches are much faster than a regular noise call, but constrain the input position patterns.
Ideal for terrain and texture generation that needs uniform noise. 2D Perlin operates on 32x32 (1024 total) batches,
while 3D Perlin operates on 32x32x32 (32768 total) batches.

Scale measures how far apart gridpoints are from eachother, also known as frequency.
A higher scale means noise is smoother and changes slower from sample to sample.
Uniform grid computes larger scales faster (up until 32.0), and computes them slightly slower
for scales that are not a power of two. Scale can be any 32-bit floating point number >= 2.0.

The following results were benched on an Intel I7-13700H CPU equipped with AVX2 on an XPS 15 9530 laptop on Linux.
Results are measured in billions of points per second for a single noise pass.

| Scale | 2D Perlin | 3D Perlin |
|-------|-----------|-----------|
| 64    | 10.2 B/s  | 13.1 B/s  |
| 48    | 9.08 B/s  | 12.4 B/s  |
| 32    | 10.0 B/s  | 13.1 B/s  |
| 24    | 7.77 B/s  | 10.8 B/s  |
| 16    | 8.02 B/s  | 11.0 B/s  |
| 8     | 5.34 B/s  | 7.09 B/s  |
| 4     | 2.50 B/s  | 2.20 B/s  |

# Running

Height maps can be generated in `examples/basic.rs`. To run these examples, use:

```
RUSTFLAGS='-C target-cpu=native' cargo run --release --example basic
```

It is important that `RUSTFLAGS='-C target-cpu=native'` and `--release` is used for the best performance. Also ensure to create a folder named `noise_images` for storing image output.

Criterion benches can be run with:

```
RUSTFLAGS='-C target-cpu=native' cargo bench
```
