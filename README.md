# Quick-Noise

WIP High performance SIMD batched processing noise library. Made in Rust.

# Disclaimer

Very early work in progress! Only 2D Perlin uniform grid and 3D Perlin uniform grid is completed. Feel free to add to or improve on the library!

Rust nightly is not needed to use this library. It utilizes a custom simd wrapper rather than std::simd. However, only x86 systems are currently supported (ie SSE, AVX2, and AVX512) Aarch (arm) will be added
in the future.

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

Results are measured in billions of points per second single-threaded for one noise pass. 
- AVX2: I7-13700H | XPS 15 9530 Laptop | Linux
- AVX512: Ryzen 7 9800X3D | Linux

| Scale | 2D Perlin AVX2 | 3D Perlin AVX2 | 2D Perlin AVX512 | 3D Perlin AVX512 |
|-------|----------------|----------------|------------------|------------------|
| 64    | 10.3 B/s       | 13.5 B/s       | 17.6 B/s         | 51.0 B/s         |
| 48    | 9.23 B/s       | 12.7 B/s       | 15.4 B/s         | 42.5 B/s         |
| 32    | 10.3 B/s       | 13.5 B/s       | 17.6 B/s         | 51.0 B/s         |
| 24    | 8.03 B/s       | 11.4 B/s       | 12.8 B/s         | 32.7 B/s         |
| 16    | 8.12 B/s       | 11.9 B/s       | 14.2 B/s         | 33.3 B/s         |
| 8     | 5.48 B/s       | 8.22 B/s       | 8.74 B/s         | 20.9 B/s         |
| 4     | 2.82 B/s       | 3.20 B/s       | 4.73 B/s         | 5.42 B/s         |


## Batched

Batched noise processing is much more flexible than uniform grid, allowing for any arbitrary input and enabling
techniques such as domain warping. Results are measured in millions of points per second. Performance is still WIP.

|   Perlin    | 2D AVX2 | 3D AVX2 |
|-------------|---------|---------|
| Quick-Noise | 980 M/s | 304 M/s |
| FastNoise2  | 509 M/s | 224 M/s |

|   Simplex   | 2D AVX2 | 3D AVX2 |
|-------------|---------|---------|
| Quick-Noise | 638 M/s | 298 M/s |
| FastNoise2  | 425 M/s | 241 M/s |

|    Value    | 2D AVX2   | 3D AVX2 |
|-------------|-----------|---------|
| Quick-Noise | 1,080 M/s | 619 M/s |
| FastNoise2  | 704 M/s   | 419 M/s |

|   Cellular  | 2D AVX2 | 3D AVX2  |
|-------------|---------|----------|
| Quick-Noise | 570 M/s | 101 M/s  |
| FastNoise2  | 156 M/s | 46.5 M/s |

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
