use std::{fs, path::Path};

use crate::noise::perlin::*;

pub fn write_perlin_height_map(
    path: impl AsRef<Path>,
    dimension: usize,
    octaves: u32,
    scale: f32,
    lacunarity: f32,
    persistence: f32,
) {
    if let Some(parent) = path.as_ref().parent()
        && !parent.exists()
    {
        fs::create_dir_all(parent).expect("Failed to create parent");
    }

    let mut perlin = Perlin::new(0);

    let mut pixels = Vec::<u8>::new();
    pixels.resize(dimension * dimension * MAP_SIZE, 0);

    for x in 0..dimension {
        let x_offset = x * dimension * MAP_SIZE;
        for y in 0..dimension {
            let y_offset = y * ROW_SIZE;

            let mut noise: PerlinMap = perlin.noise_2d(
                (x as i32, y as i32).into(),
                octaves,
                scale,
                1.0,
                lacunarity,
                persistence,
                1,
                0.0,
            );

            noise = (noise + PerlinMap::new(1.0)) * PerlinMap::new(127.5);

            for dx in 0..ROW_SIZE {
                let offset = x_offset + y_offset + dx * ROW_SIZE * dimension;
                for dy in 0..ROW_SIZE {
                    pixels[offset + dy] = noise[dx * ROW_SIZE + dy] as u8;
                }
            }
        }
    }

    let pixel_dimension = (dimension * ROW_SIZE) as u32;
    image::save_buffer(
        &path,
        &pixels,
        pixel_dimension,
        pixel_dimension,
        image::ColorType::L8,
    )
    .expect("Failed to write height map!");

    println!("Wrote height map to {}!", path.as_ref().display());
}

pub fn write_perlin_octaves_height_map(
    path: impl AsRef<Path>,
    dimension: usize,
    octaves: impl IntoIterator<Item = impl Into<Octave2D>>,
    channel: i32,
) {
    if let Some(parent) = path.as_ref().parent()
        && !parent.exists()
    {
        fs::create_dir_all(parent).expect("Failed to create parent");
    }

    let mut perlin = Perlin::new(0);

    let octaves_vec: Vec<Octave2D> = octaves.into_iter().map(Into::into).collect();
    let mut pixels = Vec::<u8>::new();
    pixels.resize(dimension * dimension * MAP_SIZE, 0);

    for x in 0..dimension {
        let x_offset = x * dimension * MAP_SIZE;
        for y in 0..dimension {
            let y_offset = y * ROW_SIZE;

            let mut noise: PerlinMap = perlin.noise_2d_octaves(
                (x as i32, y as i32).into(),
                &octaves_vec,
                1.0,
                channel,
                0.0,
            );

            noise = (noise + PerlinMap::new(1.0)) * PerlinMap::new(127.5);

            for dx in 0..ROW_SIZE {
                let offset = x_offset + y_offset + dx * ROW_SIZE * dimension;
                for dy in 0..ROW_SIZE {
                    pixels[offset + dy] = noise[dx * ROW_SIZE + dy] as u8;
                }
            }
        }
    }

    let pixel_dimension = (dimension * ROW_SIZE) as u32;
    image::save_buffer(
        &path,
        &pixels,
        pixel_dimension,
        pixel_dimension,
        image::ColorType::L8,
    )
    .expect("Failed to write height map!");

    println!("Wrote height map to {}!", path.as_ref().display());
}
