//! Bakes cubes for static hosting: each cube's max and mean pyramids as compressed cubelets, after an index of them,
//! in `CUBE.fits.cubelets`, which the viewer reads by range request.

use std::{fs::File, path::PathBuf, time::Instant};

use fits_core::{
    baked::{self, Index},
    cache, read_fits,
};

const USAGE: &str = "usage: fits-bake [--sigmas S] [--exact-level-0] [--out DIR] CUBE.fits...
  --sigmas S        what the cubelets may be off by, in units of the cube's noise σ (default: 1, as the viewer's)
  --exact-level-0   keeps level 0 exact, and with it the cursor's values
  --out DIR         where the .cubelets files go (default: beside each cube)";

fn main() {
    if let Err(e) = bake() {
        eprintln!("{e}");
        std::process::exit(2);
    }
}

fn bake() -> Result<(), String> {
    let (mut sigmas, mut exact, mut out, mut cubes) = (1.0, false, None::<PathBuf>, Vec::new());
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or(USAGE);
        match arg.as_str() {
            "--sigmas" => sigmas = value()?.parse().map_err(|_| USAGE)?,
            "--exact-level-0" => exact = true,
            "--out" => out = Some(value()?.into()),
            _ if arg.starts_with("--") => return Err(USAGE.into()),
            _ => cubes.push(PathBuf::from(arg)),
        }
    }
    if cubes.is_empty() {
        return Err(USAGE.into());
    }
    for path in cubes {
        let start = Instant::now();
        let file = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let size = file.metadata().map_err(|e| e.to_string())?.len();
        let cube = read_fits(&file, None)?;
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        // What the file is, by its name, size and header.
        let content_id = format!(
            "{name}:{size}:{:016x}",
            cache::fnv(cube.info.cards.concat().as_bytes())
        );
        let bytes = baked::bake(cube, content_id, sigmas, exact)?;
        let index = Index::read(&bytes, Some(bytes.len() as u64))?;
        let level_0: u64 = (index.cubelets.iter())
            .filter(|(id, ..)| id.level == 0)
            .map(|&(_, _, len)| u64::from(len))
            .sum();
        let dir = out
            .clone()
            .or_else(|| path.parent().map(Into::into))
            .unwrap_or_default();
        let baked = dir.join(format!("{name}.cubelets"));
        std::fs::write(&baked, &bytes).map_err(|e| format!("{}: {e}", baked.display()))?;
        println!(
            "{}: {} cubelets, {:.1} MB ({:.1} MB of it level 0), {:.1}x smaller than the cube, in {:.1} s",
            baked.display(),
            index.cubelets.len(),
            bytes.len() as f64 / 1e6,
            level_0 as f64 / 1e6,
            size as f64 / bytes.len() as f64,
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}
