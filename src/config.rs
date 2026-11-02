use std::ops::RangeInclusive;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

/// iDaVIE's `[Range]` attributes on the material properties, shared by validation and the sliders.
pub const JITTER: RangeInclusive<f32> = 0.0..=1.0;
pub const RAYMARCHING_STEPS: RangeInclusive<u32> = 16..=512;

#[derive(serde::Deserialize, serde::Serialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub gpu_memory_limit_mb: u64,
    pub max_mode_downsampling: bool,
    pub max_raymarching_steps: u32,
    pub default_color_map: String,
    pub default_scaling_type: String,
    pub jitter: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            gpu_memory_limit_mb: 384,
            max_mode_downsampling: true,
            max_raymarching_steps: 384,
            default_color_map: "Inferno".into(),
            default_scaling_type: "Linear".into(),
            jitter: 1.0,
        }
    }
}

#[rustfmt::skip]
pub const COLOR_MAPS: [&str; 80] = [
    "Accent", "Afmhot", "Autumn", "Binary", "Blues", "Bone", "BrBg", "Brg", "BuGn", "BuPu", "Bwr", "CmRmap", "Cool", "Coolwarm",
    "Copper", "Cubehelix", "Dark2", "Flag", "GistEarth", "GistGray", "GistHeat", "GistNcar", "GistRainbow", "GistStern", "GistYarg",
    "GnBu", "Gnuplot", "Gnuplot2", "Gray", "Greens", "Greys", "Hot", "Hsv", "Inferno", "Jet", "Magma", "NipySpectral", "Ocean", "Oranges",
    "OrRd", "Paired", "Pastel1", "Pastel2", "Pink", "PiYg", "Plasma", "PrGn", "Prism", "PuBu", "PuBuGn", "PuOr", "PuRd", "Purples", "Rainbow",
    "RdBu", "RdGy", "RdPu", "RdYlBu", "RdYlGn", "Reds", "Seismic", "Set1", "Set2", "Set3", "Spectral", "Spring", "Summer", "Tab10", "Tab20",
    "Tab20B", "Tab20C", "Terrain", "Viridis", "Winter", "Wistia", "YlGn", "YlGnBu", "YlOrBr", "YlOrRd", "Turbo",
];
pub const SCALING_TYPES: [&str; 6] = ["Linear", "Log", "Sqrt", "Square", "Power", "Gamma"];

fn find(names: &[&str], name: &str) -> Option<usize> {
    names.iter().position(|n| n.eq_ignore_ascii_case(name))
}

impl Config {
    pub fn color_map_index(&self) -> usize {
        find(&COLOR_MAPS, &self.default_color_map).expect("validated by from_toml")
    }

    pub fn scaling_type_index(&self) -> u32 {
        find(&SCALING_TYPES, &self.default_scaling_type).expect("validated by from_toml") as u32
    }
}

/// Parse TOML text (None = defaults), then apply overrides, then validate.
pub fn from_toml(text: Option<&str>, overrides: &[(String, String)]) -> Result<Config, String> {
    let mut table: toml::Table = text
        .unwrap_or("")
        .parse()
        .map_err(|e| format!("invalid TOML: {e}"))?;
    for (key, value) in overrides {
        let parsed = value
            .parse()
            .unwrap_or_else(|_| toml::Value::String(value.clone()));
        table.insert(key.clone(), parsed);
    }
    let config: Config = table
        .try_into()
        .map_err(|e| e.to_string().trim().to_string())?;

    for (key, name, names) in [
        (
            "default_color_map",
            &config.default_color_map,
            &COLOR_MAPS[..],
        ),
        (
            "default_scaling_type",
            &config.default_scaling_type,
            &SCALING_TYPES[..],
        ),
    ] {
        if find(names, name).is_none() {
            return Err(format!(
                "unknown {key} '{name}', expected one of: {}",
                names.join(", ")
            ));
        }
    }
    if !RAYMARCHING_STEPS.contains(&config.max_raymarching_steps) {
        return Err(format!(
            "max_raymarching_steps must be in {RAYMARCHING_STEPS:?}, got {}",
            config.max_raymarching_steps
        ));
    }
    if !JITTER.contains(&config.jitter) {
        return Err(format!(
            "jitter must be in {JITTER:?}, got {}",
            config.jitter
        ));
    }
    if config.gpu_memory_limit_mb == 0 {
        return Err("gpu_memory_limit_mb must be > 0".into());
    }
    Ok(config)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn default_path() -> Option<PathBuf> {
    let var = |name| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let base = var("XDG_CONFIG_HOME")
        .or_else(|| var("HOME").map(|home| home.join(".config")))
        .or_else(|| var("APPDATA"))?;
    Some(base.join("idavie-bevy").join("config.toml"))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load(path: &Path, overrides: &[(String, String)]) -> Result<Config, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => Some(text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let text = format!(
                "# Any key can be overridden on the command line as --key value, e.g. --gpu-memory-limit-mb 512\n\n{}",
                toml::to_string(&Config::default()).unwrap()
            );
            match path
                .parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|_| std::fs::write(path, text))
            {
                Ok(()) => println!("Wrote default config to {}", path.display()),
                Err(e) => eprintln!("Could not write default config to {}: {e}", path.display()),
            }
            None
        }
        Err(e) => return Err(e.to_string()),
    };
    from_toml(text.as_deref(), overrides)
}

#[cfg(not(target_arch = "wasm32"))]
pub struct Args {
    pub file: PathBuf,
    pub config: Option<PathBuf>,
    pub hdu: Option<usize>,
    pub overrides: Vec<(String, String)>,
}

#[cfg(not(target_arch = "wasm32"))]
pub const USAGE: &str = concat!(
    "usage: ",
    env!("CARGO_PKG_NAME"),
    " [--config PATH] [--hdu N] [--<config-key> VALUE]... CUBE.fits\n",
    "  --config PATH       config file (default: $XDG_CONFIG_HOME/idavie-bevy/config.toml)\n",
    "  --hdu N             HDU index, 0 = primary (default: first image with NAXIS >= 3)\n",
    "  --<config-key> V    override any config key, e.g. --gpu-memory-limit-mb 512"
);

/// `Ok(None)` when help was asked for.
#[cfg(not(target_arch = "wasm32"))]
pub fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Option<Args>, String> {
    let (mut file, mut config, mut hdu, mut overrides) = (None, None, None, Vec::new());
    while let Some(arg) = args.next() {
        if arg == "-h" || arg == "--help" {
            return Ok(None);
        }
        let Some(key) = arg.strip_prefix("--") else {
            if file.replace(PathBuf::from(arg)).is_some() {
                return Err(USAGE.into());
            }
            continue;
        };
        let value = args
            .next()
            .ok_or_else(|| format!("--{key} needs a value\n{USAGE}"))?;
        match key {
            "config" => config = Some(PathBuf::from(value)),
            "hdu" => {
                hdu = Some(
                    value
                        .parse()
                        .map_err(|_| format!("--hdu needs a number, got '{value}'"))?,
                )
            }
            _ => overrides.push((key.replace('-', "_"), value)),
        }
    }
    Ok(Some(Args {
        file: file.ok_or(USAGE)?,
        config,
        hdu,
        overrides,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_apply_on_top_of_toml_and_unknown_keys_fail() {
        let args = parse_args(
            [
                "cube.fits",
                "--gpu-memory-limit-mb",
                "512",
                "--default-color-map",
                "viridis",
                "--max-mode-downsampling",
                "false",
            ]
            .map(String::from)
            .into_iter(),
        )
        .unwrap()
        .unwrap();
        let config = from_toml(
            Some("jitter = 0.5\ndefault_scaling_type = \"Sqrt\""),
            &args.overrides,
        )
        .unwrap();
        assert_eq!(
            (
                config.gpu_memory_limit_mb,
                config.max_mode_downsampling,
                config.jitter
            ),
            (512, false, 0.5)
        );
        assert_eq!(
            (config.color_map_index(), config.scaling_type_index()),
            (72, 2)
        );

        let err = from_toml(None, &[("max_steps".into(), "100".into())]).unwrap_err();
        assert!(err.contains("max_steps"), "{err}");
    }
}
