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
    pub max_raymarching_steps: u32,
    pub default_color_map: ColorMap,
    pub default_scaling_type: Scaling,
    pub jitter: f32,
    /// Every voxel along each ray rather than fixed steps.
    pub high_precision: bool,
    pub threshold_min: f32,
    pub threshold_max: f32,
    pub z_scale: ZScale,
    /// What a server's cubelets may be off by, in units of the cube's noise σ: one of `TOLERANCES`.
    pub zfp_tolerance_sigmas: f64,
    /// How much of a server's cubelets, compressed, are kept in memory to be shown again without asking for them. Less
    /// in the browser, whose worker shares its memory with any file it reads, and which usually keeps them in OPFS too.
    pub cache_memory_mb: u64,
    /// Gone, now that each projection mode reads its own pyramid, but still read so that older files parse.
    #[serde(skip_serializing)]
    max_mode_downsampling: serde::de::IgnoredAny,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            gpu_memory_limit_mb: 384,
            max_raymarching_steps: 384,
            default_color_map: ColorMap::try_from("Inferno".to_string()).unwrap(),
            default_scaling_type: Scaling::try_from("Linear".to_string()).unwrap(),
            jitter: 1.0,
            high_precision: false,
            threshold_min: 0.0,
            threshold_max: 1.0,
            z_scale: ZScale::Cube,
            zfp_tolerance_sigmas: 1.0,
            cache_memory_mb: if cfg!(target_arch = "wasm32") {
                256
            } else {
                1024
            },
            max_mode_downsampling: serde::de::IgnoredAny,
        }
    }
}

/// The z scale a cube opens at: "cube" for as deep as it's wide, "voxels" for cubic voxels, or a number relative to
/// cubic voxels.
#[derive(serde::Deserialize, serde::Serialize, Clone, Copy, Debug, PartialEq)]
#[serde(try_from = "toml::Value", into = "toml::Value")]
pub enum ZScale {
    Cube,
    Voxels,
    Factor(f32),
}

impl TryFrom<toml::Value> for ZScale {
    type Error = String;
    fn try_from(value: toml::Value) -> Result<Self, String> {
        let factor = match &value {
            toml::Value::String(s) if s.eq_ignore_ascii_case("cube") => return Ok(Self::Cube),
            toml::Value::String(s) if s.eq_ignore_ascii_case("voxels") => return Ok(Self::Voxels),
            toml::Value::Float(f) => *f as f32,
            toml::Value::Integer(i) => *i as f32,
            _ => f32::NAN,
        };
        if factor > 0.0 && factor.is_finite() {
            Ok(Self::Factor(factor))
        } else {
            Err(format!(
                "z_scale must be \"cube\", \"voxels\" or a number above 0, got {value}"
            ))
        }
    }
}

impl From<ZScale> for toml::Value {
    fn from(scale: ZScale) -> Self {
        match scale {
            ZScale::Cube => "cube".into(),
            ZScale::Voxels => "voxels".into(),
            ZScale::Factor(f) => f64::from(f).into(),
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
/// The zfp tolerances a server's cubelets can come at, in σ.
pub const TOLERANCES: [f64; 3] = [0.5, 1.0, 2.0];
pub const SCALING_TYPES: [&str; 7] = ["Linear", "Log", "Sqrt", "Square", "Power", "Gamma", "Asinh"];
/// The index of Gamma in `SCALING_TYPES`, as volume.wesl's GAMMA.
pub const GAMMA: u32 = 5;
/// Asinh's, which volume.wesl compiles as a pipeline of its own.
pub const ASINH: u32 = 6;

/// A name from `COLOR_MAPS`, matched case-insensitively and held as its row in the atlas.
#[derive(serde::Deserialize, serde::Serialize, Clone, Copy, Debug, PartialEq)]
#[serde(try_from = "String", into = "String")]
pub struct ColorMap(pub u32);

/// A name from `SCALING_TYPES`, matched case-insensitively and held as the shader's scale type.
#[derive(serde::Deserialize, serde::Serialize, Clone, Copy, Debug, PartialEq)]
#[serde(try_from = "String", into = "String")]
pub struct Scaling(pub u32);

fn index(kind: &str, names: &[&str], name: &str) -> Result<u32, String> {
    names
        .iter()
        .position(|n| n.eq_ignore_ascii_case(name))
        .map(|i| i as u32)
        .ok_or_else(|| {
            format!(
                "unknown {kind} '{name}', expected one of: {}",
                names.join(", ")
            )
        })
}

impl TryFrom<String> for ColorMap {
    type Error = String;
    fn try_from(name: String) -> Result<Self, String> {
        index("colour map", &COLOR_MAPS, &name).map(Self)
    }
}

impl From<ColorMap> for String {
    fn from(map: ColorMap) -> Self {
        COLOR_MAPS[map.0 as usize].into()
    }
}

impl TryFrom<String> for Scaling {
    type Error = String;
    fn try_from(name: String) -> Result<Self, String> {
        index("scaling type", &SCALING_TYPES, &name).map(Self)
    }
}

impl From<Scaling> for String {
    fn from(scaling: Scaling) -> Self {
        SCALING_TYPES[scaling.0 as usize].into()
    }
}

/// Parse TOML text (None = defaults), then apply overrides (whose keys may use - for _), then validate.
pub fn from_toml(text: Option<&str>, overrides: &[(String, String)]) -> Result<Config, String> {
    let mut table: toml::Table = text
        .unwrap_or("")
        .parse()
        .map_err(|e| format!("invalid TOML: {e}"))?;
    for (key, value) in overrides {
        let parsed = value
            .parse()
            .unwrap_or_else(|_| toml::Value::String(value.clone()));
        table.insert(key.replace('-', "_"), parsed);
    }
    let config: Config = table
        .try_into()
        .map_err(|e| e.to_string().trim().to_string())?;
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
    let (min, max) = (config.threshold_min, config.threshold_max);
    if !(0.0 <= min && min <= max && max <= 1.0) {
        return Err(format!(
            "threshold_min and threshold_max must satisfy 0 <= min <= max <= 1, got {min} and {max}"
        ));
    }
    if !TOLERANCES.contains(&config.zfp_tolerance_sigmas) {
        return Err(format!(
            "zfp_tolerance_sigmas must be one of {TOLERANCES:?}, got {}",
            config.zfp_tolerance_sigmas
        ));
    }
    if config.gpu_memory_limit_mb == 0 {
        return Err("gpu_memory_limit_mb must be > 0".into());
    }
    Ok(config)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn default_path() -> Option<PathBuf> {
    Some(user_dir("XDG_CONFIG_HOME", ".config", "APPDATA")?.join("config.toml"))
}

/// Where a server's cubelets are kept on disk.
#[cfg(not(target_arch = "wasm32"))]
pub fn cache_dir() -> Option<PathBuf> {
    Some(user_dir("XDG_CACHE_HOME", ".cache", "LOCALAPPDATA")?.join("cubelets"))
}

/// The viewer's folder under the XDG one, or the one in `$HOME` it defaults to, or on Windows the one in `windows`.
#[cfg(not(target_arch = "wasm32"))]
fn user_dir(xdg: &str, home: &str, windows: &str) -> Option<PathBuf> {
    let var = |name| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let base = var(xdg)
        .or_else(|| var("HOME").map(|dir| dir.join(home)))
        .or_else(|| var(windows))?;
    Some(base.join("idavie-bevy"))
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
    /// A local file, or with `server` a path under the server's root.
    pub file: Option<PathBuf>,
    pub server: Option<String>,
    pub config: Option<PathBuf>,
    pub hdu: Option<usize>,
    /// The plane of a local file's Stokes axis.
    pub stokes: Option<usize>,
    pub overrides: Vec<(String, String)>,
}

#[cfg(not(target_arch = "wasm32"))]
pub const USAGE: &str = concat!(
    "usage: ",
    env!("CARGO_PKG_NAME"),
    " [--config PATH] [--hdu N] [--stokes N] [--<config-key> VALUE]... CUBE.fits\n",
    "       ",
    env!("CARGO_PKG_NAME"),
    " --server URL [OPTIONS]... [PATH]\n",
    "  --server URL        a fits-server's ws://host:port/?token=... URL, to browse its files or open PATH under its root\n",
    "  --config PATH       config file (default: $XDG_CONFIG_HOME/idavie-bevy/config.toml)\n",
    "  --hdu N             HDU index, 0 = primary (default: first image with NAXIS >= 3)\n",
    "  --stokes N          plane of a local file's Stokes axis, 0 = the first (default: 0)\n",
    "  --<config-key> V    override any config key, e.g. --gpu-memory-limit-mb 512"
);

/// `Ok(None)` when help was asked for.
#[cfg(not(target_arch = "wasm32"))]
pub fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Option<Args>, String> {
    let (mut file, mut config, mut hdu, mut overrides) = (None, None, None, Vec::new());
    let mut stokes = None;
    let mut server = None;
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
            "server" => server = Some(value),
            "hdu" => hdu = Some(crate::fits::parse_hdu(&value)?),
            "stokes" => {
                let plane = value
                    .parse()
                    .map_err(|_| format!("--stokes needs a number, got '{value}'"));
                stokes = Some(plane?)
            }
            _ => overrides.push((key.to_string(), value)),
        }
    }
    if file.is_none() && server.is_none() {
        return Err(USAGE.into());
    }
    if server.is_some() && stokes.is_some() {
        return Err("--stokes is for a local file".into());
    }
    Ok(Some(Args {
        file,
        server,
        config,
        hdu,
        stokes,
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
                "--threshold-max",
                "0.75",
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
                config.threshold_max,
                config.jitter
            ),
            (512, 0.75, 0.5)
        );
        assert_eq!(
            (config.default_color_map, config.default_scaling_type),
            (ColorMap(72), Scaling(2))
        );

        let err = from_toml(None, &[("max_steps".into(), "100".into())]).unwrap_err();
        assert!(err.contains("max_steps"), "{err}");
        // A key that's gone, which the default file used to have.
        let config = from_toml(Some("max_mode_downsampling = true"), &[]).unwrap();
        assert!(!toml::to_string(&config).unwrap().contains("max_mode"));

        // A preset by name, or a number, which an override of "2" makes an integer.
        let written = toml::to_string(&Config::default()).unwrap();
        assert_eq!(
            from_toml(Some(&written), &[]).unwrap().z_scale,
            ZScale::Cube
        );
        let z_scale =
            |value: &str| from_toml(None, &[("z-scale".into(), value.into())]).map(|c| c.z_scale);
        assert_eq!(z_scale("Voxels"), Ok(ZScale::Voxels));
        assert_eq!(z_scale("2"), Ok(ZScale::Factor(2.0)));
        assert_eq!(z_scale("0.25"), Ok(ZScale::Factor(0.25)));
        assert!(z_scale("0").is_err() && z_scale("flat").is_err());
    }
}
