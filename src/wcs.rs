use std::f64::consts::{PI, TAU};
use std::sync::Mutex;

use starlink_ast::constants::BAD_VALUE;
use starlink_ast::coords::PointSet;
use starlink_ast::frames::{DSBSpecFrame, FrameSet, SkyFrame, SpecFrame, SpecSystem};
use starlink_ast::prelude::*;
use starlink_ast::{AnyObject, FitsChan};

pub struct Wcs {
    // FrameSet is Send but not Sync, and a Bevy Resource needs both.
    frames: Mutex<FrameSet>,
    /// The FITS pixel axes (0-based) shown as the cube's x, y and z.
    axes: [usize; 3],
}

impl Wcs {
    /// None when the header has no WCS; an AST error reading it is logged.
    pub fn from_cards(cards: &[String], axes: [usize; 3]) -> Option<Wcs> {
        let read = || {
            let mut chan = FitsChan::new()?;
            // Commentary cards hold no WCS but dominate read time in long (e.g. CASA) headers. A keyword with
            // '=' in it ("OBJECT='M42'") would make AST reject the whole header.
            let header: String = cards
                .iter()
                .filter(|c| !matches!(c[..8].trim_end(), "COMMENT" | "HISTORY" | ""))
                .filter(|c| !c[..8].contains('='))
                .map(String::as_str)
                .collect();
            chan.put_cards(&header)?;
            chan.read()
        };
        let Some(AnyObject::FrameSet(mut fs)) = read()
            .map_err(|e| bevy::log::warn!("No WCS, AST could not read the header: {e}"))
            .ok()?
        else {
            return None;
        };
        if fs.nframe() <= 1 {
            return None;
        }
        for axis in 1..=fs.naxes() {
            let unit = match spec_frame(fs.current(), axis).map(SpecFrame::system) {
                Some(SpecSystem::Freq) => "MHz",
                Some(SpecSystem::Vradio | SpecSystem::Voptical | SpecSystem::Vrel) => "km/s",
                _ => continue,
            };
            // Setting Unit on the FrameSet remaps it, so transformed values come out in the new unit.
            let _ = fs.set_unit(axis, unit);
        }
        Some(Wcs {
            frames: Mutex::new(fs),
            axes,
        })
    }

    /// The GRID position of the cube's x, y and z `grid`, on the shown (first) plane of any other axes.
    fn grid_point(&self, nin: usize, grid: [f64; 3]) -> Vec<f64> {
        (0..nin)
            .map(|p| {
                self.axes
                    .iter()
                    .position(|&a| a == p)
                    .map_or(1.0, |i| grid[i])
            })
            .collect()
    }

    pub fn summary(&self) -> Vec<String> {
        let fs = self.frames.lock().unwrap();
        let frame = fs.current();
        let mut lines = vec![frame.title().unwrap_or_default()];
        for axis in 1..=frame.naxes() {
            let label = frame.label(axis).unwrap_or_default();
            lines.push(match display_unit(frame, axis) {
                Some(unit) => format!("{label} ({unit})"),
                None => label,
            });
            if let Some(spec) = spec_frame(frame, axis) {
                // Without SPECSYS this is only AST's default.
                if spec.test_std_of_rest() {
                    lines.push(format!("Standard of rest: {}", spec.std_of_rest().as_str()));
                }
                if spec.test_rest_freq() {
                    lines.push(format!("Rest frequency: {:.4} MHz", spec.rest_freq() / 1e6));
                }
            }
        }
        lines
    }

    pub fn grid_to_world(&self, grid: [f64; 3]) -> Option<Vec<String>> {
        let fs = self.frames.lock().unwrap();
        let point = self.grid_point(fs.nin(), grid);
        let mut world = fs
            .transform(&PointSet::from_rows([point]), true)
            .ok()?
            .point(0)
            .to_vec();
        if world.iter().any(|&v| v == BAD_VALUE || !v.is_finite()) {
            return None;
        }
        let frame = fs.current();
        frame.norm(&mut world).ok()?;
        (1..=frame.naxes())
            .map(|axis| {
                let value = frame.format(axis, world[axis - 1]).ok()?;
                Some(match display_unit(frame, axis) {
                    Some(unit) => format!("{value} {unit}"),
                    None => value,
                })
            })
            .collect()
    }

    /// "Declination", "Frequency (MHz)" and so on, for the cube's `cube_axis` (1-based x, y or z).
    pub fn axis_title(&self, cube_axis: usize) -> Option<String> {
        let axis = self.axes[cube_axis - 1] + 1;
        let fs = self.frames.lock().unwrap();
        let frame = fs.current();
        if axis > frame.naxes() {
            return None;
        }
        let label = frame.label(axis).ok()?;
        Some(match display_unit(frame, axis) {
            Some(unit) => format!("{label} ({unit})"),
            None => label,
        })
    }

    /// Ticks at round values of the world axis along the cube's `cube_axis` (1-based x, y or z), on the
    /// straight segment `from`..`to` in the cube's GRID coordinates. Stands in for AST's Plot3D grid, which
    /// the Rust port doesn't include.
    pub fn edge_ticks(&self, cube_axis: usize, from: [f64; 3], to: [f64; 3]) -> Vec<Tick> {
        const SAMPLES: usize = 64;
        let axis = self.axes[cube_axis - 1] + 1;
        let fs = self.frames.lock().unwrap();
        if axis > fs.naxes() {
            return Vec::new();
        }
        let grid = (0..SAMPLES).map(|s| {
            let t = s as f64 / (SAMPLES - 1) as f64;
            self.grid_point(
                fs.nin(),
                std::array::from_fn(|i| from[i] + (to[i] - from[i]) * t),
            )
        });
        let Ok(world) = fs.transform(&PointSet::from_rows(grid), true) else {
            return Vec::new();
        };
        let bad = |v: f64| v == BAD_VALUE || !v.is_finite();
        let mut values = world.axis(axis - 1).to_vec();
        let sky = sky_as_time(fs.current(), axis);
        // Longitudes wrap at 2pi; unwrap them so crossings are found on a continuous curve.
        for i in 1..values.len() {
            if sky.is_some() && !bad(values[i]) && !bad(values[i - 1]) {
                let jump = values[i] - values[i - 1];
                if jump.abs() > PI {
                    let shift = -TAU * jump.signum();
                    values[i..]
                        .iter_mut()
                        .filter(|v| !bad(**v))
                        .for_each(|v| *v += shift);
                }
            }
        }
        let (min, max) = values
            .iter()
            .filter(|v| !bad(**v))
            .fold((f64::MAX, f64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        if min >= max {
            return Vec::new();
        }

        // AST's Gap isn't ported either: step in seconds of time or arc for sky axes, 1-2-5 otherwise.
        let (unit, fields) = match sky {
            Some(true) => (TAU / 86400.0, Some("hms")),
            Some(false) => (PI / 648000.0, Some("dms")),
            None => (1.0, None),
        };
        let step = nice_step((max - min) / unit / 5.0, fields.is_some());
        let gap = step * unit;
        let decimals = (-step.log10()).ceil().max(0.0) as usize;
        let format = match fields {
            Some(f) if step < 1.0 => format!("{f}.{decimals}"),
            Some(f) if step >= 3600.0 => f[..1].to_string(),
            Some(f) if step >= 60.0 => f[..2].to_string(),
            Some(f) => f.to_string(),
            None => format!("%.{decimals}f"),
        };
        // Labels are formatted on a copy, so the tick Format never reaches the cursor readout.
        let mut frame = fs.current().snapshot();
        if frame.set_format_str(axis, &format).is_err() {
            return Vec::new();
        }

        let mut ticks: Vec<(i64, Tick)> = Vec::new();
        for (s, pair) in values.windows(2).enumerate() {
            let [w0, w1] = [pair[0], pair[1]];
            if bad(w0) || bad(w1) {
                continue;
            }
            for k in (w0.min(w1) / gap).ceil() as i64..=(w0.max(w1) / gap).floor() as i64 {
                if ticks.iter().any(|(j, _)| *j == k) {
                    continue;
                }
                let value = k as f64 * gap;
                let along = if w1 == w0 {
                    0.0
                } else {
                    (value - w0) / (w1 - w0)
                };
                // Transformed values aren't normalised (an RA can come out negative), so let AST do it.
                let mut point = world.point(s).to_vec();
                point[axis - 1] = value;
                let _ = frame.norm(&mut point);
                if let Ok(label) = frame.format(axis, point[axis - 1]) {
                    let at = ((s as f64 + along) / (SAMPLES - 1) as f64) as f32;
                    ticks.push((k, Tick { at, label }));
                }
            }
        }
        ticks.into_iter().map(|(_, tick)| tick).collect()
    }
}

pub struct Tick {
    /// Fraction of the way along the edge.
    pub at: f32,
    pub label: String,
}

fn nice_step(raw: f64, sexagesimal: bool) -> f64 {
    let decade = |raw: f64| {
        let p = 10f64.powf(raw.log10().floor());
        [1.0, 2.0, 5.0, 10.0]
            .into_iter()
            .map(|m| m * p)
            .find(|&s| s >= raw)
            .unwrap_or(10.0 * p)
    };
    if !sexagesimal || raw < 1.0 {
        return decade(raw);
    }
    [1.0, 60.0, 3600.0]
        .into_iter()
        .flat_map(|scale| [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0].map(|m| m * scale))
        // Degrees beyond 30, for edges spanning much of the sky.
        .chain([45.0, 60.0, 90.0, 180.0].map(|d| d * 3600.0))
        .find(|&s| s >= raw)
        .unwrap_or_else(|| decade(raw))
}

/// The SpecFrame behind a spectral axis, including the one inside the DSBSpecFrame FitsChan makes when
/// the header has IMAGFREQ.
fn spec_frame(frame: &dyn Frame, axis: usize) -> Option<&SpecFrame> {
    let primary = frame.primary_frame(axis - 1).0.as_any();
    primary.downcast_ref().or_else(|| {
        primary
            .downcast_ref::<DSBSpecFrame>()
            .map(DSBSpecFrame::spec)
    })
}

/// For a sky axis, whether it's shown as a time (RA) rather than an angle (Dec); None otherwise.
fn sky_as_time(frame: &dyn Frame, axis: usize) -> Option<bool> {
    let (primary, primary_axis) = frame.primary_frame(axis - 1);
    let sky: &SkyFrame = primary.as_any().downcast_ref()?;
    // primary_frame gives the internal axis, and as_time takes the permuted one (Dec first in some headers).
    let permuted = sky
        .axis_permutation()
        .iter()
        .position(|&a| a == primary_axis)?;
    sky.as_time(permuted + 1).ok()
}

/// The Unit worth printing after a value: a sky axis's "hh:mm:ss.s" describes its format, not a unit.
fn display_unit(frame: &dyn Frame, axis: usize) -> Option<String> {
    frame
        .unit(axis)
        .ok()
        .filter(|unit| !unit.is_empty() && !unit.contains(':'))
}

#[cfg(test)]
mod tests {
    use super::Wcs;

    // WCS cards of XMMLSS13.HI2L.ts.NGC895_large.fits; the expected values agree with astropy.wcs.
    const CARDS: [&str; 23] = [
        "NAXIS   = 3",
        "NAXIS1  = 540",
        "NAXIS2  = 540",
        "NAXIS3  = 21",
        "CTYPE1  = 'RA---SIN'",
        "CTYPE2  = 'DEC--SIN'",
        "CTYPE3  = 'FREQ'",
        "CRPIX1  = 677.0",
        "CRPIX2  = 1509.0",
        "CRPIX3  = -131.0",
        "CDELT1  = -0.00055555555543",
        "CDELT2  = 0.00055555555543",
        "CDELT3  = 208986.0516772",
        "CRVAL1  = 35.17499974853",
        "CRVAL2  = -4.833055419407",
        "CRVAL3  = 1380143885.278",
        "CUNIT1  = 'deg'",
        "CUNIT2  = 'deg'",
        "CUNIT3  = 'Hz'",
        "RADESYS = 'FK5'",
        "EQUINOX = 2000.0",
        "SPECSYS = 'LSRK'",
        "RESTFRQ = 1420405750.0",
    ];

    #[test]
    fn real_cube_centre() {
        let cards: Vec<String> = CARDS.iter().map(|c| format!("{c:<80}")).collect();
        assert!(Wcs::from_cards(&cards[..4], [0, 1, 2]).is_none());
        let wcs = Wcs::from_cards(&cards, [0, 1, 2]).unwrap();
        assert_eq!(
            wcs.grid_to_world([270.5, 270.5, 11.0]).unwrap(),
            ["2:21:36.5", "-5:31:16", "1409.82 MHz"]
        );
    }

    #[test]
    fn frequency_after_a_skipped_stokes_axis_is_z() {
        let cards: Vec<String> = [
            "NAXIS   = 4",
            "NAXIS1  = 2",
            "NAXIS2  = 2",
            "NAXIS3  = 1",
            "NAXIS4  = 4",
            "CTYPE1  = 'RA---SIN'",
            "CTYPE2  = 'DEC--SIN'",
            "CTYPE3  = 'STOKES'",
            "CTYPE4  = 'FREQ'",
            "CRVAL1  = 10.0",
            "CRVAL2  = -30.0",
            "CRVAL3  = 1.0",
            "CRVAL4  = 1.4E9",
            "CDELT1  = -0.001",
            "CDELT2  = 0.001",
            "CDELT3  = 1.0",
            "CDELT4  = 1.0E6",
            "CUNIT4  = 'Hz'",
            "CRPIX1  = 1.0",
            "CRPIX2  = 1.0",
            "CRPIX3  = 1.0",
            "CRPIX4  = 1.0",
        ]
        .iter()
        .map(|c| format!("{c:<80}"))
        .collect();
        let wcs = Wcs::from_cards(&cards, [0, 1, 3]).unwrap();
        assert_eq!(wcs.axis_title(3).as_deref(), Some("Frequency (MHz)"));
        assert_eq!(wcs.grid_to_world([1.0, 1.0, 3.0]).unwrap()[3], "1402 MHz");
        let ticks = wcs.edge_ticks(3, [0.5, 0.5, 0.5], [0.5, 0.5, 4.5]);
        let labels: Vec<&str> = ticks.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(labels, ["1400", "1401", "1402", "1403"]);
    }

    #[test]
    fn latitude_first_header_ticks_dec_as_an_angle() {
        let swap = |c: &&str| match &c[..5] {
            k @ ("CTYPE" | "CRPIX" | "CDELT" | "CRVAL" | "CUNIT") if c.as_bytes()[5] != b'3' => {
                let other = if c.as_bytes()[5] == b'1' { '2' } else { '1' };
                format!("{k}{other}{:<74}", &c[6..])
            }
            _ => format!("{c:<80}"),
        };
        let cards: Vec<String> = CARDS.iter().map(swap).collect();
        let wcs = Wcs::from_cards(&cards, [0, 1, 2]).unwrap();
        let dec = wcs.edge_ticks(1, [0.5, 0.5, 0.5], [540.5, 0.5, 0.5]);
        let ra = wcs.edge_ticks(2, [0.5, 0.5, 0.5], [0.5, 540.5, 0.5]);
        let labels = |t: &[super::Tick]| t.iter().map(|t| t.label.clone()).collect::<Vec<_>>();
        assert!(
            !dec.is_empty() && dec.iter().all(|t| t.label.starts_with("-5:")),
            "{:?}",
            labels(&dec)
        );
        assert!(
            !ra.is_empty() && ra.iter().all(|t| t.label.starts_with("2:2")),
            "{:?}",
            labels(&ra)
        );
    }

    #[test]
    fn edge_ticks_at_round_values() {
        let cards: Vec<String> = CARDS.iter().map(|c| format!("{c:<80}")).collect();
        let wcs = Wcs::from_cards(&cards, [0, 1, 2]).unwrap();
        // Channels span 1407.63 to 1412.01 MHz, so 1 MHz steps.
        let freq = wcs.edge_ticks(3, [0.5, 0.5, 0.5], [0.5, 0.5, 21.5]);
        let labels: Vec<&str> = freq.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(labels, ["1408", "1409", "1410", "1411", "1412"]);
        assert!((freq[0].at - 0.0853).abs() < 1e-3, "{}", freq[0].at);
        // 540 pixels of 2 arcsec is 18 arcmin of Dec, so 5 arcmin steps shown as d:m.
        let dec = wcs.edge_ticks(2, [0.5, 0.5, 0.5], [0.5, 540.5, 0.5]);
        assert!(
            dec.len() >= 3
                && dec
                    .iter()
                    .all(|t| t.label.starts_with("-5:") && t.label.ends_with(['0', '5'])),
            "{:?}",
            dec.iter().map(|t| &t.label).collect::<Vec<_>>()
        );
        // An RA range across 0h is labelled 0:01, 0:00, 23:59 rather than going negative.
        let at_zero: Vec<String> = [
            "NAXIS   = 2",
            "NAXIS1  = 100",
            "NAXIS2  = 100",
            "CTYPE1  = 'RA---TAN'",
            "CTYPE2  = 'DEC--TAN'",
            "CRPIX1  = 50.5",
            "CRPIX2  = 50.5",
            "CDELT1  = -0.01",
            "CDELT2  = 0.01",
            "CRVAL1  = 0.0",
            "CRVAL2  = 0.0",
        ]
        .iter()
        .map(|c| format!("{c:<80}"))
        .collect();
        let ra = Wcs::from_cards(&at_zero, [0, 1, 2]).unwrap().edge_ticks(
            1,
            [0.5, 50.5, 1.0],
            [100.5, 50.5, 1.0],
        );
        let labels: Vec<&str> = ra.iter().map(|t| t.label.as_str()).collect();
        assert!(
            labels.contains(&"0:00") && labels.contains(&"23:59"),
            "{labels:?}"
        );
        assert!(labels.iter().all(|l| !l.starts_with('-')), "{labels:?}");
        // The temporary tick format doesn't leak into the cursor readout.
        assert_eq!(
            wcs.grid_to_world([270.5, 270.5, 11.0]).unwrap()[2],
            "1409.82 MHz"
        );
    }
}
