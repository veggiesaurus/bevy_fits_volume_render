//! A cubelet's voxels for CUBELET_DATA: zfp in fixed-accuracy mode, with the non-finite voxels filled beforehand and
//! their runs sent apart as CARTA's tiles' are; or float32 as they are. A cubelet with no finite voxels is just
//! that count, and comes back all NaN.

use std::borrow::Cow;

use zfp_rs::{ZfpBitStream, ZfpConfig, ZfpField, ZfpFieldMut, ZfpHeaderMask, ZfpScalarType};

use crate::{
    CUBELET, Cell, Cubelet, Voxels,
    carta::proto::{CompressionType, CubeletData},
};

/// `cubelet` as a message, its address left for the caller: within `tolerance` of its voxels when there's one, and
/// exact otherwise.
pub fn encode(cubelet: &Cubelet, tolerance: Option<f64>) -> CubeletData {
    let mut message = CubeletData {
        dims: cubelet.dims.map(|d| d as i32).to_vec(),
        cell_max: cubelet.cell.max,
        cell_has_nan: cubelet.cell.has_nan,
        ..Default::default()
    };
    let n = cubelet.dims.iter().product();
    let voxels = match &cubelet.voxels {
        Voxels::AllNan => return message,
        Voxels::Constant(value) => Cow::Owned(vec![*value; n]),
        Voxels::Data(voxels) => Cow::Borrowed(voxels),
    };
    message.finite_count = voxels.iter().filter(|v| v.is_finite()).count() as i32;
    if message.finite_count == 0 {
        return message;
    }
    match tolerance {
        None => message.image_data = voxels.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Some(tolerance) => {
            message.compression_type = CompressionType::ZfpExplicit.into();
            let runs = nan_runs(&voxels);
            let filled = if runs.is_empty() {
                voxels
            } else {
                message.nan_encodings = runs.iter().flat_map(|r| r.to_le_bytes()).collect();
                Cow::Owned(filled(&voxels, cubelet.dims))
            };
            message.image_data = zfp(&filled, cubelet.dims, tolerance);
        }
    }
    message
}

/// The cubelet in `message`, checked against its stated size, so that a bad message is an error rather than a panic
/// or a huge allocation.
pub fn decode(message: &CubeletData) -> Result<Cubelet, String> {
    let dims: [usize; 3] = match message.dims[..] {
        [x, y, z] if [x, y, z].iter().all(|d| (1..=CUBELET as i32).contains(d)) => {
            [x, y, z].map(|d| d as usize)
        }
        _ => return Err(format!("a cubelet of {:?}", message.dims)),
    };
    let n: usize = dims.iter().product();
    let cell = Cell {
        max: message.cell_max,
        has_nan: message.cell_has_nan,
    };
    let finite = usize::try_from(message.finite_count)
        .ok()
        .filter(|&finite| finite <= n)
        .ok_or_else(|| format!("{} finite of {n} voxels", message.finite_count))?;
    if finite == 0 {
        let voxels = Voxels::AllNan;
        return Ok(Cubelet { dims, voxels, cell });
    }
    let voxels = match message.compression_type {
        t if t == CompressionType::None as i32 => {
            let bytes = &message.image_data;
            if bytes.len() != 4 * n {
                return Err(format!("{} bytes for {n} voxels", bytes.len()));
            }
            let floats = bytes.as_chunks::<4>().0.iter();
            floats.map(|b| f32::from_le_bytes(*b)).collect()
        }
        t if t == CompressionType::ZfpExplicit as i32 => {
            let mut voxels = unzfp(&message.image_data, dims)?;
            restore_nans(&mut voxels, &message.nan_encodings)?;
            voxels
        }
        t => return Err(format!("compression type {t}")),
    };
    let counted = voxels.iter().filter(|v| v.is_finite()).count();
    if counted != finite {
        return Err(format!("{counted} finite voxels, not the {finite} stated"));
    }
    let voxels = Voxels::Data(voxels);
    Ok(Cubelet { dims, voxels, cell })
}

/// Alternately finite and not, starting with finite, x fastest; none when every voxel is finite.
fn nan_runs(voxels: &[f32]) -> Vec<i32> {
    let (mut runs, mut finite, mut run) = (Vec::new(), true, 0);
    for v in voxels {
        if v.is_finite() != finite {
            runs.push(run);
            (finite, run) = (!finite, 0);
        }
        run += 1;
    }
    runs.push(run);
    if runs.len() == 1 {
        runs.clear();
    }
    runs
}

fn restore_nans(voxels: &mut [f32], encoded: &[u8]) -> Result<(), String> {
    if encoded.is_empty() {
        return Ok(());
    }
    let (runs, rest) = encoded.as_chunks::<4>();
    let mut at = 0usize;
    for (i, run) in runs.iter().enumerate() {
        let end = usize::try_from(i32::from_le_bytes(*run))
            .ok()
            .and_then(|run| at.checked_add(run))
            .filter(|&end| end <= voxels.len())
            .ok_or("NaN runs past the cubelet")?;
        if i % 2 == 1 {
            voxels[at..end].fill(f32::NAN);
        }
        at = end;
    }
    if at != voxels.len() || !rest.is_empty() {
        return Err("NaN runs short of the cubelet".into());
    }
    Ok(())
}

/// Each non-finite voxel replaced by the mean of the finite ones in its zfp block of 4³, or by zero when there are
/// none, since zfp's lossy modes can't take them and a block's mean costs the fewest bits.
fn filled(voxels: &[f32], [nx, ny, nz]: [usize; 3]) -> Vec<f32> {
    let mut out = voxels.to_vec();
    for bz in (0..nz).step_by(4) {
        for by in (0..ny).step_by(4) {
            for bx in (0..nx).step_by(4) {
                let block = || {
                    (bz..(bz + 4).min(nz)).flat_map(move |z| {
                        (by..(by + 4).min(ny)).flat_map(move |y| {
                            (bx..(bx + 4).min(nx)).map(move |x| x + nx * (y + ny * z))
                        })
                    })
                };
                let finite = block().map(|i| voxels[i]).filter(|v| v.is_finite());
                let (sum, count) = finite.fold((0.0, 0), |(sum, n), v| (sum + f64::from(v), n + 1));
                let fill = if count == 0 {
                    0.0
                } else {
                    (sum / f64::from(count)) as f32
                };
                block()
                    .filter(|&i| !voxels[i].is_finite())
                    .for_each(|i| out[i] = fill);
            }
        }
    }
    out
}

fn zfp(voxels: &[f32], dims: [usize; 3], tolerance: f64) -> Vec<u8> {
    let encode = || -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let field = ZfpField::new(voxels, dims)?;
        let config = ZfpConfig::fixed_accuracy(tolerance);
        let capacity = config
            .maximum_size(ZfpScalarType::F32, dims)
            .ok_or("too big")?;
        let mut stream = ZfpBitStream::new(capacity)?;
        stream.write_header(&config, &field.metadata(), ZfpHeaderMask::FULL)?;
        stream.compress(&config, &field)?;
        Ok(stream.into_bytes()?)
    };
    // The field is a cubelet, finite, and the tolerance checked by the server, so nothing here can fail.
    encode().expect("zfp compression")
}

fn unzfp(bytes: &[u8], dims: [usize; 3]) -> Result<Vec<f32>, String> {
    let error = |e: &dyn std::fmt::Display| format!("zfp: {e}");
    let mut stream = ZfpBitStream::from_bytes(bytes).map_err(|e| error(&e))?;
    let header = stream
        .read_header(ZfpHeaderMask::FULL)
        .map_err(|e| error(&e))?;
    let (Some(metadata), Some(config)) = (header.metadata, header.config) else {
        return Err("zfp: no header".into());
    };
    if metadata.scalar_type != ZfpScalarType::F32 || metadata.dims != [dims[0], dims[1], dims[2], 0]
    {
        return Err(format!(
            "zfp: a stream of {:?} for a cubelet of {dims:?}",
            metadata.dims
        ));
    }
    let mut voxels = vec![0.0; dims.iter().product()];
    let mut field = ZfpFieldMut::new(&mut voxels, dims).map_err(|e| error(&e))?;
    stream
        .decompress(&config, &mut field)
        .map_err(|e| error(&e))?;
    Ok(voxels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lossy_cubelet_comes_back_within_its_tolerance_with_its_nans_where_they_were() {
        // Cut short at the cube's edge, with a ragged patch of NaN and an infinity.
        let dims = [64, 37, 5];
        let voxels: Vec<f32> = (0..64 * 37 * 5)
            .map(|i| match i {
                i if i % 64 < 9 && (i / 64) % 37 < 6 => f32::NAN,
                1000 => f32::INFINITY,
                i => ((i * 7919) % 1013) as f32 * 0.01 - 3.0,
            })
            .collect();
        let cell = Cell {
            max: 7.0,
            has_nan: true,
        };
        let cubelet = Cubelet {
            dims,
            voxels: Voxels::Data(voxels.clone()),
            cell,
        };
        for tolerance in [0.05, 0.5] {
            let message = encode(&cubelet, Some(tolerance));
            assert!(message.image_data.len() < voxels.len() * 4);
            let back = decode(&message).unwrap();
            assert_eq!((back.dims, back.cell), (dims, cell));
            let Voxels::Data(back) = back.voxels else {
                panic!("no voxels")
            };
            for (&v, &b) in voxels.iter().zip(&back) {
                assert!(if v.is_finite() {
                    (v - b).abs() <= tolerance as f32
                } else {
                    b.is_nan()
                });
            }
        }
        // Without a tolerance, exactly as they were.
        let back = decode(&encode(&cubelet, None)).unwrap().voxels;
        assert_eq!(format!("{back:?}"), format!("{:?}", cubelet.voxels));

        // A message that doesn't add up is an error.
        let mut message = encode(&cubelet, Some(0.5));
        message.dims[1] = 38;
        assert!(decode(&message).is_err());
        let mut message = encode(&cubelet, Some(0.5));
        message.nan_encodings = encode_runs(&[5, 10]);
        assert!(decode(&message).is_err());
        let mut message = encode(&cubelet, None);
        message.finite_count += 1;
        assert!(decode(&message).is_err());

        // A constant cubelet goes as any other, and one with nothing finite as its count alone.
        let constant = |voxels| Cubelet { dims, voxels, cell };
        for tolerance in [None, Some(0.5)] {
            let back = decode(&encode(&constant(Voxels::Constant(7.0)), tolerance)).unwrap();
            let Voxels::Data(back) = back.voxels else {
                panic!("no voxels")
            };
            assert!(back.iter().all(|&v| (v - 7.0).abs() <= 0.5));
            let message = encode(&constant(Voxels::AllNan), tolerance);
            assert!(message.image_data.is_empty());
            assert_eq!(decode(&message).unwrap().voxels, Voxels::AllNan);
        }
    }

    fn encode_runs(runs: &[i32]) -> Vec<u8> {
        runs.iter().flat_map(|r| r.to_le_bytes()).collect()
    }
}
