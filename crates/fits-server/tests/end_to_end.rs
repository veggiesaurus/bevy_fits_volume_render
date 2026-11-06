//! The server on a free port, and the viewer's remote adapter talking to it.

use std::{
    path::{Path, PathBuf},
    sync::mpsc,
    time::Duration,
};

use fits_core::{
    self as fits, Cube, Voxels,
    carta::{self, cubelet_key, frame, proto},
    codec,
    engine::{CubeletId, Event, Mode, Request},
    remote::{Budget, Command, Connection, Opened, Received},
};
use proto::EventType;

const TOKEN: &str = "0123456789abcdef";
const BUDGET: Budget = Budget {
    gpu_memory_limit_mb: 384,
    max_depth: 2048,
};

/// A float32 FITS cube whose voxels are `value` of their place, with `cards` added to its header.
fn write_fits(path: &Path, dims: [usize; 3], cards: &[&str], value: impl Fn([usize; 3]) -> f32) {
    let mut bytes: Vec<u8> = ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3"]
        .into_iter()
        .zip([
            "T".into(),
            "-32".into(),
            "3".into(),
            dims[0].to_string(),
            dims[1].to_string(),
            dims[2].to_string(),
        ])
        .flat_map(|(key, value)| format!("{key:<8}= {value:>20}{:50}", "").into_bytes())
        .chain(
            (["BUNIT   = 'K       '"].iter().chain(cards))
                .flat_map(|c| format!("{c:<80}").into_bytes()),
        )
        .chain(format!("{:<80}", "END").into_bytes())
        .collect();
    bytes.resize(2880, b' ');
    for z in 0..dims[2] {
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                bytes.extend(value([x, y, z]).to_be_bytes());
            }
        }
    }
    bytes.resize(bytes.len().div_ceil(2880) * 2880, 0);
    std::fs::write(path, bytes).unwrap();
}

/// NaN in a corner, and constant past x = 128, so that level 0's cubelets at x = 2 are.
fn voxel([x, y, z]: [usize; 3]) -> f32 {
    match (x, y) {
        (..10, ..20) => f32::NAN,
        (128.., _) => 0.5,
        _ => ((x * 7 + y * 13 + z * 29) % 97) as f32 * 0.1 - 2.0,
    }
}

/// A cubelet's voxels, however it travelled.
fn values(cubelet: &fits::Cubelet) -> Vec<f32> {
    let n = cubelet.dims.iter().product();
    match &cubelet.voxels {
        Voxels::AllNan => vec![f32::NAN; n],
        Voxels::Constant(v) => vec![*v; n],
        Voxels::Data(v) => v.clone(),
    }
}

/// A root of its own for a test, with a cube and a folder in it, and a cube outside it with a link to it from inside.
fn root(test: &str) -> PathBuf {
    let base = std::env::temp_dir().join(format!("fits-server-{test}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let root = base.join("root");
    std::fs::create_dir_all(root.join("folder")).unwrap();
    std::fs::create_dir_all(root.join(".hidden")).unwrap();
    write_fits(&root.join("cube.fits"), [150, 70, 80], &[], voxel);
    write_fits(&base.join("outside.fits"), [4, 4, 4], &[], |_| 1.0);
    #[cfg(unix)]
    std::os::unix::fs::symlink(base.join("outside.fits"), root.join("link.fits")).unwrap();
    root
}

fn start(root: &Path) -> u16 {
    let (tx, rx) = mpsc::channel();
    let root = root.to_path_buf();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        runtime.block_on(async {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(listener.local_addr().unwrap().port()).unwrap();
            fits_server::serve(listener, &root, TOKEN.into(), None)
                .await
                .unwrap();
        });
    });
    rx.recv().unwrap()
}

/// With the token in the URL, as the native viewer has it.
fn connect(port: u16, token: &str) -> (Connection, mpsc::Receiver<Received>) {
    connect_with(&format!("ws://127.0.0.1:{port}/?token={token}"), "")
}

fn connect_with(url: &str, api_key: &str) -> (Connection, mpsc::Receiver<Received>) {
    let (tx, rx) = mpsc::channel();
    let connection = Connection::connect(url, api_key.into(), 1 << 30, None, move |received| {
        let _ = tx.send(received);
    });
    (connection, rx)
}

/// The next of what's received that `want` takes, failing on anything else it doesn't skip.
fn next<T>(rx: &mpsc::Receiver<Received>, mut want: impl FnMut(Received) -> Option<T>) -> T {
    loop {
        let received = rx
            .recv_timeout(Duration::from_secs(20))
            .expect("an answer in time");
        if let Some(t) = want(received) {
            return t;
        }
    }
}

fn registered(port: u16) -> (Connection, mpsc::Receiver<Received>) {
    let (connection, rx) = connect(port, TOKEN);
    next(&rx, |r| {
        matches!(r, Received::Registered { .. }).then_some(())
    });
    (connection, rx)
}

fn open(connection: &Connection, rx: &mpsc::Receiver<Received>, path: &str) -> Box<Opened> {
    connection.send(Command::Open {
        path: path.into(),
        hdu: None,
        budget: BUDGET,
        sigmas: 1.0,
    });
    next(rx, |r| match r {
        Received::Opened(opened) => Some(opened),
        r => panic!("{r:?}"),
    })
}

fn read(path: &Path) -> Cube {
    fits::read_fits(&std::fs::File::open(path).unwrap(), None).unwrap()
}

/// The value of a voxel of the file opened as `file_id`, by way of the server.
fn pixel(
    connection: &Connection,
    rx: &mpsc::Receiver<Received>,
    file_id: i32,
    voxel: [usize; 3],
) -> f32 {
    let request = Request::PixelValue { id: 7, voxel };
    connection.send(Command::Request { file_id, request });
    next(rx, |r| match r {
        Received::Event {
            file_id: f,
            event: Event::PixelValue { id: 7, value },
        } if f == file_id => Some(value),
        Received::Event { .. } => None,
        r => panic!("{r:?}"),
    })
}

#[test]
fn a_cube_opens_and_its_cubelets_histogram_and_values_come_through_the_adapter() {
    let root = root("open");
    let port = start(&root);

    // A wrong token never gets a session.
    let (_wrong, rx) = connect(port, "0123456789abcdee");
    let closed = next(&rx, |r| match r {
        Received::Closed(reason) => Some(reason),
        Received::Registered { .. } => panic!("registered with a wrong token"),
        _ => None,
    });
    assert!(closed.contains("401"), "{closed}");

    let (connection, rx) = connect(port, TOKEN);
    let server = next(&rx, |r| match r {
        Received::Registered { server } => Some(server),
        r => panic!("{r:?}"),
    });
    assert!(server.starts_with("fits-server"));

    // The root's cube and folder, without the hidden folder or the link out of the root.
    connection.send(Command::List(String::new()));
    let listing = next(&rx, |r| match r {
        Received::Listing(listing) => Some(listing),
        r => panic!("{r:?}"),
    });
    assert_eq!(listing.directories, ["folder"]);
    let files: Vec<_> = listing
        .files
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert_eq!(files, ["cube.fits"]);

    // Nothing outside the root, by .., an absolute path or a link.
    let outside = root.parent().unwrap().join("outside.fits");
    let budget = BUDGET;
    for path in [
        "../outside.fits",
        "folder/../../outside.fits",
        outside.to_str().unwrap(),
        "link.fits",
    ] {
        connection.send(Command::Open {
            path: path.into(),
            hdu: None,
            budget,
            sigmas: 1.0,
        });
        let failed = next(&rx, |r| match r {
            Received::OpenFailed(message) => Some(message),
            r => panic!("{path}: {r:?}"),
        });
        assert!(
            failed.contains("root") || failed.contains("no such file"),
            "{path}: {failed}"
        );
    }
    connection.send(Command::List("..".into()));
    next(&rx, |r| matches!(r, Received::Failed(_)).then_some(()));

    connection.send(Command::Info("cube.fits".into()));
    let dims = next(&rx, |r| match r {
        Received::Info { dims, .. } => Some(dims),
        r => panic!("{r:?}"),
    });
    assert_eq!(dims, [150, 70, 80]);

    connection.send(Command::Open {
        path: "cube.fits".into(),
        hdu: None,
        budget,
        sigmas: 1.0,
    });
    let opened = next(&rx, |r| match r {
        Received::Opened(opened) => Some(opened),
        r => panic!("{r:?}"),
    });
    let cube =
        fits::read_fits(&std::fs::File::open(root.join("cube.fits")).unwrap(), None).unwrap();
    let (info, file_id) = (&opened.info, opened.file_id);
    assert_eq!(
        (info.dims, info.unit().as_deref()),
        (cube.info.dims, Some("K"))
    );
    assert_eq!(
        format!("{:?}", info.stats),
        format!("{:?}", cube.info.stats)
    );
    assert_eq!(info.sigma, cube.info.sigma);
    for percent in fits::PERCENTILES {
        assert_eq!(
            info.histogram.clip(percent),
            cube.info.histogram.clip(percent)
        );
    }
    let levels = fits::levels(cube.info.dims, 384, 2048);
    let chain = levels.chain.len();
    assert_eq!(opened.levels, levels.list().0);
    let tolerance = opened.tolerance.unwrap();
    assert_eq!(tolerance, cube.info.sigma.unwrap());
    let tolerance = |id: CubeletId| {
        let factors = opened.levels[usize::from(id.level)];
        (tolerance * id.mode.noise_ratio(factors)) as f32
    };

    let request = |request| Command::Request { file_id, request };
    connection.send(request(Request::Levels {
        factors: opened.levels.clone(),
        chain,
    }));
    // The coarsest of the max pyramid, then a constant one of level 0 and a level between of the mean one.
    let coarsest = chain - 1;
    let constant = fits::cubelet(&cube, [1; 3], [2, 1, 0], true)
        .unwrap()
        .voxels;
    assert_eq!(constant, Voxels::Constant(0.5));
    let ids = vec![
        CubeletId::new(Mode::Max, Default::default(), coarsest, [0; 3]),
        CubeletId::new(Mode::Max, Default::default(), 0, [2, 1, 0]),
        CubeletId::new(Mode::Mean, Default::default(), 1, [0, 0, 0]),
    ];
    connection.send(request(Request::AddCubelets { ids: ids.clone() }));
    connection.send(request(Request::PixelValue {
        id: 41,
        voxel: [123, 45, 67],
    }));
    let mut arrived = Vec::new();
    while arrived.len() < 3 {
        let (id, cubelet) = next(&rx, |r| match r {
            Received::Event {
                file_id: f,
                event: Event::Cubelet { id, cubelet },
            } if f == file_id => Some((id, cubelet)),
            Received::Event {
                event: Event::PixelValue { id: 41, value },
                ..
            } => {
                assert_eq!(value, voxel([123, 45, 67]));
                None
            }
            r => panic!("{r:?}"),
        });
        let factors = opened.levels[usize::from(id.level)];
        let local = fits::cubelet(&cube, factors, id.at(), id.mode == Mode::Max).unwrap();
        assert_eq!((cubelet.dims, cubelet.cell), (local.dims, local.cell));
        for (g, w) in values(&cubelet).iter().zip(&values(&local)) {
            assert!(if w.is_nan() {
                g.is_nan()
            } else {
                (g - w).abs() <= tolerance(id)
            });
        }
        arrived.push(id);
    }
    arrived.sort_by_key(|id| id.level);
    assert_eq!(arrived, [ids[1], ids[2], ids[0]]);

    // A message that isn't CARTA's ends that session only.
    let (mut raw, _) =
        tungstenite::connect(format!("ws://127.0.0.1:{port}/?token={TOKEN}")).unwrap();
    raw.send(tungstenite::Message::binary(vec![1, 0, 30, 0]))
        .unwrap();
    loop {
        match raw.read() {
            Ok(tungstenite::Message::Close(_)) | Err(_) => break,
            Ok(_) => {}
        }
    }
    connection.send(request(Request::PixelValue {
        id: 42,
        voxel: [0, 0, 0],
    }));
    let value = next(&rx, |r| match r {
        Received::Event {
            event: Event::PixelValue { id: 42, value },
            ..
        } => Some(value),
        _ => None,
    });
    assert!(value.is_nan());
    let _ = std::fs::remove_dir_all(root.parent().unwrap());
}

#[test]
fn a_failed_open_leaves_the_cube_before_open_unless_it_had_to_go() {
    let root = root("failed-open");
    // A header that reads, over data that doesn't.
    write_fits(
        &root.join("bad.fits"),
        [8, 8, 8],
        &["BSCALE  = 'x'"],
        |_| 1.0,
    );
    let port = start(&root);
    let (connection, rx) = registered(port);
    let file_id = open(&connection, &rx, "cube.fits").file_id;

    // A path that isn't there fails before the cube goes, and it still answers.
    connection.send(Command::Open {
        path: "missing.fits".into(),
        hdu: None,
        budget: BUDGET,
        sigmas: 1.0,
    });
    next(&rx, |r| match r {
        Received::OpenFailed(_) => Some(()),
        r => panic!("{r:?}"),
    });
    assert_eq!(
        pixel(&connection, &rx, file_id, [12, 34, 56]),
        voxel([12, 34, 56])
    );

    // Data that doesn't read fails once the cube has gone, and the client hears that it has.
    connection.send(Command::Open {
        path: "bad.fits".into(),
        hdu: None,
        budget: BUDGET,
        sigmas: 1.0,
    });
    let closed = next(&rx, |r| match r {
        Received::FileClosed { file_id, .. } => Some(file_id),
        r => panic!("{r:?}"),
    });
    assert_eq!(closed, file_id);
    let failed = next(&rx, |r| match r {
        Received::OpenFailed(message) => Some(message),
        r => panic!("{r:?}"),
    });
    assert!(failed.contains("BSCALE"), "{failed}");
    let _ = std::fs::remove_dir_all(root.parent().unwrap());
}

/// A client that reuses a file ID, as CARTA's frontend does, by way of CARTA's messages.
struct Raw(tungstenite::WebSocket<tungstenite::stream::MaybeTlsStream<std::net::TcpStream>>);

impl Raw {
    fn send(&mut self, event: EventType, message: &impl prost::Message) {
        let bytes = frame(event, 0, message);
        self.0.send(tungstenite::Message::binary(bytes)).unwrap();
    }

    /// The next message's event type and its bytes.
    fn next(&mut self) -> (EventType, Vec<u8>) {
        match self.0.read().unwrap() {
            tungstenite::Message::Binary(bytes) => {
                let event = carta::unframe(&bytes).unwrap().event.unwrap();
                (event, bytes.to_vec())
            }
            message => panic!("{message:?}"),
        }
    }
}

#[test]
fn a_reused_file_id_gets_only_the_new_cubes_cubelets() {
    let root = root("reuse");
    write_fits(&root.join("other.fits"), [150, 70, 80], &[], |at| {
        voxel(at) + 100.0
    });
    let other = read(&root.join("other.fits"));
    let port = start(&root);
    let (socket, _) =
        tungstenite::connect(format!("ws://127.0.0.1:{port}/?token={TOKEN}")).unwrap();
    let mut raw = Raw(socket);
    raw.send(EventType::RegisterViewer, &proto::RegisterViewer::default());
    assert_eq!(raw.next().0, EventType::RegisterViewerAck);

    let opening = |file: &str| proto::OpenFile {
        file: file.into(),
        file_id: 0,
        ..Default::default()
    };
    raw.send(EventType::OpenFile, &opening("cube.fits"));
    let (event, bytes) = raw.next();
    assert_eq!(event, EventType::OpenFileAck);
    let ack: proto::OpenFileAck = carta::unframe(&bytes).unwrap().decode().unwrap();
    let levels: Vec<[usize; 3]> = (ack.volume_info.unwrap().levels.iter())
        .map(|f| [f.x, f.y, f.z].map(|f| f as usize))
        .collect();
    // Level 0's, and level 1's of the mean pyramid, which waits on a level job.
    let ids: Vec<CubeletId> = (0..2)
        .flat_map(|level| {
            let [x, y, z] = [150, 70, 80].map(|d: usize| d.div_ceil(64));
            (0..x * y * z).map(move |i| {
                CubeletId::new(
                    Mode::Mean,
                    Default::default(),
                    level,
                    [i % x, i / x % y, i / x / y],
                )
            })
        })
        .filter(|id| {
            fits::cubelet_region([150, 70, 80], levels[usize::from(id.level)], id.at()).is_some()
        })
        .collect();
    let add = proto::AddRequiredCubelets {
        file_id: 0,
        mode: proto::SmoothingMode::BlockAverage.into(),
        cubelets: ids
            .iter()
            .map(|id| cubelet_key(id.level.into(), [id.x, id.y, id.z]))
            .collect(),
        ..Default::default()
    };
    // The second open comes while the first cube's jobs are out.
    raw.send(EventType::AddRequiredCubelets, &add);
    raw.send(EventType::OpenFile, &opening("other.fits"));
    raw.send(EventType::AddRequiredCubelets, &add);
    while raw.next().0 != EventType::OpenFileAck {}
    let mut arrived = Vec::new();
    while arrived.len() < ids.len() {
        let (event, bytes) = raw.next();
        assert_eq!(event, EventType::CubeletData);
        let data: proto::CubeletData = carta::unframe(&bytes).unwrap().decode().unwrap();
        let level = data.level as usize;
        let id = CubeletId::new(
            carta::mode(data.mode).unwrap(),
            Default::default(),
            level,
            [data.x, data.y, data.z].map(|v| v as usize),
        );
        let local = fits::cubelet(&other, levels[level], id.at(), id.mode == Mode::Max).unwrap();
        let got = codec::decode(&data).unwrap();
        assert_eq!(
            format!("{:?}", values(&got)),
            format!("{:?}", values(&local)),
            "{id:?}"
        );
        arrived.push(id);
    }
    arrived.sort_by_key(|id| (id.level, id.z, id.y, id.x));
    let mut wanted = ids.clone();
    wanted.sort_by_key(|id| (id.level, id.z, id.y, id.x));
    assert_eq!(arrived, wanted);
    let _ = std::fs::remove_dir_all(root.parent().unwrap());
}

#[test]
fn a_level_list_unlike_the_servers_closes_the_file() {
    let root = root("levels");
    let port = start(&root);
    let (connection, rx) = registered(port);
    let opened = open(&connection, &rx, "cube.fits");
    let file_id = opened.file_id;
    let mut factors = opened.levels.clone();
    factors.push([64, 64, 64]);
    let request = |request| Command::Request { file_id, request };
    connection.send(request(Request::Levels { factors, chain: 1 }));
    let reason = next(&rx, |r| match r {
        Received::FileClosed { file_id: f, reason } if f == file_id => Some(reason),
        r => panic!("{r:?}"),
    });
    assert!(reason.contains("levels"), "{reason}");

    // Nothing more goes for it, and the next open is answered as usual.
    let id = CubeletId::new(Mode::Max, Default::default(), 0, [0; 3]);
    connection.send(request(Request::AddCubelets { ids: vec![id] }));
    connection.send(request(Request::PixelValue {
        id: 7,
        voxel: [12, 34, 56],
    }));
    let opened = open(&connection, &rx, "cube.fits");
    assert_eq!(
        pixel(&connection, &rx, opened.file_id, [12, 34, 56]),
        voxel([12, 34, 56])
    );
    let _ = std::fs::remove_dir_all(root.parent().unwrap());
}

#[test]
fn removed_cubelets_never_arrive_and_the_rest_do() {
    let root = root("remove");
    write_fits(&root.join("wide.fits"), [512, 512, 8], &[], |[x, y, z]| {
        ((x * 3 + y * 5 + z) % 89) as f32
    });
    let port = start(&root);
    let (connection, rx) = registered(port);
    let opened = open(&connection, &rx, "wide.fits");
    let file_id = opened.file_id;
    let request = |request| Command::Request { file_id, request };
    let chain = fits::levels([512, 512, 8], BUDGET.gpu_memory_limit_mb, BUDGET.max_depth)
        .chain
        .len();
    connection.send(request(Request::Levels {
        factors: opened.levels.clone(),
        chain,
    }));
    let ids: Vec<_> = (0..64)
        .map(|i| CubeletId::new(Mode::Max, Default::default(), 0, [i % 8, i / 8, 0]))
        .collect();
    // The first are being cut by the time the removal comes, and the last still queued.
    let removed = [&ids[..2], &ids[40..]].concat();
    connection.send(request(Request::AddCubelets { ids: ids.clone() }));
    connection.send(request(Request::RemoveCubelets {
        ids: removed.clone(),
    }));
    let wanted: Vec<_> = (ids.iter())
        .filter(|id| !removed.contains(id))
        .map(|id| id.at())
        .collect();
    let mut arrived = Vec::new();
    // Until the rest are in, then a while longer for any that shouldn't come.
    for wait in [Duration::from_secs(20), Duration::from_millis(500)] {
        while let Ok(r) = rx.recv_timeout(wait) {
            match r {
                Received::Event {
                    event: Event::Cubelet { id, .. },
                    ..
                } => arrived.push(id.at()),
                r => panic!("{r:?}"),
            }
            if arrived.len() == wanted.len() && wait > Duration::from_secs(1) {
                break;
            }
        }
    }
    arrived.sort_by_key(|&[x, y, _]| (y, x));
    assert_eq!(arrived, wanted);
    let _ = std::fs::remove_dir_all(root.parent().unwrap());
}

#[test]
fn a_browser_gives_the_token_when_it_registers_rather_than_in_the_url() {
    let root = root("api-key");
    let port = start(&root);
    let url = format!("ws://127.0.0.1:{port}/");
    let (_wrong, rx) = connect_with(&url, "0123456789abcdee");
    let closed = next(&rx, |r| match r {
        Received::Closed(reason) => Some(reason),
        r => panic!("{r:?}"),
    });
    assert!(closed.contains("Invalid token"), "{closed}");
    let (_none, rx) = connect_with(&url, "");
    next(&rx, |r| matches!(r, Received::Closed(_)).then_some(()));

    let (connection, rx) = connect_with(&url, TOKEN);
    next(&rx, |r| {
        matches!(r, Received::Registered { .. }).then_some(())
    });
    let file_id = open(&connection, &rx, "cube.fits").file_id;
    assert_eq!(
        pixel(&connection, &rx, file_id, [12, 34, 56]),
        voxel([12, 34, 56])
    );
    let _ = std::fs::remove_dir_all(root.parent().unwrap());
}
