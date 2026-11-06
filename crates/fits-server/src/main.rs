use std::path::PathBuf;

const USAGE: &str = "usage: fits-server [--root DIR] [--port N] [--host ADDRESS] [--memory-limit-gb GB]
                   [--tls-cert PEM --tls-key PEM]
  --root DIR             the folder whose FITS files are served (default: the current one)
  --port N               the port to listen on, 0 for any free one (default: 3002)
  --host ADDRESS         the address to listen on (default: 127.0.0.1)
  --memory-limit-gb GB   how much memory the cubes open may take, shared by every client; an open that needs
                         more fails (default: half the machine's memory)
  --tls-cert PEM         a certificate for wss://, such as one made with mkcert
  --tls-key PEM          its private key
Clients need the token in CARTA_AUTH_TOKEN, or else one made at start and printed with the URL. The port takes ws://
and wss:// both. Without --tls-cert, wss:// uses a certificate for localhost that fits-server makes on first run and
keeps in $XDG_CONFIG_HOME/fits-server (~/.config/fits-server when that's unset). Without a certificate, it serves
ws:// only.";

/// The machine's memory, from Linux's /proc/meminfo or macOS's sysctl.
fn physical_memory() -> Option<u64> {
    if let Ok(info) = std::fs::read_to_string("/proc/meminfo") {
        let total = info
            .lines()
            .find_map(|line| line.strip_prefix("MemTotal:"))?;
        return Some(
            total
                .trim()
                .strip_suffix("kB")?
                .trim()
                .parse::<u64>()
                .ok()?
                * 1024,
        );
    }
    let sysctl = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output();
    String::from_utf8(sysctl.ok()?.stdout)
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// Where fits-server keeps its own certificate, found as the viewer finds its config.
fn config_dir() -> Option<PathBuf> {
    let var = |name| {
        std::env::var_os(name)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    };
    let base = var("XDG_CONFIG_HOME")
        .or_else(|| var("HOME").map(|home| home.join(".config")))
        .or_else(|| var("APPDATA"))?;
    Some(base.join("fits-server"))
}

fn main() -> std::io::Result<()> {
    let (mut root, mut port, mut host) = (PathBuf::from("."), 3002, "127.0.0.1".to_string());
    let mut memory_limit_gb = None;
    let (mut cert, mut key) = (None::<PathBuf>, None::<PathBuf>);
    let mut args = std::env::args().skip(1);
    let usage = || -> ! {
        eprintln!("{USAGE}");
        std::process::exit(2)
    };
    while let Some(arg) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--root" => root = value().into(),
            "--port" => port = value().parse().unwrap_or_else(|_| usage()),
            "--host" => host = value(),
            "--memory-limit-gb" => {
                let gb = value().parse().ok().filter(|gb: &f64| *gb > 0.0);
                memory_limit_gb = Some(gb.unwrap_or_else(|| usage()))
            }
            "--tls-cert" => cert = Some(value().into()),
            "--tls-key" => key = Some(value().into()),
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => usage(),
        }
    }
    let (token, made) = match std::env::var("CARTA_AUTH_TOKEN") {
        Ok(token) if !token.is_empty() => (token, false),
        _ => {
            let mut bytes = [0u8; 16];
            getrandom::fill(&mut bytes).map_err(std::io::Error::other)?;
            (bytes.iter().map(|b| format!("{b:02x}")).collect(), true)
        }
    };
    // A certificate that can't be had leaves ws://, which is all the native viewer needs.
    let tls = match (&cert, &key) {
        (Some(cert), Some(key)) => {
            fits_server::tls::server_config(cert, key).map(|tls| (tls, None))
        }
        (None, None) => (config_dir()
            .ok_or(std::io::Error::other("no HOME or APPDATA to keep one in")))
        .and_then(|dir| fits_server::tls::own_config(&dir))
        .map(|(tls, cert)| (tls, Some(cert))),
        _ => usage(),
    };
    let (tls, own) = match tls {
        Ok((tls, own)) => (Some(tls), own),
        Err(e) => {
            println!("No wss://, for want of a certificate: {e}");
            (None, None)
        }
    };
    tokio::runtime::Runtime::new()?.block_on(async {
        let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
        let address = listener.local_addr()?;
        let also = if tls.is_some() {
            ", and wss:// on the same port"
        } else {
            ""
        };
        // The token only when it was made here, since there's no other way to learn it.
        match made {
            true => println!(
                "Serving {} at ws://{address}/?token={token}{also}",
                root.display()
            ),
            false => println!(
                "Serving {} at ws://{address}/{also}, with the token in CARTA_AUTH_TOKEN",
                root.display()
            ),
        }
        // Safari reaches a local server from an https page over wss:// only, with a certificate it trusts. The
        // server's own names this machine only.
        if let Some(cert) = own {
            let ip = address.ip();
            match ip.is_loopback() || ip.is_unspecified() {
                true => {
                    let at = if ip.is_loopback() {
                        address.to_string()
                    } else {
                        format!("127.0.0.1:{}", address.port())
                    };
                    println!(
                        "For Safari, trust {} once (see the README), then use wss://{at}",
                        cert.display()
                    );
                }
                false => println!(
                    "Its own certificate is for localhost only; give --tls-cert for wss:// at {ip}"
                ),
            }
        }
        // ponytail: 8 GB where the machine's memory can't be learnt, as on Windows.
        let memory_limit = memory_limit_gb.map_or_else(
            || physical_memory().map_or(8_000_000_000, |bytes| bytes / 2),
            |gb| (gb * 1e9) as u64,
        );
        println!("Cubes open may take {:.1} GB", memory_limit as f64 / 1e9);
        fits_server::serve(listener, &root, token, tls, memory_limit).await
    })
}
