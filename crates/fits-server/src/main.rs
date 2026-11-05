use std::path::PathBuf;

const USAGE: &str = "usage: fits-server [--root DIR] [--port N] [--host ADDRESS]
  --root DIR       the folder whose FITS files are served (default: the current one)
  --port N         the port to listen on, 0 for any free one (default: 3002)
  --host ADDRESS   the address to listen on (default: 127.0.0.1)
Clients need the token in CARTA_AUTH_TOKEN, or else one made at start and printed with the URL.";

fn main() -> std::io::Result<()> {
    let (mut root, mut port, mut host) = (PathBuf::from("."), 3002, "127.0.0.1".to_string());
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
    tokio::runtime::Runtime::new()?.block_on(async {
        let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
        let address = listener.local_addr()?;
        // The token only when it was made here, since there's no other way to learn it.
        match made {
            true => println!(
                "Serving {} at ws://{address}/?token={token}",
                root.display()
            ),
            false => println!(
                "Serving {} at ws://{address}/ with the token in CARTA_AUTH_TOKEN",
                root.display()
            ),
        }
        fits_server::serve(listener, &root, token).await
    })
}
