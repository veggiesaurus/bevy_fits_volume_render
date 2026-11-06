//! `wss://` alongside `ws://` on the same port, for a browser on an https page: Safari only lets one reach `wss://`. A
//! connection whose first byte starts a TLS handshake is taken as TLS, and any other as plain.

use std::{
    fs, io,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use rustls::{
    ServerConfig,
    pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject},
};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::mpsc,
};
use tokio_rustls::{TlsAcceptor, server::TlsStream};
use tokio_util::either::Either;

/// How long fits-server's own certificate is valid for, under the 825 days that Apple allows a TLS server certificate.
const VALID_FOR_DAYS: i64 = 800;
/// A certificate with less than this left is made again when the server starts.
const RENEW_WITHIN: Duration = Duration::from_secs(30 * 24 * 60 * 60);
/// How long a connection has to send its first byte and, for TLS, to finish the handshake.
const HANDSHAKE: Duration = Duration::from_secs(10);

/// The first byte of a TLS handshake record.
const HANDSHAKE_RECORD: u8 = 0x16;

pub type Io = Either<TcpStream, TlsStream<TcpStream>>;

/// TLS with fits-server's own certificate for localhost, kept in `dir`. It's made on first run and kept, so that
/// trusting it once lasts, and made again when the server starts with less than 30 days of it left, or when it won't
/// load. It names this machine only, and isn't a CA, so it can't vouch for other certificates.
pub fn own_config(dir: &Path) -> io::Result<(Arc<ServerConfig>, PathBuf)> {
    let (cert, key, expires) = (
        dir.join("localhost.pem"),
        dir.join("localhost-key.pem"),
        dir.join("localhost.expires"),
    );
    // When it expires, in seconds since 1970, kept in a file beside it so that the certificate needn't be parsed.
    let left = (fs::read_to_string(&expires).ok())
        .and_then(|seconds| seconds.trim().parse().ok())
        .map(|seconds| UNIX_EPOCH + Duration::from_secs(seconds))
        .and_then(|expiry: SystemTime| expiry.duration_since(SystemTime::now()).ok());
    if left.is_some_and(|left| left > RENEW_WITHIN)
        && let Ok(config) = server_config(&cert, &key)
    {
        return Ok((config, cert));
    }
    let (cert_pem, key_pem, expiry) = make_certificate().map_err(io::Error::other)?;
    fs::create_dir_all(dir)?;
    write(&key, key_pem.as_bytes(), true)?;
    write(&cert, cert_pem.as_bytes(), false)?;
    write(&expires, expiry.to_string().as_bytes(), false)?;
    Ok((server_config(&cert, &key)?, cert))
}

/// A certificate and key in PEM, and when it expires in seconds since 1970.
fn make_certificate() -> Result<(String, String, i64), rcgen::Error> {
    let names = ["localhost", "127.0.0.1", "::1"].map(String::from).to_vec();
    let mut params = rcgen::CertificateParams::new(names)?;
    (params.distinguished_name).push(rcgen::DnType::CommonName, "fits-server on localhost");
    let now = time::OffsetDateTime::now_utc();
    params.not_before = now - time::Duration::days(1);
    params.not_after = now + time::Duration::days(VALID_FOR_DAYS);
    params.is_ca = rcgen::IsCa::ExplicitNoCa;
    params.key_usages = vec![rcgen::KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![rcgen::ExtendedKeyUsagePurpose::ServerAuth];
    let key = rcgen::KeyPair::generate()?;
    let cert = params.self_signed(&key)?;
    let expiry = params.not_after.unix_timestamp();
    Ok((cert.pem(), key.serialize_pem(), expiry))
}

/// Through a file beside it and a rename, so that a write cut short never leaves half a file. `private` is readable by
/// its owner only.
fn write(path: &Path, bytes: &[u8], private: bool) -> io::Result<()> {
    use std::io::Write;
    let partial = path.with_extension(format!("partial.{}", std::process::id()));
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    if private {
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    }
    options.open(&partial)?.write_all(bytes)?;
    fs::rename(&partial, path)
}

/// TLS with the certificate chain and key in the PEM files `cert` and `key`.
pub fn server_config(cert: &Path, key: &Path) -> io::Result<Arc<ServerConfig>> {
    let error = |what: &Path, e: &dyn std::fmt::Display| {
        io::Error::other(format!("{}: {e}", what.display()))
    };
    let chain = CertificateDer::pem_file_iter(cert)
        .and_then(|certs| certs.collect::<Result<Vec<_>, _>>())
        .map_err(|e| error(cert, &e))?;
    let private = PrivateKeyDer::from_pem_file(key).map_err(|e| error(key, &e))?;
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let config = ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .and_then(|builder| {
            builder
                .with_no_client_auth()
                .with_single_cert(chain, private)
        })
        .map_err(|e| error(cert, &e))?;
    Ok(Arc::new(config))
}

/// Plain and TLS connections on one port, each handed over once it has shown which it is, so that a slow one holds up
/// no other. Accepting stops once the server is dropped.
pub struct Listener {
    ready: mpsc::Receiver<(Io, SocketAddr)>,
    local: SocketAddr,
}

impl Listener {
    pub fn new(tcp: TcpListener, tls: Arc<ServerConfig>) -> io::Result<Self> {
        let local = tcp.local_addr()?;
        let (handed, ready) = mpsc::channel(64);
        let acceptor = TlsAcceptor::from(tls);
        tokio::spawn(async move {
            loop {
                let accepted = tokio::select! {
                    accepted = tcp.accept() => accepted,
                    () = handed.closed() => return,
                };
                let (stream, address) = match accepted {
                    Ok(accepted) => accepted,
                    Err(e) => {
                        accept_failed(e).await;
                        continue;
                    }
                };
                let (handed, acceptor) = (handed.clone(), acceptor.clone());
                tokio::spawn(async move {
                    let io = tokio::time::timeout(HANDSHAKE, sort(stream, &acceptor)).await;
                    match io {
                        Ok(Ok(Some(io))) => {
                            let _ = handed.send((io, address)).await;
                        }
                        // Most often a browser that doesn't trust the certificate yet.
                        Ok(Err(e)) => println!("TLS from {address} failed: {e}"),
                        Ok(Ok(None)) | Err(_) => {}
                    }
                });
            }
        });
        Ok(Self { ready, local })
    }
}

/// As axum's own listener does: a connection that failed as it was accepted is no matter, but something else, such as
/// running out of file descriptors, is reported and waited out.
async fn accept_failed(e: io::Error) {
    use io::ErrorKind::*;
    if !matches!(
        e.kind(),
        ConnectionAborted | ConnectionRefused | ConnectionReset
    ) {
        println!("Accepting a connection: {e}");
        tokio::time::sleep(Duration::from_secs(1)).await;
    }
}

/// None for a connection that closed before sending anything.
async fn sort(stream: TcpStream, acceptor: &TlsAcceptor) -> io::Result<Option<Io>> {
    let mut first = [0];
    if stream.peek(&mut first).await? == 0 {
        return Ok(None);
    }
    Ok(Some(match first[0] {
        HANDSHAKE_RECORD => Either::Right(acceptor.accept(stream).await?),
        _ => Either::Left(stream),
    }))
}

impl axum::serve::Listener for Listener {
    type Io = Io;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Io, SocketAddr) {
        match self.ready.recv().await {
            Some(connection) => connection,
            // The accepting task only ends with the runtime.
            None => std::future::pending().await,
        }
    }

    fn local_addr(&self) -> io::Result<SocketAddr> {
        Ok(self.local)
    }
}
