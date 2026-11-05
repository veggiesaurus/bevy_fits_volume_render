//! carta-protobuf/ is carta-protobuf (github.com/CARTAvis/carta-protobuf), a submodule on its angus/volumes branch,
//! which adds the messages for volumes to CARTA's.

fn main() {
    let dirs = ["shared", "control", "request", "stream"].map(|d| format!("carta-protobuf/{d}"));
    let mut files = Vec::new();
    for dir in &dirs {
        for entry in std::fs::read_dir(dir).into_iter().flatten() {
            let path = entry.expect("reading carta-protobuf").path();
            if path.extension().is_some_and(|e| e == "proto") {
                files.push(path);
            }
        }
    }
    assert!(
        !files.is_empty(),
        "carta-protobuf is empty: git submodule update --init"
    );
    files.sort();
    let descriptors = protox::compile(&files, &dirs).unwrap_or_else(|e| panic!("{e:?}"));
    prost_build::Config::new()
        .compile_fds(descriptors)
        .expect("generating the protobuf code");
    println!("cargo:rerun-if-changed=carta-protobuf");
}
