use std::env;
use std::path::PathBuf;

fn main() {
    // When the `embed-rules` feature is enabled, locate and verify the rules file exists.
    // The actual embedding is done via include_str! in the source code — we just need to
    // tell Cargo to re-run when the file changes and verify it exists at build time.
    if env::var("CARGO_FEATURE_EMBED_RULES").is_ok() {
        let candidates = [PathBuf::from("rules/ja2zh_context_rules.json")];

        let found = candidates.iter().find(|p| p.exists());
        match found {
            Some(path) => {
                let abs = std::fs::canonicalize(path).expect("Failed to canonicalize rules path");
                println!("cargo:rustc-env=ORION_RULES_PATH={}", abs.display());
                println!("cargo:rerun-if-changed={}", abs.display());
            }
            None => {
                panic!("embed-rules feature enabled but rules file not found");
            }
        }
    }

    // Always rebuild if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}
