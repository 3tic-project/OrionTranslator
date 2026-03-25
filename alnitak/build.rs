fn main() {
    // ── Windows: embed logo.ico as PE resource (exe icon + taskbar icon) ──────
    // GPUI on Windows loads the icon via LoadImageW(module, PCWSTR(1 as _), …),
    // i.e. resource ID 1.  winresource writes the icon as ID 1 by default.
    #[cfg(target_os = "windows")]
    {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let ico = std::path::Path::new(&manifest_dir).join("../assets/logo.ico");

        if ico.exists() {
            let mut res = winresource::WindowsResource::new();
            res.set_icon(ico.to_str().unwrap());
            res.set("FileDescription", "Orion 翻译器");
            res.set("ProductName", "OrionTranslator");
            if let Err(e) = res.compile() {
                eprintln!("winresource compile error: {e}");
                std::process::exit(1);
            }
            println!("cargo:rerun-if-changed={}", ico.display());
        } else {
            println!(
                "cargo:warning=assets/logo.ico not found — \
                 Windows exe will use the default icon. \
                 Run `magick assets/logo.png -define icon:auto-resize=256,128,64,48,32,16 assets/logo.ico` to generate it."
            );
        }
    }

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../assets/logo.ico");
}
