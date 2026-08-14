use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn resolved_path(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return path
            .canonicalize()
            .with_context(|| format!("无法解析路径: {}", path.display()));
    }

    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let resolved_parent = parent
        .canonicalize()
        .with_context(|| format!("无法解析父目录: {}", parent.display()))?;
    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("输出路径缺少文件名: {}", path.display()))?;
    Ok(resolved_parent.join(file_name))
}

pub(crate) fn ensure_distinct_paths(input: &Path, output: &Path) -> Result<()> {
    if resolved_path(input)? == resolved_path(output)? {
        bail!(
            "输入与输出不能是同一文件: {}",
            resolved_path(input)?.display()
        );
    }
    Ok(())
}

fn create_adjacent_temp(target: &Path) -> Result<(PathBuf, File)> {
    let parent = target.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("无法创建输出目录: {}", parent.display()))?;
    let name = target
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("output");

    for _ in 0..100 {
        let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temp_path = parent.join(format!(
            ".{}.orion-tmp-{}-{}",
            name,
            std::process::id(),
            counter
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp_path)
        {
            Ok(file) => return Ok((temp_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("无法创建临时输出文件: {}", temp_path.display()));
            }
        }
    }

    bail!("无法为输出创建唯一临时文件: {}", target.display())
}

pub(crate) fn atomic_write_with<F>(target: &Path, write: F) -> Result<()>
where
    F: FnOnce(&mut File) -> Result<()>,
{
    let (temp_path, mut file) = create_adjacent_temp(target)?;
    let result = (|| -> Result<()> {
        write(&mut file)?;
        file.flush()
            .with_context(|| format!("刷新临时文件失败: {}", temp_path.display()))?;
        file.sync_all()
            .with_context(|| format!("同步临时文件失败: {}", temp_path.display()))?;
        drop(file);
        std::fs::rename(&temp_path, target).with_context(|| {
            format!(
                "原子替换输出失败: {} -> {}",
                temp_path.display(),
                target.display()
            )
        })?;
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
    result
}

pub(crate) fn atomic_write(target: &Path, bytes: &[u8]) -> Result<()> {
    atomic_write_with(target, |file| {
        file.write_all(bytes)
            .with_context(|| format!("写入临时文件失败: {}", target.display()))?;
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("orion_io_{name}_{}_{}", std::process::id(), nonce))
    }

    #[test]
    fn atomic_write_replaces_complete_file() {
        let path = unique_path("replace");
        std::fs::write(&path, b"old").unwrap();

        atomic_write(&path, b"new-complete").unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"new-complete");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn failed_atomic_write_preserves_previous_file() {
        let path = unique_path("preserve");
        std::fs::write(&path, b"old").unwrap();

        let result = atomic_write_with(&path, |file| {
            file.write_all(b"partial")?;
            bail!("injected failure")
        });

        assert!(result.is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"old");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn rejects_same_input_and_output() {
        let path = unique_path("same");
        std::fs::write(&path, b"data").unwrap();

        let error = ensure_distinct_paths(&path, &path).unwrap_err();

        assert!(error.to_string().contains("同一文件"));
        let _ = std::fs::remove_file(path);
    }
}
