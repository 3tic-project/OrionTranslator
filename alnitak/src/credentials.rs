//! LLM 连接凭证按预设持久化（URL / 模型名 / API Key）。
//!
//! 文件内容经 XOR 混淆 + 十六进制编码，避免明文落盘。
//! 密钥硬编码为 `114514`，仅作本地混淆，**不是**强加密。

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};

use crate::types::ModelPreset;

/// 混淆密钥（用户指定）。仅用于本地防明文窥视，非安全加密。
const OBFUSCATION_KEY: &[u8] = b"114514";
const STORE_VERSION: u32 = 1;
const FILE_NAME: &str = "llm_credentials.v1";
static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PresetCredentials {
    #[serde(default)]
    pub llm_url: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub api_key: String,
}

impl PresetCredentials {
    pub fn from_preset_defaults(preset: ModelPreset) -> Self {
        Self {
            llm_url: preset.llm_url().to_string(),
            model: preset.model_name().to_string(),
            api_key: String::new(),
        }
    }

    /// 用内置默认值补全空的 URL / 模型；API Key 保持原样。
    pub fn filled_with_defaults(self, preset: ModelPreset) -> Self {
        let defaults = Self::from_preset_defaults(preset);
        Self {
            llm_url: if self.llm_url.trim().is_empty() {
                defaults.llm_url
            } else {
                self.llm_url
            },
            model: if self.model.trim().is_empty() && !defaults.model.is_empty() {
                defaults.model
            } else {
                self.model
            },
            api_key: self.api_key,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CredentialStore {
    pub version: u32,
    #[serde(default = "default_active_preset")]
    pub active_preset: String,
    #[serde(default)]
    pub deepseek: PresetCredentials,
    #[serde(default)]
    pub volcengine: PresetCredentials,
    #[serde(default)]
    pub orion: PresetCredentials,
}

fn default_active_preset() -> String {
    ModelPreset::DeepSeek.storage_key().to_string()
}

impl Default for CredentialStore {
    fn default() -> Self {
        Self {
            version: STORE_VERSION,
            active_preset: default_active_preset(),
            deepseek: PresetCredentials::from_preset_defaults(ModelPreset::DeepSeek),
            volcengine: PresetCredentials::from_preset_defaults(ModelPreset::Volcengine),
            orion: PresetCredentials::from_preset_defaults(ModelPreset::Orion),
        }
    }
}

impl CredentialStore {
    pub fn get(&self, preset: ModelPreset) -> &PresetCredentials {
        match preset {
            ModelPreset::DeepSeek => &self.deepseek,
            ModelPreset::Volcengine => &self.volcengine,
            ModelPreset::Orion => &self.orion,
        }
    }

    pub fn get_mut(&mut self, preset: ModelPreset) -> &mut PresetCredentials {
        match preset {
            ModelPreset::DeepSeek => &mut self.deepseek,
            ModelPreset::Volcengine => &mut self.volcengine,
            ModelPreset::Orion => &mut self.orion,
        }
    }

    pub fn set(&mut self, preset: ModelPreset, creds: PresetCredentials) {
        *self.get_mut(preset) = creds;
    }

    pub fn active_preset(&self) -> ModelPreset {
        ModelPreset::from_storage_key(&self.active_preset)
    }

    pub fn set_active_preset(&mut self, preset: ModelPreset) {
        self.active_preset = preset.storage_key().to_string();
    }
}

/// 应用配置目录：`…/OrionTranslator/llm_credentials.v1`
pub fn credentials_file_path() -> Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .ok_or_else(|| anyhow!("无法定位用户主目录 (HOME/USERPROFILE)"))?;

    let base = PathBuf::from(home);
    let dir = if cfg!(target_os = "macos") {
        base.join("Library/Application Support/OrionTranslator")
    } else if cfg!(target_os = "windows") {
        base.join("AppData").join("Roaming").join("OrionTranslator")
    } else {
        base.join(".config").join("orion-translator")
    };

    Ok(dir.join(FILE_NAME))
}

pub fn load_store() -> CredentialStore {
    load_store_from_path(&match credentials_file_path() {
        Ok(p) => p,
        Err(_) => return CredentialStore::default(),
    })
    .unwrap_or_default()
}

pub fn load_store_from_path(path: &Path) -> Result<CredentialStore> {
    if !path.exists() {
        return Ok(CredentialStore::default());
    }
    reject_symlink(path)?;
    set_private_permissions(path)?;
    let encoded = fs::read_to_string(path)
        .with_context(|| format!("读取凭证文件失败: {}", path.display()))?;
    let json = deobfuscate(encoded.trim())?;
    let mut store: CredentialStore = serde_json::from_str(&json).context("解析凭证 JSON 失败")?;
    store.version = STORE_VERSION;
    // 确保三个预设至少有默认 URL
    store.deepseek = store
        .deepseek
        .clone()
        .filled_with_defaults(ModelPreset::DeepSeek);
    store.volcengine = store
        .volcengine
        .clone()
        .filled_with_defaults(ModelPreset::Volcengine);
    store.orion = store.orion.clone().filled_with_defaults(ModelPreset::Orion);
    Ok(store)
}

pub fn save_store(store: &CredentialStore) -> Result<()> {
    let path = credentials_file_path()?;
    save_store_to_path(store, &path)
}

pub fn save_store_to_path(store: &CredentialStore, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("创建配置目录失败: {}", parent.display()))?;
    }
    let json = serde_json::to_string(store).context("序列化凭证失败")?;
    let encoded = obfuscate(&json);
    atomic_private_write(path, encoded.as_bytes())?;
    Ok(())
}

fn reject_symlink(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("读取凭证文件元数据失败: {}", path.display()))?;
    if metadata.file_type().is_symlink() {
        return Err(anyhow!("拒绝读取符号链接凭证文件: {}", path.display()));
    }
    Ok(())
}

#[cfg(unix)]
fn set_private_permissions(path: &Path) -> Result<()> {
    let metadata =
        fs::metadata(path).with_context(|| format!("读取凭证文件权限失败: {}", path.display()))?;
    if metadata.permissions().mode() & 0o777 != 0o600 {
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("收紧凭证文件权限失败: {}", path.display()))?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn set_private_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

fn create_private_temp(path: &Path) -> Result<(PathBuf, File)> {
    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("凭证路径缺少父目录: {}", path.display()))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("凭证文件名不是合法 UTF-8: {}", path.display()))?;

    for _ in 0..100 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let temp_path = parent.join(format!(
            ".{file_name}.tmp.{}.{}",
            std::process::id(),
            sequence
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&temp_path) {
            Ok(file) => return Ok((temp_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("创建临时凭证文件失败: {}", temp_path.display()));
            }
        }
    }

    Err(anyhow!("无法创建唯一的临时凭证文件"))
}

fn atomic_private_write(path: &Path, contents: &[u8]) -> Result<()> {
    if path.exists() {
        reject_symlink(path)?;
    }
    let (temp_path, mut file) = create_private_temp(path)?;
    let result = (|| -> Result<()> {
        file.write_all(contents)
            .with_context(|| format!("写入临时凭证文件失败: {}", temp_path.display()))?;
        file.flush()
            .with_context(|| format!("刷新临时凭证文件失败: {}", temp_path.display()))?;
        file.sync_all()
            .with_context(|| format!("同步临时凭证文件失败: {}", temp_path.display()))?;
        drop(file);
        fs::rename(&temp_path, path).with_context(|| {
            format!(
                "原子替换凭证文件失败: {} -> {}",
                temp_path.display(),
                path.display()
            )
        })?;
        set_private_permissions(path)?;
        if let Some(parent) = path.parent() {
            if let Ok(directory) = File::open(parent) {
                let _ = directory.sync_all();
            }
        }
        Ok(())
    })();

    if result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    result
}

fn obfuscate(plaintext: &str) -> String {
    let xored: Vec<u8> = plaintext
        .as_bytes()
        .iter()
        .enumerate()
        .map(|(i, &b)| b ^ OBFUSCATION_KEY[i % OBFUSCATION_KEY.len()])
        .collect();
    bytes_to_hex(&xored)
}

fn deobfuscate(encoded: &str) -> Result<String> {
    let bytes = hex_to_bytes(encoded)?;
    let plain: Vec<u8> = bytes
        .iter()
        .enumerate()
        .map(|(i, &b)| b ^ OBFUSCATION_KEY[i % OBFUSCATION_KEY.len()])
        .collect();
    String::from_utf8(plain).context("凭证解码后不是合法 UTF-8")
}

fn bytes_to_hex(data: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(data.len() * 2);
    for &b in data {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>> {
    let hex = hex.trim();
    if hex.is_empty() {
        return Ok(Vec::new());
    }
    if !hex.len().is_multiple_of(2) {
        return Err(anyhow!("十六进制长度必须为偶数"));
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    let bytes = hex.as_bytes();
    for i in (0..bytes.len()).step_by(2) {
        let hi = hex_nibble(bytes[i])?;
        let lo = hex_nibble(bytes[i + 1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(c: u8) -> Result<u8> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(anyhow!("非法十六进制字符: {}", c as char)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn roundtrip_obfuscation() {
        let plain = r#"{"api_key":"sk-test-114514","model":"deepseek-v4-flash"}"#;
        let encoded = obfuscate(plain);
        assert!(!encoded.contains("sk-test"));
        assert!(!encoded.contains("deepseek"));
        assert_eq!(deobfuscate(&encoded).unwrap(), plain);
    }

    #[test]
    fn roundtrip_store_file() {
        let mut store = CredentialStore::default();
        store.deepseek.api_key = "sk-secret-key".into();
        store.deepseek.model = "deepseek-v4-flash".into();
        store.volcengine.llm_url = "https://ark.cn-beijing.volces.com/api/v3".into();
        store.volcengine.api_key = "volc-key".into();
        store.set_active_preset(ModelPreset::Volcengine);

        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("orion_cred_test_{stamp}.v1"));
        save_store_to_path(&store, &path).unwrap();

        let disk = fs::read_to_string(&path).unwrap();
        assert!(!disk.contains("sk-secret-key"));
        assert!(!disk.contains("volc-key"));

        let loaded = load_store_from_path(&path).unwrap();
        assert_eq!(loaded.deepseek.api_key, "sk-secret-key");
        assert_eq!(loaded.volcengine.api_key, "volc-key");
        assert_eq!(loaded.active_preset(), ModelPreset::Volcengine);
        let _ = fs::remove_file(path);
    }

    #[cfg(unix)]
    #[test]
    fn credential_file_permissions_are_private() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("orion_cred_mode_test_{stamp}.v1"));
        save_store_to_path(&CredentialStore::default(), &path).unwrap();

        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let _ = fs::remove_file(path);
    }

    #[cfg(unix)]
    #[test]
    fn loading_existing_store_repairs_permissions() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("orion_cred_repair_test_{stamp}.v1"));
        save_store_to_path(&CredentialStore::default(), &path).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();

        load_store_from_path(&path).unwrap();
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn empty_fields_filled_with_defaults() {
        let filled = PresetCredentials::default().filled_with_defaults(ModelPreset::DeepSeek);
        assert_eq!(filled.llm_url, ModelPreset::DeepSeek.llm_url());
        assert_eq!(filled.model, ModelPreset::DeepSeek.model_name());
        assert!(filled.api_key.is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    #[ignore = "requires the user's saved provider credentials and network access"]
    async fn live_saved_credentials_can_translate_minimal_probe() {
        let store = load_store_from_path(&credentials_file_path().unwrap()).unwrap();
        let active = store.active_preset();
        let preset = std::iter::once(active)
            .chain(ModelPreset::ALL)
            .find(|preset| {
                let credentials = store.get(*preset);
                !credentials.api_key.trim().is_empty()
                    && !credentials.llm_url.trim().is_empty()
                    && !credentials.model.trim().is_empty()
            })
            .expect("没有可用于在线探针的已保存凭据");
        let credentials = store.get(preset);
        let client = alnilam::llm::LlmClient::with_params(
            &credentials.llm_url,
            &credentials.model,
            1,
            0.0,
            None,
            None,
            String::new(),
            None,
            Some(credentials.api_key.clone()),
        )
        .unwrap();

        let translated = tokio::time::timeout(
            std::time::Duration::from_secs(45),
            client.test_translation(),
        )
        .await
        .expect("在线探针超时")
        .expect("在线探针请求失败");
        assert!(!translated.trim().is_empty());
    }
}
