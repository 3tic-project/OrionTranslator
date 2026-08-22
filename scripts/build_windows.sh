#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# OrionTranslator — Windows 构建脚本 (原生 Windows 环境)
# 构建 Release 二进制，打包为两个 zip：
#   *-Full.zip   : 主程序 + ner_model (首次安装)
#   *-Update.zip : 仅主程序 (覆盖更新)
# 用法: ./scripts/build_windows.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ── 常量 ──────────────────────────────────────────────────────
APP_NAME="OrionTranslator"
BINARY_NAME="alnitak"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERSION="$(awk '
    /^\[package\]$/ { in_package = 1; next }
    in_package && /^\[/ { exit }
    in_package && /^version[[:space:]]*=/ {
        value = $0
        sub(/^[^=]*=[[:space:]]*"/, "", value)
        sub(/"[[:space:]]*$/, "", value)
        print value
        exit
    }
' "${PROJECT_ROOT}/alnitak/Cargo.toml")"
if [ -z "${VERSION}" ]; then
    echo "❌ 无法从 alnitak/Cargo.toml 读取版本号"
    exit 1
fi
ZIP_NAME="${APP_NAME}-${VERSION}-Windows-x86_64"
DIST_DIR="${PROJECT_ROOT}/dist"
BUILD_DIR="${DIST_DIR}/.build_windows"

NER_MODEL_SRC="${PROJECT_ROOT}/alnilam/ner_model"

# A full release must contain the pinned default model. Reuse a verified local
# copy when available; otherwise download it from Hugging Face.
"${SCRIPT_DIR}/fetch_ner_model.sh" "${NER_MODEL_SRC}"

create_zip() {
    local source_dir="$1"
    local output_zip="$2"
    if command -v zip >/dev/null 2>&1; then
        (cd "${source_dir}" && zip -r -9 "${output_zip}" .)
    elif command -v 7z >/dev/null 2>&1; then
        (cd "${source_dir}" && 7z a -tzip -mx=9 "${output_zip}" .)
    else
        echo "❌ 需要 zip 或 7z 创建 Windows 发布包" >&2
        return 1
    fi
}

# ── 检测目标平台 ──────────────────────────────────────────────
OS="$(uname -s)"
case "${OS}" in
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        TARGET=""          # 原生 Windows，不需要交叉编译目标
        EXE_SUFFIX=".exe"
        ;;
    *)
        TARGET="x86_64-pc-windows-msvc"
        EXE_SUFFIX=".exe"
        echo "⚠️  非 Windows 环境，将尝试交叉编译到 ${TARGET}"
        echo "   请确保已安装交叉编译工具链: rustup target add ${TARGET}"
        ;;
esac

# ── 清理 ──────────────────────────────────────────────────────
echo "🧹 清理旧构建产物…"
rm -rf "${BUILD_DIR}"
mkdir -p "${DIST_DIR}" "${BUILD_DIR}"

# ── 生成 logo.ico (Windows 图标) ─────────────────────────────
echo "🎨 生成 logo.ico…"
ICO_PATH="${PROJECT_ROOT}/assets/logo.ico"
if command -v magick &>/dev/null; then
    magick "${PROJECT_ROOT}/assets/logo.png" \
        -define "icon:auto-resize=256,128,64,48,32,16" \
        "${ICO_PATH}"
    echo "   生成: ${ICO_PATH}"
elif command -v convert &>/dev/null; then
    convert "${PROJECT_ROOT}/assets/logo.png" \
        -define "icon:auto-resize=256,128,64,48,32,16" \
        "${ICO_PATH}"
    echo "   生成: ${ICO_PATH}"
else
    echo "⚠️  未找到 ImageMagick (magick/convert)，跳过 .ico 生成"
    echo "   exe 将使用默认图标。安装 ImageMagick 后重试，或手动放置 assets/logo.ico"
fi

# ── 编译 Release ──────────────────────────────────────────────
echo "🔨 编译 Release 二进制 (${BINARY_NAME})…"
cd "${PROJECT_ROOT}"

if [ -n "${TARGET}" ]; then
    cargo build --release -p alnitak --target "${TARGET}"
    BINARY_SRC="${PROJECT_ROOT}/target/${TARGET}/release/${BINARY_NAME}${EXE_SUFFIX}"
else
    cargo build --release -p alnitak
    BINARY_SRC="${PROJECT_ROOT}/target/release/${BINARY_NAME}${EXE_SUFFIX}"
fi

if [ ! -f "${BINARY_SRC}" ]; then
    echo "❌ 找不到编译产物: ${BINARY_SRC}"
    exit 1
fi

# ── 组装并打包 ────────────────────────────────────────────────
echo "📦 打包完整版 (含 NER 模型)…"
BASE_NAME="${ZIP_NAME}"

# ── 完整版 (含 NER 模型) ──────────────────────────────────────
FULL_STAGE="${BUILD_DIR}/full"
mkdir -p "${FULL_STAGE}"
cp "${BINARY_SRC}" "${FULL_STAGE}/${BINARY_NAME}${EXE_SUFFIX}"
echo "📂 拷贝 Orion-NER-30M-v1…"
cp -R "${NER_MODEL_SRC}" "${FULL_STAGE}/ner_model"
"${SCRIPT_DIR}/fetch_ner_model.sh" --verify-only "${FULL_STAGE}/ner_model"
FULL_ZIP="${DIST_DIR}/${BASE_NAME}-Full.zip"
create_zip "${FULL_STAGE}" "${FULL_ZIP}"
echo "✅ 完整版: ${FULL_ZIP} ($(du -h "${FULL_ZIP}" | cut -f1))"

# ── 更新版 (仅主程序) ─────────────────────────────────────────
echo "📦 打包更新版 (仅主程序)…"
LITE_STAGE="${BUILD_DIR}/lite"
mkdir -p "${LITE_STAGE}"
cp "${BINARY_SRC}" "${LITE_STAGE}/${BINARY_NAME}${EXE_SUFFIX}"
LITE_ZIP="${DIST_DIR}/${BASE_NAME}-Update.zip"
create_zip "${LITE_STAGE}" "${LITE_ZIP}"
echo "✅ 更新版: ${LITE_ZIP} ($(du -h "${LITE_ZIP}" | cut -f1))"

# ── 清理临时文件 ──────────────────────────────────────────────
rm -rf "${BUILD_DIR}"

# ── 完成 ──────────────────────────────────────────────────────
echo ""
echo "✅ 构建完成！"
echo "   完整版: ${FULL_ZIP}"
echo "   更新版: ${LITE_ZIP}"
