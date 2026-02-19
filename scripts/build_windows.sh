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
VERSION="0.1.0"
ZIP_NAME="${APP_NAME}-${VERSION}-Windows-x86_64"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${PROJECT_ROOT}/dist"
BUILD_DIR="${DIST_DIR}/.build_windows"

NER_MODEL_SRC="${PROJECT_ROOT}/alnilam/ner_model"

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
if [ -d "${NER_MODEL_SRC}" ]; then
    echo "📂 拷贝 NER 模型…"
    cp -R "${NER_MODEL_SRC}" "${FULL_STAGE}/ner_model"
else
    echo "⚠️  警告: 找不到 NER 模型目录 ${NER_MODEL_SRC}"
fi
FULL_ZIP="${DIST_DIR}/${BASE_NAME}-Full.zip"
(cd "${FULL_STAGE}" && zip -r -9 "${FULL_ZIP}" .)
echo "✅ 完整版: ${FULL_ZIP} ($(du -h "${FULL_ZIP}" | cut -f1))"

# ── 更新版 (仅主程序) ─────────────────────────────────────────
echo "📦 打包更新版 (仅主程序)…"
LITE_STAGE="${BUILD_DIR}/lite"
mkdir -p "${LITE_STAGE}"
cp "${BINARY_SRC}" "${LITE_STAGE}/${BINARY_NAME}${EXE_SUFFIX}"
LITE_ZIP="${DIST_DIR}/${BASE_NAME}-Update.zip"
(cd "${LITE_STAGE}" && zip -r -9 "${LITE_ZIP}" .)
echo "✅ 更新版: ${LITE_ZIP} ($(du -h "${LITE_ZIP}" | cut -f1))"

# ── 清理临时文件 ──────────────────────────────────────────────
rm -rf "${BUILD_DIR}"

# ── 完成 ──────────────────────────────────────────────────────
echo ""
echo "✅ 构建完成！"
echo "   完整版: ${FULL_ZIP}"
echo "   更新版: ${LITE_ZIP}"
