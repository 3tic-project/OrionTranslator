#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# OrionTranslator — Windows 构建脚本 (交叉编译 / 原生均可)
# 构建 Release 二进制，打包为 zip (主程序 + ner_model)
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

# ── 组装打包目录 ──────────────────────────────────────────────
echo "📦 组装打包目录…"
STAGE_DIR="${BUILD_DIR}/${APP_NAME}"
mkdir -p "${STAGE_DIR}"

# 拷贝二进制
cp "${BINARY_SRC}" "${STAGE_DIR}/${BINARY_NAME}${EXE_SUFFIX}"

# 拷贝 NER 模型
if [ -d "${NER_MODEL_SRC}" ]; then
    echo "📂 拷贝 NER 模型…"
    cp -R "${NER_MODEL_SRC}" "${STAGE_DIR}/ner_model"
else
    echo "⚠️  警告: 找不到 NER 模型目录 ${NER_MODEL_SRC}，跳过"
fi

# ── 创建 ZIP ──────────────────────────────────────────────────
echo "📦 创建 ZIP 压缩包…"
ZIP_PATH="${DIST_DIR}/${ZIP_NAME}.zip"
rm -f "${ZIP_PATH}"

cd "${BUILD_DIR}"
zip -r -9 "${ZIP_PATH}" "${APP_NAME}/"

# ── 清理临时文件 ──────────────────────────────────────────────
rm -rf "${BUILD_DIR}"

# ── 完成 ──────────────────────────────────────────────────────
echo ""
echo "✅ 构建完成！"
echo "   ZIP: ${ZIP_PATH}"
echo "   大小: $(du -h "${ZIP_PATH}" | cut -f1)"
