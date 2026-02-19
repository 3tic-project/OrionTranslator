#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# OrionTranslator — macOS 构建脚本
# 构建 Release 二进制，打包为 .app 并生成 .dmg
# 用法: ./scripts/build_macos.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ── 常量 ──────────────────────────────────────────────────────
APP_NAME="OrionTranslator"
APP_BUNDLE="${APP_NAME}.app"
BINARY_NAME="alnitak"
VERSION="0.1.0"
IDENTIFIER="com.orion.translator"
DMG_NAME="${APP_NAME}-${VERSION}-macOS"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DIST_DIR="${PROJECT_ROOT}/dist"
BUILD_DIR="${DIST_DIR}/.build_macos"
APP_DIR="${BUILD_DIR}/${APP_BUNDLE}"

NER_MODEL_SRC="${PROJECT_ROOT}/alnilam/ner_model"
LOGO_SRC="${PROJECT_ROOT}/assets/logo.png"

# ── 清理 ──────────────────────────────────────────────────────
echo "🧹 清理旧构建产物…"
rm -rf "${BUILD_DIR}"
mkdir -p "${DIST_DIR}" "${BUILD_DIR}"

# ── 编译 Release ──────────────────────────────────────────────
echo "🔨 编译 Release 二进制 (${BINARY_NAME})…"
cd "${PROJECT_ROOT}"
cargo build --release -p alnitak

# ── 生成 .icns 图标 ───────────────────────────────────────────
echo "🎨 生成应用图标…"
ICONSET_DIR="${BUILD_DIR}/AppIcon.iconset"
mkdir -p "${ICONSET_DIR}"

if [ ! -f "${LOGO_SRC}" ]; then
    echo "❌ 找不到 ${LOGO_SRC}"
    exit 1
fi

# 生成各种尺寸的图标
sips -z   16   16 "${LOGO_SRC}" --out "${ICONSET_DIR}/icon_16x16.png"      > /dev/null 2>&1
sips -z   32   32 "${LOGO_SRC}" --out "${ICONSET_DIR}/icon_16x16@2x.png"   > /dev/null 2>&1
sips -z   32   32 "${LOGO_SRC}" --out "${ICONSET_DIR}/icon_32x32.png"      > /dev/null 2>&1
sips -z   64   64 "${LOGO_SRC}" --out "${ICONSET_DIR}/icon_32x32@2x.png"   > /dev/null 2>&1
sips -z  128  128 "${LOGO_SRC}" --out "${ICONSET_DIR}/icon_128x128.png"    > /dev/null 2>&1
sips -z  256  256 "${LOGO_SRC}" --out "${ICONSET_DIR}/icon_128x128@2x.png" > /dev/null 2>&1
sips -z  256  256 "${LOGO_SRC}" --out "${ICONSET_DIR}/icon_256x256.png"    > /dev/null 2>&1
sips -z  512  512 "${LOGO_SRC}" --out "${ICONSET_DIR}/icon_256x256@2x.png" > /dev/null 2>&1
sips -z  512  512 "${LOGO_SRC}" --out "${ICONSET_DIR}/icon_512x512.png"    > /dev/null 2>&1
sips -z 1024 1024 "${LOGO_SRC}" --out "${ICONSET_DIR}/icon_512x512@2x.png" > /dev/null 2>&1

ICNS_PATH="${BUILD_DIR}/AppIcon.icns"
iconutil -c icns "${ICONSET_DIR}" -o "${ICNS_PATH}"

# ── 创建 .app 目录结构 ────────────────────────────────────────
echo "📦 创建 ${APP_BUNDLE}…"
mkdir -p "${APP_DIR}/Contents/MacOS"
mkdir -p "${APP_DIR}/Contents/Resources"

# 拷贝二进制
cp "${PROJECT_ROOT}/target/release/${BINARY_NAME}" "${APP_DIR}/Contents/MacOS/${BINARY_NAME}"

# 拷贝图标
cp "${ICNS_PATH}" "${APP_DIR}/Contents/Resources/AppIcon.icns"

# 拷贝 NER 模型
if [ -d "${NER_MODEL_SRC}" ]; then
    echo "📂 拷贝 NER 模型…"
    cp -R "${NER_MODEL_SRC}" "${APP_DIR}/Contents/Resources/ner_model"
else
    echo "⚠️  警告: 找不到 NER 模型目录 ${NER_MODEL_SRC}，跳过"
fi

# ── 写入 Info.plist ───────────────────────────────────────────
cat > "${APP_DIR}/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleDisplayName</key>
    <string>Orion 翻译器</string>
    <key>CFBundleIdentifier</key>
    <string>${IDENTIFIER}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleExecutable</key>
    <string>${BINARY_NAME}</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
</dict>
</plist>
PLIST

# ── ad-hoc 签名 ───────────────────────────────────────────────
echo "🔏 对 .app 进行 ad-hoc 签名…"
codesign --force --deep --sign - "${APP_DIR}" 2>/dev/null || true

# ── 创建 DMG ──────────────────────────────────────────────────
echo "💿 创建 DMG…"
DMG_PATH="${DIST_DIR}/${DMG_NAME}.dmg"
rm -f "${DMG_PATH}"

# 创建空白 DMG 目录
DMG_STAGE="${BUILD_DIR}/dmg_stage"
mkdir -p "${DMG_STAGE}"
cp -R "${APP_DIR}" "${DMG_STAGE}/"

# 创建 Applications 快捷方式
ln -s /Applications "${DMG_STAGE}/Applications"

# 使用 hdiutil 创建 DMG
hdiutil create \
    -volname "${APP_NAME}" \
    -srcfolder "${DMG_STAGE}" \
    -ov \
    -format UDZO \
    "${DMG_PATH}"

# ── 清理临时文件 ──────────────────────────────────────────────
rm -rf "${BUILD_DIR}"

# ── 完成 ──────────────────────────────────────────────────────
echo ""
echo "✅ 构建完成！"
echo "   DMG: ${DMG_PATH}"
echo "   大小: $(du -h "${DMG_PATH}" | cut -f1)"
