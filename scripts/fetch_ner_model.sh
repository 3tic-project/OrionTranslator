#!/usr/bin/env bash
# Fetch and verify the pinned default Orion NER model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=ner_model.conf
source "${SCRIPT_DIR}/ner_model.conf"

VERIFY_ONLY=false
if [ "${1:-}" = "--verify-only" ]; then
    VERIFY_ONLY=true
    shift
fi

MODEL_DIR="${1:-${ORION_NER_MODEL_DIR:-${PROJECT_ROOT}/alnilam/ner_model}}"

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "❌ 需要 sha256sum 或 shasum 校验 NER 模型" >&2
        return 1
    fi
}

verify_model() {
    local dir="$1"
    local index file expected actual
    for index in "${!NER_MODEL_FILES[@]}"; do
        file="${NER_MODEL_FILES[$index]}"
        expected="${NER_MODEL_SHA256[$index]}"
        if [ ! -f "${dir}/${file}" ]; then
            echo "❌ NER 模型缺少 ${dir}/${file}" >&2
            return 1
        fi
        actual="$(sha256_file "${dir}/${file}")"
        if [ "${actual}" != "${expected}" ]; then
            echo "❌ NER 模型校验失败: ${file}" >&2
            echo "   expected ${expected}" >&2
            echo "   actual   ${actual}" >&2
            return 1
        fi
    done
}

if verify_model "${MODEL_DIR}" 2>/dev/null; then
    echo "✅ Orion NER 模型已就绪: ${MODEL_DIR}"
    echo "   ${NER_MODEL_REPO}@${NER_MODEL_REVISION}"
    exit 0
fi

if [ "${VERIFY_ONLY}" = true ]; then
    verify_model "${MODEL_DIR}"
    exit 1
fi

DOWNLOAD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/orion-ner-download.XXXXXX")"
INSTALL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/orion-ner-install.XXXXXX")"
cleanup() {
    rm -rf "${DOWNLOAD_DIR}" "${INSTALL_DIR}"
}
trap cleanup EXIT

echo "📥 下载 ${NER_MODEL_REPO}@${NER_MODEL_REVISION}…"
if command -v hf >/dev/null 2>&1; then
    hf download "${NER_MODEL_REPO}" "${NER_MODEL_FILES[@]}" \
        --revision "${NER_MODEL_REVISION}" \
        --local-dir "${DOWNLOAD_DIR}" \
        --quiet
elif command -v curl >/dev/null 2>&1; then
    for file in "${NER_MODEL_FILES[@]}"; do
        curl --fail --location --retry 3 \
            --output "${DOWNLOAD_DIR}/${file}" \
            "https://huggingface.co/${NER_MODEL_REPO}/resolve/${NER_MODEL_REVISION}/${file}"
    done
else
    echo "❌ 需要 Hugging Face CLI（hf）或 curl 下载默认 NER 模型" >&2
    exit 1
fi

for file in "${NER_MODEL_FILES[@]}"; do
    cp "${DOWNLOAD_DIR}/${file}" "${INSTALL_DIR}/${file}"
done
verify_model "${INSTALL_DIR}"

cat >"${INSTALL_DIR}/MODEL_PROVENANCE.txt" <<EOF
repository=${NER_MODEL_REPO}
revision=${NER_MODEL_REVISION}
license=${NER_MODEL_LICENSE}
downloaded_by=OrionTranslator/scripts/fetch_ner_model.sh
EOF

mkdir -p "$(dirname "${MODEL_DIR}")"
BACKUP_DIR="${MODEL_DIR}.previous.$$"
if [ -e "${MODEL_DIR}" ]; then
    mv "${MODEL_DIR}" "${BACKUP_DIR}"
fi
if mv "${INSTALL_DIR}" "${MODEL_DIR}"; then
    if [ -e "${BACKUP_DIR}" ]; then
        rm -rf "${BACKUP_DIR}"
    fi
else
    if [ -e "${BACKUP_DIR}" ]; then
        mv "${BACKUP_DIR}" "${MODEL_DIR}"
    fi
    exit 1
fi

verify_model "${MODEL_DIR}"
echo "✅ Orion NER 模型安装完成: ${MODEL_DIR}"
