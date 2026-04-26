use anyhow::{Context, Result};
use regex::Regex;
use roxmltree::Document as XmlDocument;
use scraper::{Html, Selector};
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::Path;
use std::sync::OnceLock;

/// Block-level HTML tags that contain translatable text
pub const EPUB_BLOCK_TAGS: &[&str] = &[
    "p",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "blockquote",
    "div",
    "caption",
    "td",
    "th",
];

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EpubPackage {
    pub manifest: HashMap<String, String>,
    pub media_types: HashMap<String, String>,
    pub spine_ids: Vec<String>,
    pub toc_pages: HashSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpubLeafBlock {
    pub index: usize,
    pub html: String,
    pub text: String,
    pub raw_text: String,
}

/// Extract text lines from an EPUB file.
///
/// Each line corresponds to a block-level element (p, h1-h6, li, div, etc.)
/// from the EPUB content files, processed in spine order.
/// Ruby annotations (<rt> tags) are stripped.
pub fn extract_epub_lines(epub_path: &Path) -> Result<Vec<String>> {
    let file = std::fs::File::open(epub_path)
        .with_context(|| format!("EPUB文件不存在: {}", epub_path.display()))?;
    let mut archive = zip::ZipArchive::new(file).with_context(|| "Failed to read EPUB as ZIP")?;

    // Step 1: Read all files into memory
    let mut raw_items: HashMap<String, Vec<u8>> = HashMap::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).context("Failed to read ZIP entry")?;
        let name = entry.name().to_string();
        let mut content = Vec::new();
        entry
            .read_to_end(&mut content)
            .with_context(|| format!("Failed to read: {}", name))?;
        raw_items.insert(name, content);
    }

    // Step 2: Find OPF path
    let opf_path = find_opf_path(&raw_items)?;
    let opf_dir = if let Some(pos) = opf_path.rfind('/') {
        &opf_path[..=pos]
    } else {
        ""
    };

    // Step 3: Parse OPF manifest and spine
    let opf_content = raw_items
        .get(&opf_path)
        .ok_or_else(|| anyhow::anyhow!("OPF file not found: {}", opf_path))?;
    let opf_str = String::from_utf8_lossy(opf_content);

    let package = parse_opf_package(&opf_str, opf_dir)?;
    let html_manifest: HashMap<String, String> = package
        .manifest
        .iter()
        .filter(|(id, _)| {
            package.media_types.get(*id).is_some_and(|media_type| {
                media_type.contains("html") || media_type.contains("xhtml")
            })
        })
        .map(|(id, path)| (id.clone(), path.clone()))
        .collect();

    // Step 4: Load and extract text from content files in spine order
    let ordered_files = resolve_spine_order(&html_manifest, &package.spine_ids);

    log::debug!("EPUB spine has {} content files", ordered_files.len());

    let mut all_lines: Vec<String> = Vec::new();

    for (_, file_path) in &ordered_files {
        let content_bytes = match find_item_content(&raw_items, file_path) {
            Some(bytes) => bytes,
            None => {
                log::debug!("Content file not found in ZIP: {}", file_path);
                continue;
            }
        };

        let content_str = String::from_utf8_lossy(&content_bytes).to_string();

        // Preprocess XHTML for html5ever compatibility
        let content_str = fix_xhtml_for_html5(&content_str);
        let content_str = normalize_void_elements(&content_str);

        // Extract text lines from this document
        let lines = extract_lines_from_html(&content_str);
        all_lines.extend(lines);
    }

    Ok(all_lines)
}

// ── OPF Parsing ──────────────────────────────────────────────────────────

pub fn find_opf_path(raw_items: &HashMap<String, Vec<u8>>) -> Result<String> {
    // Try container.xml (regex, since it's namespaced XML)
    if let Some(container_bytes) = raw_items.get("META-INF/container.xml") {
        let container_str = String::from_utf8_lossy(container_bytes);
        if let Ok(doc) = XmlDocument::parse(&container_str) {
            for node in doc
                .descendants()
                .filter(|node| is_xml_element(*node, "rootfile"))
            {
                if let Some(path) = node.attribute("full-path") {
                    return Ok(path.to_string());
                }
            }
        }

        let rootfile_re = Regex::new(r#"<rootfile\s+([^>]*)/?>"#)?;
        for caps in rootfile_re.captures_iter(&container_str) {
            let attrs = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            if let Some(path) = extract_attr(attrs, "full-path") {
                return Ok(path);
            }
        }
    }

    // Fallback: find any .opf file
    for key in raw_items.keys() {
        if key.ends_with(".opf") {
            return Ok(key.clone());
        }
    }

    anyhow::bail!("Could not find OPF path in EPUB")
}

pub fn parse_opf_package(opf_str: &str, opf_dir: &str) -> Result<EpubPackage> {
    if let Ok(doc) = XmlDocument::parse(opf_str) {
        let mut package = EpubPackage::default();
        for item in doc
            .descendants()
            .filter(|node| is_xml_element(*node, "item"))
        {
            if let (Some(id), Some(href), Some(media_type)) = (
                item.attribute("id"),
                item.attribute("href"),
                item.attribute("media-type"),
            ) {
                let full_href = resolve_epub_href(opf_dir, href);
                package.manifest.insert(id.to_string(), full_href.clone());
                package
                    .media_types
                    .insert(id.to_string(), media_type.to_string());

                if item
                    .attribute("properties")
                    .is_some_and(|props| props.split_whitespace().any(|prop| prop == "nav"))
                {
                    package.toc_pages.insert(full_href);
                }
            }
        }

        package.spine_ids = doc
            .descendants()
            .filter(|node| is_xml_element(*node, "itemref"))
            .filter_map(|node| node.attribute("idref").map(|idref| idref.to_string()))
            .collect();

        for reference in doc
            .descendants()
            .filter(|node| is_xml_element(*node, "reference"))
        {
            if reference.attribute("type") == Some("toc") {
                if let Some(href) = reference.attribute("href") {
                    package.toc_pages.insert(resolve_epub_href(opf_dir, href));
                }
            }
        }

        return Ok(package);
    }

    // Fallback for malformed OPF: <item id="..." href="..." media-type="..." />
    let item_tag_re = Regex::new(r#"<item\s+([^>]*)/?>"#)?;

    let mut package = EpubPackage::default();

    for caps in item_tag_re.captures_iter(opf_str) {
        let attrs = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let id = extract_attr(attrs, "id");
        let href = extract_attr(attrs, "href");
        let media_type = extract_attr(attrs, "media-type");

        if let (Some(id), Some(href), Some(media_type)) = (id, href, media_type) {
            let full_href = resolve_epub_href(opf_dir, &href);
            package.manifest.insert(id.clone(), full_href.clone());
            package.media_types.insert(id, media_type);

            if extract_attr(attrs, "properties")
                .is_some_and(|props| props.split_whitespace().any(|prop| prop == "nav"))
            {
                package.toc_pages.insert(full_href);
            }
        }
    }

    // Spine: <itemref idref="..." />
    let spine_re = Regex::new(r#"<itemref\s+([^>]*)/?>"#)?;
    package.spine_ids = spine_re
        .captures_iter(opf_str)
        .filter_map(|c| c.get(1).and_then(|m| extract_attr(m.as_str(), "idref")))
        .collect();

    let guide_ref_re = Regex::new(r#"<reference\s+([^>]*)/?>"#)?;
    for caps in guide_ref_re.captures_iter(opf_str) {
        let attrs = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        if extract_attr(attrs, "type").as_deref() == Some("toc") {
            if let Some(href) = extract_attr(attrs, "href") {
                package.toc_pages.insert(resolve_epub_href(opf_dir, &href));
            }
        }
    }

    Ok(package)
}

pub fn resolve_spine_order(
    manifest: &HashMap<String, String>,
    spine_ids: &[String],
) -> Vec<(String, String)> {
    let mut ordered: Vec<(String, String)> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Spine order first
    for spine_id in spine_ids {
        if let Some(path) = manifest.get(spine_id) {
            if seen.insert(spine_id.clone()) {
                ordered.push((spine_id.clone(), path.clone()));
            }
        }
    }

    // Non-spine items at the end
    for (id, path) in manifest {
        if seen.insert(id.clone()) {
            ordered.push((id.clone(), path.clone()));
        }
    }

    ordered
}

pub fn find_item_content<'a>(
    raw_items: &'a HashMap<String, Vec<u8>>,
    name: &str,
) -> Option<&'a Vec<u8>> {
    // Exact match
    if let Some(content) = raw_items.get(name) {
        return Some(content);
    }

    // Try basename matching, requiring a path separator before the basename
    // to avoid partial matches (e.g. "old_chapter1.xhtml")
    let basename = name.rsplit('/').next().unwrap_or(name);
    let suffix = format!("/{}", basename);
    for (key, content) in raw_items {
        if key == basename || key.ends_with(&suffix) || key.ends_with(name) {
            return Some(content);
        }
    }

    None
}

// ── XHTML Preprocessing (from OT) ───────────────────────────────────────

/// Convert self-closing non-void tags for HTML5 parser compatibility.
/// e.g., `<script ... />` → `<script ...></script>`
pub fn fix_xhtml_for_html5(content: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"(?i)<(script|style|textarea|title|iframe|noscript|noframes)\b([^>]*?)\s*/>")
            .unwrap()
    });
    re.replace_all(content, "<$1$2></$1>").to_string()
}

/// Normalize void elements to match html5ever output.
/// e.g., `<br/>` → `<br>`, `<img .../>` → `<img ...>`
pub fn normalize_void_elements(content: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r#"(?i)<(br|hr|img|input|meta|link|col|area|base|embed|param|source|track|wbr)\b([^>]*?)\s*/>"#)
            .unwrap()
    });
    re.replace_all(content, "<$1$2>").to_string()
}

// ── Text Extraction ─────────────────────────────────────────────────────

/// Extract text lines from an HTML/XHTML document.
/// Each block-level element (p, h1-h6, li, div, etc.) that is a leaf
/// (no nested block children) produces one line.
/// Uses a combined CSS selector to preserve original document order.
pub fn extract_lines_from_html(html: &str) -> Vec<String> {
    extract_leaf_blocks_from_html(html)
        .into_iter()
        .map(|block| block.text)
        .collect()
}

pub fn extract_leaf_blocks_from_html(html: &str) -> Vec<EpubLeafBlock> {
    let fragment = Html::parse_document(html);
    let mut blocks = Vec::new();

    // Combined selector preserves document order (unlike per-tag iteration)
    let selector_str = EPUB_BLOCK_TAGS.join(", ");
    let selector = match Selector::parse(&selector_str) {
        Ok(s) => s,
        Err(_) => return blocks,
    };

    let mut block_index = 0usize;

    for element in fragment.select(&selector) {
        if has_orion_translation_class(&element) {
            continue;
        }

        // Skip elements with nested block children (only extract leaf blocks)
        let has_nested_block = element.descendants().skip(1).any(|node| {
            node.value()
                .as_element()
                .map_or(false, |e| EPUB_BLOCK_TAGS.iter().any(|bt| *bt == e.name()))
        });

        if has_nested_block {
            continue;
        }

        // Extract text, skipping <rt> (ruby annotations)
        let text = get_clean_text(&element);
        let trimmed = text.trim();

        if !trimmed.is_empty() {
            blocks.push(EpubLeafBlock {
                index: block_index,
                html: element.html(),
                text: trimmed.to_string(),
                raw_text: text,
            });
            block_index += 1;
        }
    }

    blocks
}

pub fn extract_attr(attrs: &str, name: &str) -> Option<String> {
    let escaped_name = regex::escape(name);
    for quote in ['"', '\''] {
        let pattern = format!(
            r#"(?i)\b{}\s*=\s*{}([^{}]*){}"#,
            escaped_name, quote, quote, quote
        );
        if let Ok(re) = Regex::new(&pattern) {
            if let Some(value) = re
                .captures(attrs)
                .and_then(|caps| caps.get(1).map(|m| m.as_str().to_string()))
            {
                return Some(value);
            }
        }
    }
    None
}

pub fn resolve_epub_href(base_dir: &str, href: &str) -> String {
    let href = href.split('#').next().unwrap_or(href);
    let combined = if href.starts_with('/') || base_dir.is_empty() {
        href.trim_start_matches('/').to_string()
    } else {
        format!("{}{}", base_dir, href)
    };

    let mut parts = Vec::new();
    for part in combined.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

fn is_xml_element(node: roxmltree::Node<'_, '_>, local_name: &str) -> bool {
    node.is_element() && node.tag_name().name().eq_ignore_ascii_case(local_name)
}

fn has_orion_translation_class(element: &scraper::ElementRef) -> bool {
    element.value().attr("class").is_some_and(|classes| {
        classes
            .split_whitespace()
            .any(|class_name| class_name == "orion-translation")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extraction_skips_injected_translation_blocks() {
        let html = r#"<html><body><p>原文一</p><p class="orion-translation">译文一</p><p>原文二</p></body></html>"#;
        assert_eq!(extract_lines_from_html(html), vec!["原文一", "原文二"]);
    }

    #[test]
    fn parse_opf_supports_single_quotes_and_relative_paths() {
        let opf = r#"<package><manifest><item id='chap' href='../Text/ch1.xhtml' media-type='application/xhtml+xml'/></manifest><spine><itemref idref='chap'/></spine></package>"#;
        let package = parse_opf_package(opf, "OPS/package/").unwrap();
        assert_eq!(
            package.manifest.get("chap"),
            Some(&"OPS/Text/ch1.xhtml".to_string())
        );
        assert_eq!(package.spine_ids, vec!["chap"]);
    }
}

/// Extract text from an element, ignoring <rt> (ruby annotation text).
/// Uses edge traversal to properly handle nested <rt> elements.
pub fn get_clean_text(element: &scraper::ElementRef) -> String {
    use ego_tree::iter::Edge;
    use scraper::Node;

    let mut text = String::new();
    let mut rt_depth = 0u32;

    for edge in element.traverse() {
        match edge {
            Edge::Open(node) => match node.value() {
                Node::Element(elem) if elem.name() == "rt" => {
                    rt_depth += 1;
                }
                Node::Text(t) if rt_depth == 0 => {
                    text.push_str(t);
                }
                _ => {}
            },
            Edge::Close(node) => {
                if let Node::Element(elem) = node.value() {
                    if elem.name() == "rt" && rt_depth > 0 {
                        rt_depth -= 1;
                    }
                }
            }
        }
    }

    text
}
