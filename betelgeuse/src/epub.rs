use anyhow::{Context, Result};
use regex::Regex;
use scraper::{Html, Selector};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::OnceLock;

/// Block-level HTML tags that contain translatable text
const BLOCK_TAGS: &[&str] = &[
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "blockquote", "div", "caption", "td", "th",
];

/// Extract text lines from an EPUB file.
///
/// Each line corresponds to a block-level element (p, h1-h6, li, div, etc.)
/// from the EPUB content files, processed in spine order.
/// Ruby annotations (<rt> tags) are stripped.
pub fn extract_epub_lines(epub_path: &Path) -> Result<Vec<String>> {
    let file = std::fs::File::open(epub_path)
        .with_context(|| format!("EPUB文件不存在: {}", epub_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| "Failed to read EPUB as ZIP")?;

    // Step 1: Read all files into memory
    let mut raw_items: HashMap<String, Vec<u8>> = HashMap::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i).context("Failed to read ZIP entry")?;
        let name = entry.name().to_string();
        let mut content = Vec::new();
        entry.read_to_end(&mut content)
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

    let (manifest, spine_ids) = parse_opf(&opf_str, opf_dir)?;

    // Step 4: Load and extract text from content files in spine order
    let ordered_files = resolve_spine_order(&manifest, &spine_ids);

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

fn find_opf_path(raw_items: &HashMap<String, Vec<u8>>) -> Result<String> {
    // Try container.xml (regex, since it's namespaced XML)
    if let Some(container_bytes) = raw_items.get("META-INF/container.xml") {
        let container_str = String::from_utf8_lossy(container_bytes);
        let re = Regex::new(r#"full-path="([^"]+)""#)?;
        if let Some(caps) = re.captures(&container_str) {
            if let Some(m) = caps.get(1) {
                return Ok(m.as_str().to_string());
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

/// Parse OPF manifest and spine using regex (handles namespaced XML correctly)
fn parse_opf(
    opf_str: &str,
    opf_dir: &str,
) -> Result<(HashMap<String, String>, Vec<String>)> {
    // Manifest: <item id="..." href="..." media-type="..." />
    let item_tag_re = Regex::new(r#"<item\s+([^>]*)/?>"#)?;
    let id_re = Regex::new(r#"id="([^"]*)""#)?;
    let href_re = Regex::new(r#"href="([^"]*)""#)?;
    let media_type_re = Regex::new(r#"media-type="([^"]*)""#)?;

    let mut manifest: HashMap<String, String> = HashMap::new();

    for caps in item_tag_re.captures_iter(opf_str) {
        let attrs = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let id = id_re.captures(attrs).and_then(|c| c.get(1).map(|m| m.as_str().to_string()));
        let href = href_re.captures(attrs).and_then(|c| c.get(1).map(|m| m.as_str().to_string()));
        let media_type = media_type_re
            .captures(attrs)
            .and_then(|c| c.get(1).map(|m| m.as_str().to_string()));

        if let (Some(id), Some(href), Some(media_type)) = (id, href, media_type) {
            if media_type.contains("html") || media_type.contains("xhtml") {
                let full_href = format!("{}{}", opf_dir, href);
                manifest.insert(id, full_href);
            }
        }
    }

    // Spine: <itemref idref="..." />
    let spine_re = Regex::new(r#"<itemref\s+[^>]*?idref="([^"]*)"[^>]*/?\s*>"#)?;
    let spine_ids: Vec<String> = spine_re
        .captures_iter(opf_str)
        .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
        .collect();

    Ok((manifest, spine_ids))
}

fn resolve_spine_order(
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

fn find_item_content<'a>(raw_items: &'a HashMap<String, Vec<u8>>, name: &str) -> Option<&'a Vec<u8>> {
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
fn fix_xhtml_for_html5(content: &str) -> String {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"(?i)<(script|style|textarea|title|iframe|noscript|noframes)\b([^>]*?)\s*/>")
            .unwrap()
    });
    re.replace_all(content, "<$1$2></$1>").to_string()
}

/// Normalize void elements to match html5ever output.
/// e.g., `<br/>` → `<br>`, `<img .../>` → `<img ...>`
fn normalize_void_elements(content: &str) -> String {
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
fn extract_lines_from_html(html: &str) -> Vec<String> {
    let fragment = Html::parse_document(html);
    let mut lines = Vec::new();

    // Combined selector preserves document order (unlike per-tag iteration)
    let selector_str = BLOCK_TAGS.join(", ");
    let selector = match Selector::parse(&selector_str) {
        Ok(s) => s,
        Err(_) => return lines,
    };

    for element in fragment.select(&selector) {
        // Skip elements with nested block children (only extract leaf blocks)
        let has_nested_block = element.descendants().skip(1).any(|node| {
            node.value()
                .as_element()
                .map_or(false, |e| BLOCK_TAGS.iter().any(|bt| *bt == e.name()))
        });

        if has_nested_block {
            continue;
        }

        // Extract text, skipping <rt> (ruby annotations)
        let text = get_clean_text(&element);
        let trimmed = text.trim();

        if !trimmed.is_empty() {
            lines.push(trimmed.to_string());
        }
    }

    lines
}

/// Extract text from an element, ignoring <rt> (ruby annotation text).
/// Uses edge traversal to properly handle nested <rt> elements.
fn get_clean_text(element: &scraper::ElementRef) -> String {
    use ego_tree::iter::Edge;
    use scraper::Node;

    let mut text = String::new();
    let mut rt_depth = 0u32;

    for edge in element.traverse() {
        match edge {
            Edge::Open(node) => {
                match node.value() {
                    Node::Element(elem) if elem.name() == "rt" => {
                        rt_depth += 1;
                    }
                    Node::Text(t) if rt_depth == 0 => {
                        text.push_str(t);
                    }
                    _ => {}
                }
            }
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
