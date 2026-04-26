use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};

use anyhow::{Context, Result};
use betelgeuse::{
    extract_attr, extract_leaf_blocks_from_html, find_item_content as find_epub_item_content,
    find_opf_path as find_epub_opf_path, fix_xhtml_for_html5,
    get_clean_text as get_epub_clean_text, normalize_void_elements, parse_opf_package,
    resolve_epub_href,
};
use regex::Regex;
use roxmltree::Document as XmlDocument;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

use crate::config::validate_css_length;

use super::format_fixer;

// ── Data Types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationBlock {
    pub file_id: String,
    pub file_name: String,
    pub title: String,
    pub index: usize,
    pub src_text: String,
    pub dst_text: Option<String>,
    /// Page type: "toc" for table-of-contents pages, "content" for normal content
    #[serde(default = "default_page_type")]
    pub page_type: String,
}

fn default_page_type() -> String {
    "content".to_string()
}

/// Internal representation of an EPUB document item
#[derive(Debug)]
pub(crate) struct DocumentItem {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) content: String,
}

struct MatchedElement {
    index: usize,
    html: String,
    raw_text: String,
    start: usize,
    end: usize,
}

// ── EPUB Handler ─────────────────────────────────────────────────────────

pub struct EpubHandler {
    epub_path: String,
    /// Map: item_id -> DocumentItem
    documents: Vec<DocumentItem>,
    /// Map: file_name -> chapter title (from TOC)
    toc_map: HashMap<String, String>,
    /// All items in the ZIP
    raw_items: HashMap<String, Vec<u8>>,
    /// Spine order (list of item IDs)
    spine_ids: Vec<String>,
    /// Map: item_id -> file_name
    manifest: HashMap<String, String>,
    /// Map: item_id -> media_type
    media_types: HashMap<String, String>,
    /// Set of file names identified as ToC pages (from EPUB metadata)
    toc_pages: HashSet<String>,
}

impl EpubHandler {
    pub fn new(epub_path: &str) -> Self {
        Self {
            epub_path: epub_path.to_string(),
            documents: Vec::new(),
            toc_map: HashMap::new(),
            raw_items: HashMap::new(),
            spine_ids: Vec::new(),
            manifest: HashMap::new(),
            media_types: HashMap::new(),
            toc_pages: HashSet::new(),
        }
    }

    /// Load and parse the EPUB file
    pub fn load(&mut self) -> Result<()> {
        tracing::info!("Loading {}...", self.epub_path);
        let file = std::fs::File::open(&self.epub_path)
            .with_context(|| format!("Failed to open EPUB: {}", self.epub_path))?;
        let mut archive =
            zip::ZipArchive::new(file).with_context(|| "Failed to read EPUB as ZIP")?;

        // Read all files
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).context("Failed to read ZIP entry")?;
            let name = entry.name().to_string();
            let mut content = Vec::new();
            entry
                .read_to_end(&mut content)
                .with_context(|| format!("Failed to read: {}", name))?;
            self.raw_items.insert(name, content);
        }

        // Find and parse OPF
        let opf_path = self.find_opf_path()?;
        self.parse_opf(&opf_path)?;

        // Parse TOC
        self.build_toc_map()?;

        // Load document items
        for (id, name) in &self.manifest {
            let media_type = self.media_types.get(id).map(|s| s.as_str()).unwrap_or("");
            if media_type.contains("html")
                || media_type.contains("xhtml")
                || media_type.contains("xml")
            {
                // Find the actual file in raw_items
                if let Some(content_bytes) = self.find_item_content(name) {
                    let content_str = String::from_utf8_lossy(&content_bytes).to_string();
                    // Preprocess XHTML for html5ever compatibility:
                    // 1. Convert self-closing non-void tags (<script/>, <style/>)
                    // 2. Normalize void elements (<br/> -> <br>) to match html5ever output
                    let content_str = fix_xhtml_for_html5(&content_str);
                    let content_str = normalize_void_elements(&content_str);
                    self.documents.push(DocumentItem {
                        id: id.clone(),
                        name: name.clone(),
                        content: content_str,
                    });
                }
            }
        }

        tracing::info!("Loaded {} document items", self.documents.len());
        Ok(())
    }

    /// Find the OPF file path from container.xml
    fn find_opf_path(&self) -> Result<String> {
        find_epub_opf_path(&self.raw_items)
    }

    /// Parse the OPF (Open Packaging Format) file
    fn parse_opf(&mut self, opf_path: &str) -> Result<()> {
        let content = self
            .raw_items
            .get(opf_path)
            .ok_or_else(|| anyhow::anyhow!("OPF file not found: {}", opf_path))?;
        let opf_str = String::from_utf8_lossy(content);

        // Determine OPF directory for resolving relative paths
        let opf_dir = if let Some(pos) = opf_path.rfind('/') {
            &opf_path[..=pos]
        } else {
            ""
        };

        let package = parse_opf_package(&opf_str, opf_dir)?;
        for page in &package.toc_pages {
            tracing::info!("ToC page detected from OPF metadata: {}", page);
        }

        self.manifest = package.manifest;
        self.media_types = package.media_types;
        self.spine_ids = package.spine_ids;
        self.toc_pages = package.toc_pages;

        Ok(())
    }

    /// Build TOC map from NCX/NAV files
    fn build_toc_map(&mut self) -> Result<()> {
        // Collect candidates first to avoid borrow issues
        let manifest_entries: Vec<(String, String, String)> = self
            .manifest
            .iter()
            .map(|(id, name)| {
                let media_type = self.media_types.get(id).cloned().unwrap_or_default();
                (id.clone(), name.clone(), media_type)
            })
            .collect();

        // Try to find TOC NCX
        for (_id, name, media_type) in &manifest_entries {
            if media_type.contains("ncx") || name.ends_with(".ncx") {
                if let Some(content) = self.find_item_content(name) {
                    let ncx_str = String::from_utf8_lossy(&content).to_string();
                    self.parse_ncx_toc(&ncx_str, name)?;
                    return Ok(());
                }
            }
        }

        // Try NAV document
        for (_id, name, media_type) in &manifest_entries {
            if media_type.contains("xhtml") && name.contains("nav") {
                if let Some(content) = self.find_item_content(name) {
                    let nav_str = String::from_utf8_lossy(&content).to_string();
                    self.parse_nav_toc(&nav_str, name)?;
                    return Ok(());
                }
            }
        }

        // Fallback: use spine IDs
        let spine_ids = self.spine_ids.clone();
        for id in &spine_ids {
            if let Some(name) = self.manifest.get(id) {
                self.toc_map
                    .entry(name.clone())
                    .or_insert_with(|| id.clone());
            }
        }

        Ok(())
    }

    fn parse_ncx_toc(&mut self, ncx_str: &str, ncx_path: &str) -> Result<()> {
        let ncx_dir = if let Some(pos) = ncx_path.rfind('/') {
            &ncx_path[..=pos]
        } else {
            ""
        };

        if let Ok(doc) = XmlDocument::parse(ncx_str) {
            for nav_point in doc
                .descendants()
                .filter(|node| is_xml_element(*node, "navPoint"))
            {
                let title = nav_point
                    .descendants()
                    .find(|node| is_xml_element(*node, "text"))
                    .map(xml_node_text)
                    .filter(|text| !text.is_empty());
                let src = nav_point
                    .descendants()
                    .find(|node| is_xml_element(*node, "content"))
                    .and_then(|node| node.attribute("src"));
                if let (Some(title), Some(src)) = (title, src) {
                    let full_path = resolve_epub_href(ncx_dir, src);
                    self.toc_map.insert(full_path, title);
                }
            }
            return Ok(());
        }

        // Fallback for malformed NCX: parse common navPoint shape.
        let navpoint_re = Regex::new(
            r#"<navPoint[^>]*>[\s\S]*?<text>([^<]+)</text>[\s\S]*?<content\s+src="([^"]+)"[\s\S]*?</navPoint>"#,
        );
        if let Ok(re) = navpoint_re {
            for caps in re.captures_iter(ncx_str) {
                if let (Some(title), Some(src)) = (caps.get(1), caps.get(2)) {
                    let full_path = resolve_epub_href(ncx_dir, src.as_str());
                    self.toc_map
                        .insert(full_path, title.as_str().trim().to_string());
                }
            }
        }

        Ok(())
    }

    fn parse_nav_toc(&mut self, nav_str: &str, nav_path: &str) -> Result<()> {
        let nav_dir = if let Some(pos) = nav_path.rfind('/') {
            &nav_path[..=pos]
        } else {
            ""
        };

        if let Ok(doc) = XmlDocument::parse(nav_str) {
            for link in doc.descendants().filter(|node| is_xml_element(*node, "a")) {
                if let Some(href) = link.attribute("href") {
                    let title = xml_node_text(link);
                    if title.is_empty() {
                        continue;
                    }
                    let full_path = resolve_epub_href(nav_dir, href);
                    self.toc_map.insert(full_path, title);
                }
            }
            return Ok(());
        }

        let link_re = Regex::new(r#"<a\s+([^>]*)>([^<]+)</a>"#)?;
        for caps in link_re.captures_iter(nav_str) {
            if let (Some(attrs), Some(title)) = (caps.get(1), caps.get(2)) {
                let Some(href) = extract_attr(attrs.as_str(), "href") else {
                    continue;
                };
                let full_path = resolve_epub_href(nav_dir, &href);
                self.toc_map
                    .insert(full_path, title.as_str().trim().to_string());
            }
        }

        Ok(())
    }

    /// Find item content, trying both exact path and basename matching
    fn find_item_content(&self, name: &str) -> Option<Vec<u8>> {
        // Try exact match first
        find_epub_item_content(&self.raw_items, name).cloned()
    }

    // ── Extract translation data ─────────────────────────────────────────

    /// Extract translatable text blocks following spine order
    pub fn extract_translation_data(&self) -> Vec<TranslationBlock> {
        let mut data = Vec::new();

        // Process documents in spine order
        let ordered_docs: Vec<&DocumentItem> = if self.spine_ids.is_empty() {
            self.documents.iter().collect()
        } else {
            let mut ordered = Vec::new();
            for spine_id in &self.spine_ids {
                if let Some(doc) = self.documents.iter().find(|d| &d.id == spine_id) {
                    ordered.push(doc);
                }
            }
            // Add non-spine docs at the end
            for doc in &self.documents {
                if !self.spine_ids.contains(&doc.id) {
                    ordered.push(doc);
                }
            }
            ordered
        };

        for doc in &ordered_docs {
            tracing::debug!(
                "Processing doc: {} (id={}), content len={}",
                doc.name,
                doc.id,
                doc.content.len()
            );
        }
        tracing::info!("Processing {} ordered documents", ordered_docs.len());

        for doc in ordered_docs {
            let title = self
                .toc_map
                .get(&doc.name)
                .cloned()
                .unwrap_or_else(|| "Unknown Section".to_string());

            let page_type =
                if self.toc_pages.contains(&doc.name) || is_probable_toc_document(&doc.content) {
                    "toc".to_string()
                } else {
                    "content".to_string()
                };

            let leaf_blocks = extract_leaf_blocks_from_html(&doc.content);
            for leaf in &leaf_blocks {
                data.push(TranslationBlock {
                    file_id: doc.id.clone(),
                    file_name: doc.name.clone(),
                    title: title.clone(),
                    index: leaf.index,
                    src_text: leaf.text.clone(),
                    dst_text: None,
                    page_type: page_type.clone(),
                });
            }
            if !leaf_blocks.is_empty() {
                tracing::debug!(
                    "Doc '{}': {} leaf blocks extracted",
                    doc.name,
                    leaf_blocks.len()
                );
            }
        }

        data
    }

    // ── Inject translations ──────────────────────────────────────────────

    /// Inject translations back into the EPUB documents
    pub fn inject_translations(
        &mut self,
        data: &[TranslationBlock],
        mode: crate::config::TranslationMode,
        translation_gap: Option<&str>,
    ) -> Result<()> {
        tracing::info!("Injecting translations (mode: {})...", mode);
        if let Some(gap) = translation_gap {
            validate_css_length(gap)?;
        }

        // Group blocks by file_id
        let mut by_file: HashMap<String, Vec<&TranslationBlock>> = HashMap::new();
        for block in data {
            if block
                .dst_text
                .as_ref()
                .is_some_and(|translated| !translated.trim().is_empty())
            {
                by_file
                    .entry(block.file_id.clone())
                    .or_default()
                    .push(block);
            }
        }

        let mut count = 0;

        for doc in &mut self.documents {
            let blocks = match by_file.get(&doc.id) {
                Some(b) => b,
                None => continue,
            };

            let mut new_content = doc.content.clone();
            let elements_info = extract_leaf_blocks_from_html(&doc.content);

            tracing::debug!(
                "Doc '{}': found {} leaf block elements for injection",
                doc.id,
                elements_info.len()
            );

            let mut matched_elements = Vec::with_capacity(elements_info.len());
            let mut search_from = 0usize;
            let mut position_failures = 0usize;
            for leaf in elements_info {
                let index = leaf.index;
                let html = leaf.html;
                let raw_text = leaf.raw_text;
                if let Some(relative_start) = new_content[search_from..].find(&html) {
                    let start = search_from + relative_start;
                    let end = start + html.len();
                    matched_elements.push(MatchedElement {
                        index,
                        html,
                        raw_text,
                        start,
                        end,
                    });
                    search_from = end;
                } else if let Some((start, end, source_html)) = find_element_range_by_text(
                    &new_content,
                    tag_name_from_html(&html).unwrap_or(""),
                    raw_text.trim(),
                    search_from,
                ) {
                    matched_elements.push(MatchedElement {
                        index,
                        html: source_html,
                        raw_text,
                        start,
                        end,
                    });
                    search_from = end;
                } else {
                    position_failures += 1;
                    if position_failures <= 5 {
                        let preview = if html.len() > 200 {
                            &html[..200]
                        } else {
                            html.as_str()
                        };
                        tracing::debug!(
                            "Position failure in '{}' block {}: element.html() not found after byte {}.\nelement.html(): {:?}",
                            doc.id,
                            index,
                            search_from,
                            preview
                        );
                    }
                }
            }

            // Check if this document is a ToC page
            let is_toc_page =
                self.toc_pages.contains(&doc.name) || is_probable_toc_document(&doc.content);

            // Build replacements against exact byte ranges in the canonical HTML.
            // Replacing by range avoids corrupting documents with repeated identical blocks.
            let mut match_failures = 0;
            let mut replacements: Vec<(usize, usize, String)> = Vec::new();
            for &block in blocks {
                if let Some(translated) = &block.dst_text {
                    if let Some(matched) = matched_elements
                        .iter()
                        .find(|element| element.index == block.index)
                    {
                        // ToC pages or blocks with page_type "toc": use "original / translation" format
                        let effective_toc = is_toc_page || block.page_type == "toc";
                        let replacement = if effective_toc {
                            replace_tag_content_toc(&matched.html, &block.src_text, translated)
                        } else {
                            match mode {
                                crate::config::TranslationMode::Replace => replace_tag_content(
                                    &matched.html,
                                    translated,
                                    &matched.raw_text,
                                ),
                                crate::config::TranslationMode::Bilingual => {
                                    let new_tag = create_translation_tag(
                                        &matched.html,
                                        translated,
                                        &matched.raw_text,
                                        translation_gap,
                                    );
                                    if let Some((start, end)) =
                                        find_following_orion_translation_range(
                                            &new_content,
                                            matched.end,
                                        )
                                    {
                                        replacements.push((start, end, new_tag));
                                        count += 1;
                                        continue;
                                    }
                                    format!("{}\n{}", matched.html, new_tag)
                                }
                            }
                        };
                        replacements.push((matched.start, matched.end, replacement));
                        count += 1;
                    } else {
                        match_failures += 1;
                        if match_failures <= 5 {
                            tracing::debug!(
                                "Match failure in '{}' block {}: no matched element range",
                                doc.id,
                                block.index
                            );
                        }
                    }
                }
            }
            replacements.sort_by(|a, b| b.0.cmp(&a.0));
            for (start, end, replacement) in replacements {
                new_content.replace_range(start..end, &replacement);
            }
            if match_failures > 0 {
                tracing::warn!(
                    "Doc '{}': {} block match failures, {} position failures (out of {} blocks)",
                    doc.id,
                    match_failures,
                    position_failures,
                    blocks.len()
                );
            } else if position_failures > 0 {
                tracing::warn!(
                    "Doc '{}': {} element position failures (out of {} elements)",
                    doc.id,
                    position_failures,
                    matched_elements.len() + position_failures
                );
            }

            doc.content = new_content;
        }

        tracing::info!("Injected {} translations.", count);
        Ok(())
    }

    // ── Save ─────────────────────────────────────────────────────────────

    /// Save the modified EPUB
    pub fn save(&mut self, output_path: &str, apply_fixes: bool) -> Result<()> {
        tracing::info!("Saving to {}...", output_path);

        if apply_fixes {
            self.apply_format_fixes();
        }

        // Update raw_items with modified documents
        let updates: Vec<(String, Vec<u8>)> = self
            .documents
            .iter()
            .filter_map(|doc| {
                self.raw_item_key_for_document(&doc.name)
                    .map(|key| (key, doc.content.as_bytes().to_vec()))
                    .or_else(|| {
                        tracing::warn!("Could not find raw EPUB item for document '{}'", doc.name);
                        None
                    })
            })
            .collect();
        for (key, content) in updates {
            self.raw_items.insert(key, content);
        }

        // Write the new EPUB
        let output_file = std::fs::File::create(output_path)
            .with_context(|| format!("Failed to create output: {}", output_path))?;
        let mut zip_writer = zip::ZipWriter::new(output_file);

        // Write mimetype first (uncompressed, as required by EPUB spec)
        if let Some(mimetype) = self.raw_items.get("mimetype") {
            let options = zip::write::SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip_writer
                .start_file("mimetype", options)
                .context("Failed to write mimetype")?;
            zip_writer
                .write_all(mimetype)
                .context("Failed to write mimetype content")?;
        }

        // Write all other files
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);

        for (name, content) in &self.raw_items {
            if name == "mimetype" {
                continue;
            }
            zip_writer
                .start_file(name.as_str(), options)
                .with_context(|| format!("Failed to start file: {}", name))?;
            zip_writer
                .write_all(content)
                .with_context(|| format!("Failed to write: {}", name))?;
        }

        zip_writer.finish().context("Failed to finish ZIP")?;
        tracing::info!("Done.");
        Ok(())
    }

    fn raw_item_key_for_document(&self, doc_name: &str) -> Option<String> {
        if self.raw_items.contains_key(doc_name) {
            return Some(doc_name.to_string());
        }

        if doc_name.contains('/') {
            let full_suffix = format!("/{}", doc_name);
            let full_suffix_matches: Vec<&String> = self
                .raw_items
                .keys()
                .filter(|key| key.ends_with(&full_suffix))
                .collect();
            if full_suffix_matches.len() == 1 {
                return Some(full_suffix_matches[0].clone());
            }
            if full_suffix_matches.len() > 1 {
                return None;
            }
        }

        let basename = doc_name.rsplit('/').next().unwrap_or(doc_name);
        let suffix = format!("/{}", basename);
        let basename_matches: Vec<&String> = self
            .raw_items
            .keys()
            .filter(|key| key.as_str() == basename || key.ends_with(&suffix))
            .collect();

        if basename_matches.len() == 1 {
            Some(basename_matches[0].clone())
        } else {
            None
        }
    }

    /// Apply format fixes (vertical->horizontal, reading direction, image styles)
    fn apply_format_fixes(&mut self) {
        format_fixer::apply_format_fixes(&mut self.raw_items, &mut self.documents);
    }

    // ── JSON I/O ─────────────────────────────────────────────────────────

    pub fn save_translation_data(data: &[TranslationBlock], output_path: &str) -> Result<()> {
        let json =
            serde_json::to_string_pretty(data).context("Failed to serialize translation data")?;
        std::fs::write(output_path, json)
            .with_context(|| format!("Failed to write: {}", output_path))?;
        tracing::info!("Translation data saved to {}", output_path);
        Ok(())
    }

    pub fn load_translation_data(path: &str) -> Result<Vec<TranslationBlock>> {
        let content =
            std::fs::read_to_string(path).with_context(|| format!("Failed to read: {}", path))?;
        let data: Vec<TranslationBlock> =
            serde_json::from_str(&content).context("Failed to parse translation data JSON")?;
        Ok(data)
    }
}

// ── Free helper functions (avoid borrow issues with &mut self) ───────────

fn is_xml_element(node: roxmltree::Node<'_, '_>, local_name: &str) -> bool {
    node.is_element() && node.tag_name().name().eq_ignore_ascii_case(local_name)
}

fn xml_node_text(node: roxmltree::Node<'_, '_>) -> String {
    node.descendants()
        .filter_map(|child| child.text())
        .collect::<Vec<_>>()
        .join("")
        .trim()
        .to_string()
}

fn tag_name_from_html(html: &str) -> Option<&str> {
    html.trim_start_matches('<')
        .split(|c: char| c.is_whitespace() || c == '>')
        .next()
        .filter(|name| !name.is_empty())
}

fn find_element_range_by_text(
    content: &str,
    tag_name: &str,
    expected_text: &str,
    search_from: usize,
) -> Option<(usize, usize, String)> {
    if tag_name.is_empty() || expected_text.is_empty() || search_from >= content.len() {
        return None;
    }

    let tag = regex::escape(tag_name);
    let pattern = format!(r"(?is)<{}\b[^>]*>.*?</{}>", tag, tag);
    let re = Regex::new(&pattern).ok()?;
    let selector = Selector::parse(tag_name).ok()?;

    for candidate in re.find_iter(&content[search_from..]) {
        let candidate_html = candidate.as_str();
        let fragment = Html::parse_fragment(candidate_html);
        if let Some(element) = fragment.select(&selector).next() {
            if get_epub_clean_text(&element).trim() == expected_text {
                let start = search_from + candidate.start();
                let end = search_from + candidate.end();
                return Some((start, end, candidate_html.to_string()));
            }
        }
    }

    None
}

fn escape_html_text(text: &str) -> String {
    text.chars()
        .flat_map(|c| match c {
            '&' => "&amp;".chars().collect::<Vec<_>>(),
            '<' => "&lt;".chars().collect(),
            '>' => "&gt;".chars().collect(),
            _ => vec![c],
        })
        .collect()
}

fn escape_html_attribute(text: &str) -> String {
    escape_html_text(text).replace('"', "&quot;")
}

fn remove_attr(attrs: &str, name: &str) -> String {
    let escaped_name = regex::escape(name);
    let pattern = format!(
        r#"(?i)\s+{}\s*=\s*"[^"]*"|\s+{}\s*=\s*'[^']*'"#,
        escaped_name, escaped_name
    );
    Regex::new(&pattern)
        .map(|re| re.replace_all(attrs, "").to_string())
        .unwrap_or_else(|_| attrs.to_string())
}

fn append_attr(attrs: &mut String, name: &str, value: &str) {
    if !attrs.trim().is_empty() {
        attrs.push(' ');
    }
    attrs.push_str(name);
    attrs.push_str("=\"");
    attrs.push_str(&escape_html_attribute(value));
    attrs.push('"');
}

fn find_following_orion_translation_range(content: &str, from: usize) -> Option<(usize, usize)> {
    if from >= content.len() {
        return None;
    }

    let tail = &content[from..];
    let leading_ws = tail.len() - tail.trim_start().len();
    let start = from + leading_ws;
    if !content[start..].starts_with('<') {
        return None;
    }

    let open_end = start + content[start..].find('>')?;
    let open_tag = &content[start..=open_end];
    if !extract_attr(open_tag, "class").is_some_and(|classes| {
        classes
            .split_whitespace()
            .any(|class_name| class_name == "orion-translation")
    }) {
        return None;
    }

    let tag_name = tag_name_from_html(open_tag)?;
    let close_pattern = format!(r"(?is)</\s*{}\s*>", regex::escape(tag_name));
    let close_re = Regex::new(&close_pattern).ok()?;
    let after_open = open_end + 1;
    let close = close_re.find(&content[after_open..])?;
    Some((start, after_open + close.end()))
}

fn is_probable_toc_document(content: &str) -> bool {
    let fragment = Html::parse_document(content);
    let Ok(link_selector) = Selector::parse("a[href]") else {
        return false;
    };
    let link_count = fragment.select(&link_selector).count();
    if link_count < 3 {
        return false;
    }

    let Ok(block_selector) = Selector::parse("p, li, div") else {
        return false;
    };
    let block_count = fragment.select(&block_selector).count().max(1);
    link_count * 2 >= block_count
}

/// Extract leading whitespace from raw text (preserves full-width spaces, half-width spaces, tabs)
fn extract_leading_whitespace(raw_text: &str) -> &str {
    let first_non_ws = raw_text.find(|c: char| !c.is_whitespace());
    match first_non_ws {
        Some(0) => "",
        Some(pos) => &raw_text[..pos],
        None => raw_text, // all whitespace
    }
}

/// Replace the text content of an HTML tag, preserving leading whitespace from original
fn replace_tag_content(html: &str, new_text: &str, raw_text: &str) -> String {
    if let (Some(open_end), Some(close_start)) = (html.find('>'), html.rfind('<')) {
        if open_end < close_start {
            let leading_ws = extract_leading_whitespace(raw_text);
            let final_text = if !leading_ws.is_empty() && !new_text.starts_with(leading_ws) {
                format!("{}{}", leading_ws, new_text.trim_start())
            } else {
                new_text.to_string()
            };
            return format!(
                "{}{}{}",
                &html[..=open_end],
                escape_html_text(&final_text),
                &html[close_start..]
            );
        }
    }
    html.to_string()
}

/// Replace tag content for ToC pages: "original / translation" format.
/// Only replaces the text content, preserving all HTML structure (<a> links, <span>, etc.)
fn replace_tag_content_toc(html: &str, original_text: &str, translated: &str) -> String {
    let toc_text = format!("{} / {}", original_text, translated);
    let escaped_toc_text = escape_html_text(&toc_text);
    if html.contains(&toc_text) || html.contains(&escaped_toc_text) {
        return html.to_string();
    }

    // Find the original text within the HTML and replace just the text,
    // preserving surrounding tags like <a href="..."> etc.
    if html.contains(original_text) {
        html.replacen(original_text, &escaped_toc_text, 1)
    } else if html.contains(&escape_html_text(original_text)) {
        html.replacen(&escape_html_text(original_text), &escaped_toc_text, 1)
    } else {
        // Fallback: if exact text not found in HTML (e.g. split across child elements),
        // replace inner content of the outermost tag
        if let (Some(open_end), Some(close_start)) = (html.find('>'), html.rfind('<')) {
            if open_end < close_start {
                return format!(
                    "{}{}{}",
                    &html[..=open_end],
                    escaped_toc_text,
                    &html[close_start..]
                );
            }
        }
        html.to_string()
    }
}

/// Create a new translation tag to insert after the original
fn create_translation_tag(
    original_html: &str,
    translated: &str,
    raw_text: &str,
    gap: Option<&str>,
) -> String {
    let tag_name = original_html
        .trim_start_matches('<')
        .split(|c: char| c.is_whitespace() || c == '>')
        .next()
        .unwrap_or("p");

    let attrs_str = if let Some(open_end) = original_html.find('>') {
        let attrs_part = &original_html[tag_name.len() + 1..open_end];
        let existing_class = extract_attr(attrs_part, "class").unwrap_or_default();
        let existing_style = extract_attr(attrs_part, "style").unwrap_or_default();
        let mut attrs = remove_attr(attrs_part, "id");
        attrs = remove_attr(&attrs, "class");
        attrs = remove_attr(&attrs, "style");

        let mut attrs_out = attrs.trim().to_string();
        let class_value = if existing_class.trim().is_empty() {
            "orion-translation".to_string()
        } else {
            format!("{} orion-translation", existing_class.trim())
        };
        append_attr(&mut attrs_out, "class", &class_value);

        let mut style_value = existing_style.trim().to_string();
        if let Some(gap_value) = gap {
            if !style_value.is_empty() && !style_value.ends_with(';') {
                style_value.push(';');
            }
            if !style_value.is_empty() {
                style_value.push(' ');
            }
            style_value.push_str("margin-bottom:");
            style_value.push_str(gap_value);
            style_value.push(';');
        }
        if !style_value.is_empty() {
            append_attr(&mut attrs_out, "style", &style_value);
        }

        attrs_out
    } else if let Some(gap_value) = gap {
        format!(
            " class=\"orion-translation\" style=\"margin-bottom:{};\"",
            escape_html_attribute(gap_value)
        )
    } else {
        " class=\"orion-translation\"".to_string()
    };

    // Ensure attrs_str starts with a space (for "<tag attrs>")
    let attrs_str = if !attrs_str.is_empty() && !attrs_str.starts_with(' ') {
        format!(" {}", attrs_str)
    } else {
        attrs_str
    };

    // Preserve leading whitespace from the original text
    let leading_ws = extract_leading_whitespace(raw_text);
    let final_text = if !leading_ws.is_empty() && !translated.starts_with(leading_ws) {
        format!("{}{}", leading_ws, translated.trim_start())
    } else {
        translated.to_string()
    };

    format!(
        "<{}{}>{}</{}>",
        tag_name,
        attrs_str,
        escape_html_text(&final_text),
        tag_name
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TranslationMode;

    fn test_handler(content: &str) -> EpubHandler {
        EpubHandler {
            epub_path: String::new(),
            documents: vec![DocumentItem {
                id: "chapter".to_string(),
                name: "Text/chapter.xhtml".to_string(),
                content: content.to_string(),
            }],
            toc_map: HashMap::new(),
            raw_items: HashMap::new(),
            spine_ids: Vec::new(),
            manifest: HashMap::new(),
            media_types: HashMap::new(),
            toc_pages: HashSet::new(),
        }
    }

    fn block(index: usize, src_text: &str, dst_text: &str) -> TranslationBlock {
        TranslationBlock {
            file_id: "chapter".to_string(),
            file_name: "Text/chapter.xhtml".to_string(),
            title: "Chapter".to_string(),
            index,
            src_text: src_text.to_string(),
            dst_text: Some(dst_text.to_string()),
            page_type: "content".to_string(),
        }
    }

    fn unique_temp_epub(name: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "orion_{}_{}_{}.epub",
            name,
            std::process::id(),
            nonce
        ))
    }

    fn write_epub_entry(
        zip: &mut zip::ZipWriter<std::fs::File>,
        name: &str,
        content: &str,
        stored: bool,
    ) {
        let method = if stored {
            zip::CompressionMethod::Stored
        } else {
            zip::CompressionMethod::Deflated
        };
        let options = zip::write::SimpleFileOptions::default().compression_method(method);
        zip.start_file(name, options).unwrap();
        zip.write_all(content.as_bytes()).unwrap();
    }

    fn write_minimal_epub(path: &std::path::Path) {
        let file = std::fs::File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        write_epub_entry(&mut zip, "mimetype", "application/epub+zip", true);
        write_epub_entry(
            &mut zip,
            "META-INF/container.xml",
            r#"<?xml version='1.0'?><container version='1.0' xmlns='urn:oasis:names:tc:opendocument:xmlns:container'><rootfiles><rootfile media-type='application/oebps-package+xml' full-path='OPS/package/content.opf'/></rootfiles></container>"#,
            false,
        );
        write_epub_entry(
            &mut zip,
            "OPS/package/content.opf",
            r#"<?xml version='1.0'?><package xmlns='http://www.idpf.org/2007/opf' version='3.0'><manifest><item id='nav' href='../Text/nav.xhtml' media-type='application/xhtml+xml' properties='nav'/><item id='chap' href='../Text/ch1.xhtml' media-type='application/xhtml+xml'/></manifest><spine><itemref idref='chap'/></spine></package>"#,
            false,
        );
        write_epub_entry(
            &mut zip,
            "OPS/Text/nav.xhtml",
            r#"<?xml version='1.0'?><html xmlns='http://www.w3.org/1999/xhtml' xmlns:epub='http://www.idpf.org/2007/ops'><body><nav epub:type='toc'><ol><li><a href='../Text/ch1.xhtml#p1'><span>第一章</span></a></li></ol></nav></body></html>"#,
            false,
        );
        write_epub_entry(
            &mut zip,
            "OPS/Text/ch1.xhtml",
            r#"<?xml version='1.0'?><html xmlns='http://www.w3.org/1999/xhtml'><body><p id='p1'>「そうだな」</p><p>「そうだな」</p><p><ruby>太郎<rt>たろう</rt></ruby>は走った。</p></body></html>"#,
            false,
        );
        zip.finish().unwrap();
    }

    fn read_epub_entry(path: &std::path::Path, name: &str) -> String {
        let file = std::fs::File::open(path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();
        let mut entry = archive.by_name(name).unwrap();
        let mut content = String::new();
        entry.read_to_string(&mut content).unwrap();
        content
    }

    #[test]
    fn injects_duplicate_source_blocks_by_position() {
        let mut handler = test_handler(
            r#"<?xml version="1.0" encoding="utf-8"?><html><body><p class="calibre2">「そうだな」</p><p class="calibre2">「そうだな」</p></body></html>"#,
        );
        let blocks = vec![
            block(0, "「そうだな」", "「是啊。」"),
            block(1, "「そうだな」", "「没错。」"),
        ];

        handler
            .inject_translations(&blocks, TranslationMode::Bilingual, None)
            .unwrap();

        let content = &handler.documents[0].content;
        let original_positions: Vec<usize> = content
            .match_indices(r#"<p class="calibre2">「そうだな」</p>"#)
            .map(|(position, _)| position)
            .collect();
        assert_eq!(original_positions.len(), 2);

        let first_translation = content.find("「是啊。」").unwrap();
        let second_translation = content.find("「没错。」").unwrap();
        assert!(original_positions[0] < first_translation);
        assert!(first_translation < original_positions[1]);
        assert!(original_positions[1] < second_translation);
        assert_eq!(content.matches("orion-translation").count(), 2);
    }

    #[test]
    fn extraction_skips_existing_translation_blocks() {
        let handler = test_handler(
            r#"<html><body><p>原文一</p><p class="calibre2 orion-translation">译文一</p><p>原文二</p></body></html>"#,
        );

        let data = handler.extract_translation_data();

        let texts: Vec<&str> = data.iter().map(|block| block.src_text.as_str()).collect();
        assert_eq!(texts, vec!["原文一", "原文二"]);
        assert_eq!(data[0].index, 0);
        assert_eq!(data[1].index, 1);
    }

    #[test]
    fn raw_item_key_prefers_exact_document_path() {
        let mut handler = test_handler("<html></html>");
        handler
            .raw_items
            .insert("OPS/chapter.xhtml".to_string(), b"wrong".to_vec());
        handler
            .raw_items
            .insert("Text/chapter.xhtml".to_string(), b"right".to_vec());

        assert_eq!(
            handler.raw_item_key_for_document("Text/chapter.xhtml"),
            Some("Text/chapter.xhtml".to_string())
        );
    }

    #[test]
    fn parse_opf_supports_single_quotes_and_relative_paths() {
        let mut handler = test_handler("<html></html>");
        handler.raw_items.insert(
            "OPS/package/content.opf".to_string(),
            br#"<package><manifest><item id='nav' href='../Text/nav.xhtml' media-type='application/xhtml+xml' properties='nav'/><item id='chap' href='../Text/ch1.xhtml' media-type='application/xhtml+xml'/></manifest><spine><itemref idref='chap'/></spine><guide><reference type='toc' href='../Text/nav.xhtml#toc'/></guide></package>"#.to_vec(),
        );

        handler.parse_opf("OPS/package/content.opf").unwrap();

        assert_eq!(
            handler.manifest.get("chap"),
            Some(&"OPS/Text/ch1.xhtml".to_string())
        );
        assert_eq!(handler.spine_ids, vec!["chap"]);
        assert!(handler.toc_pages.contains("OPS/Text/nav.xhtml"));
    }

    #[test]
    fn injection_escapes_translation_text() {
        let mut handler = test_handler(r#"<html><body><p>原文</p></body></html>"#);
        let blocks = vec![block(0, "原文", r#"<script>alert("x")</script> & done"#)];

        handler
            .inject_translations(&blocks, TranslationMode::Bilingual, Some("1rem"))
            .unwrap();

        let content = &handler.documents[0].content;
        assert!(content.contains("&lt;script&gt;alert(\"x\")&lt;/script&gt; &amp; done"));
        assert!(!content.contains("<script>alert"));
    }

    #[test]
    fn injection_rejects_unsafe_gap() {
        let mut handler = test_handler(r#"<html><body><p>原文</p></body></html>"#);
        let blocks = vec![block(0, "原文", "译文")];

        let err = handler
            .inject_translations(&blocks, TranslationMode::Bilingual, Some("1rem; color:red"))
            .unwrap_err();

        assert!(err.to_string().contains("译文间距"));
    }

    #[test]
    fn injection_preserves_document_preamble_and_original_block() {
        let original = r#"<?xml version="1.0" encoding="utf-8"?><!DOCTYPE html><html><body><p data-x="1" class="a">原文</p></body></html>"#;
        let mut handler = test_handler(original);
        let blocks = vec![block(0, "原文", "译文")];

        handler
            .inject_translations(&blocks, TranslationMode::Bilingual, None)
            .unwrap();

        let content = &handler.documents[0].content;
        assert!(content.starts_with(r#"<?xml version="1.0" encoding="utf-8"?><!DOCTYPE html>"#));
        assert!(content.contains(r#"<p data-x="1" class="a">原文</p>"#));
        assert!(content.contains(r#"class="a orion-translation""#));
        assert!(!content.contains(r#"id="p1" class="a orion-translation""#));
        assert_eq!(content.matches(r#"<html"#).count(), 1);
    }

    #[test]
    fn bilingual_injection_replaces_existing_translation_block() {
        let mut handler = test_handler(
            r#"<html><body><p class='body' id='p1'>原文</p>
<p class='body orion-translation' style='margin-bottom:1rem;'>旧译文</p></body></html>"#,
        );
        let blocks = vec![block(0, "原文", "新译文")];

        handler
            .inject_translations(&blocks, TranslationMode::Bilingual, Some("1rem"))
            .unwrap();
        handler
            .inject_translations(&blocks, TranslationMode::Bilingual, Some("1rem"))
            .unwrap();

        let content = &handler.documents[0].content;
        assert_eq!(content.matches("orion-translation").count(), 1);
        assert!(content.contains("新译文"));
        assert!(!content.contains("旧译文"));
        assert!(!content.contains("id='p1' class=\"body orion-translation\""));
    }

    #[test]
    fn toc_injection_is_idempotent() {
        let mut handler = test_handler(
            r#"<html><body><nav><ol><li><a href='ch1.xhtml'><span>第一章</span></a></li></ol></nav></body></html>"#,
        );
        handler.toc_pages.insert("Text/chapter.xhtml".to_string());
        let mut toc_block = block(0, "第一章", "第一章");
        toc_block.page_type = "toc".to_string();
        let blocks = vec![toc_block];

        handler
            .inject_translations(&blocks, TranslationMode::Bilingual, None)
            .unwrap();
        handler
            .inject_translations(&blocks, TranslationMode::Bilingual, None)
            .unwrap();

        let content = &handler.documents[0].content;
        assert_eq!(content.matches("第一章 / 第一章").count(), 1);
        assert!(content.contains(r#"href='ch1.xhtml'"#));
    }

    #[test]
    fn minimal_epub_round_trip_preserves_structure_and_links() {
        let input = unique_temp_epub("input");
        let output = unique_temp_epub("output");
        write_minimal_epub(&input);

        let mut handler = EpubHandler::new(&input.to_string_lossy());
        handler.load().unwrap();
        let mut data = handler.extract_translation_data();
        assert!(data.iter().any(|block| block.src_text == "「そうだな」"));

        let mut duplicate_seen = 0usize;
        for block in &mut data {
            block.dst_text = Some(match block.src_text.as_str() {
                "「そうだな」" => {
                    duplicate_seen += 1;
                    if duplicate_seen == 1 {
                        "「是啊。」".to_string()
                    } else {
                        "「没错。」".to_string()
                    }
                }
                "太郎は走った。" => "太郎跑了。".to_string(),
                "第一章" => "第一章".to_string(),
                other => format!("{}-译", other),
            });
        }

        handler
            .inject_translations(&data, TranslationMode::Bilingual, Some("1rem"))
            .unwrap();
        handler.save(&output.to_string_lossy(), false).unwrap();

        let chapter = read_epub_entry(&output, "OPS/Text/ch1.xhtml");
        assert!(chapter.contains(r#"<p id='p1'>「そうだな」</p>"#));
        assert!(chapter.contains("「是啊。」"));
        assert!(chapter.contains("「没错。」"));
        assert!(chapter.contains("太郎跑了。"));
        assert_eq!(chapter.matches("orion-translation").count(), 3);
        assert!(XmlDocument::parse(&chapter).is_ok());

        let nav = read_epub_entry(&output, "OPS/Text/nav.xhtml");
        assert!(nav.contains(r#"href='../Text/ch1.xhtml#p1'"#));
        assert!(nav.contains("第一章 / 第一章"));
        assert!(XmlDocument::parse(&nav).is_ok());

        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
    }
}
