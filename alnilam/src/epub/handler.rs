use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::sync::OnceLock;

use anyhow::{Context, Result};
use regex::Regex;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};

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
        let mut archive = zip::ZipArchive::new(file)
            .with_context(|| "Failed to read EPUB as ZIP")?;

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
            if media_type.contains("html") || media_type.contains("xhtml") || media_type.contains("xml") {
                // Find the actual file in raw_items
                if let Some(content_bytes) = self.find_item_content(name) {
                    let content_str = String::from_utf8_lossy(&content_bytes).to_string();
                    // Preprocess XHTML for html5ever compatibility:
                    // 1. Convert self-closing non-void tags (<script/>, <style/>)
                    // 2. Normalize void elements (<br/> -> <br>) to match html5ever output
                    let content_str = Self::fix_xhtml_for_html5(&content_str);
                    let content_str = Self::normalize_void_elements(&content_str);
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
        let container_content = self
            .raw_items
            .get("META-INF/container.xml")
            .ok_or_else(|| anyhow::anyhow!("Missing META-INF/container.xml"))?;
        let container_str = String::from_utf8_lossy(container_content);

        // Parse with regex since it's simpler for this one file
        let re = Regex::new(r#"full-path="([^"]+)""#)?;
        let caps = re
            .captures(&container_str)
            .ok_or_else(|| anyhow::anyhow!("Could not find OPF path in container.xml"))?;
        Ok(caps
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("No capture group"))?
            .as_str()
            .to_string())
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

        // Parse manifest items: extract <item> tags and their id, href, media-type attributes
        // Use a flexible approach: first find all <item...> tags, then extract attrs individually
        let item_tag_re = Regex::new(r#"<item\s+([^>]*)/?>"#)?;
        let id_re = Regex::new(r#"id="([^"]*)""#)?;
        let href_re = Regex::new(r#"href="([^"]*)""#)?;
        let media_type_re = Regex::new(r#"media-type="([^"]*)""#)?;

        for caps in item_tag_re.captures_iter(&opf_str) {
            let attrs = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let id = id_re.captures(attrs).and_then(|c| c.get(1).map(|m| m.as_str().to_string()));
            let href = href_re.captures(attrs).and_then(|c| c.get(1).map(|m| m.as_str().to_string()));
            let media_type = media_type_re.captures(attrs).and_then(|c| c.get(1).map(|m| m.as_str().to_string()));

            if let (Some(id), Some(href), Some(media_type)) = (id, href, media_type) {
                let full_href = format!("{}{}", opf_dir, href);
                self.manifest.insert(id.clone(), full_href);
                self.media_types.insert(id, media_type);
            }
        }

        // Parse spine: <itemref idref="..." />
        let spine_re = Regex::new(r#"<itemref\s+[^>]*?idref="([^"]*)"[^>]*/?\s*>"#)?;
        for caps in spine_re.captures_iter(&opf_str) {
            if let Some(idref) = caps.get(1) {
                self.spine_ids.push(idref.as_str().to_string());
            }
        }

        // ── Detect ToC pages from EPUB metadata ─────────────────────────
        // 1. EPUB3: manifest items with properties="nav"
        let props_re = Regex::new(r#"properties="([^"]*)""#)?;
        for caps in item_tag_re.captures_iter(&opf_str) {
            let attrs = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let href = href_re.captures(attrs).and_then(|c| c.get(1).map(|m| m.as_str().to_string()));
            let props = props_re.captures(attrs).and_then(|c| c.get(1).map(|m| m.as_str().to_string()));
            if let (Some(href), Some(props)) = (href, props) {
                if props.split_whitespace().any(|p| p == "nav") {
                    let full_href = format!("{}{}", opf_dir, href);
                    tracing::info!("ToC page detected (EPUB3 nav): {}", full_href);
                    self.toc_pages.insert(full_href);
                }
            }
        }

        // 2. EPUB2: <guide> section with <reference type="toc" href="..." />
        let guide_ref_re = Regex::new(r#"<reference\s+[^>]*type="toc"[^>]*/?\s*>"#)?;
        for caps in guide_ref_re.captures_iter(&opf_str) {
            let tag_text = caps.get(0).map(|m| m.as_str()).unwrap_or("");
            if let Some(href_caps) = href_re.captures(tag_text) {
                if let Some(href) = href_caps.get(1) {
                    // Strip fragment identifier
                    let file_name = href.as_str().split('#').next().unwrap_or(href.as_str());
                    let full_href = format!("{}{}", opf_dir, file_name);
                    tracing::info!("ToC page detected (EPUB2 guide): {}", full_href);
                    self.toc_pages.insert(full_href);
                }
            }
        }

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

        // Parse <navPoint> with <navLabel><text>Title</text></navLabel> <content src="file.xhtml"/>
        let navpoint_re = Regex::new(
            r#"<navPoint[^>]*>[\s\S]*?<text>([^<]+)</text>[\s\S]*?<content\s+src="([^"]+)"[\s\S]*?</navPoint>"#,
        );
        if let Ok(re) = navpoint_re {
            for caps in re.captures_iter(ncx_str) {
                if let (Some(title), Some(src)) = (caps.get(1), caps.get(2)) {
                    let file_name = src.as_str().split('#').next().unwrap_or(src.as_str());
                    let full_path = format!("{}{}", ncx_dir, file_name);
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

        let link_re = Regex::new(r#"<a\s+[^>]*href="([^"]+)"[^>]*>([^<]+)</a>"#)?;
        for caps in link_re.captures_iter(nav_str) {
            if let (Some(href), Some(title)) = (caps.get(1), caps.get(2)) {
                let file_name = href.as_str().split('#').next().unwrap_or(href.as_str());
                let full_path = format!("{}{}", nav_dir, file_name);
                self.toc_map
                    .insert(full_path, title.as_str().trim().to_string());
            }
        }

        Ok(())
    }

    /// Find item content, trying both exact path and basename matching
    fn find_item_content(&self, name: &str) -> Option<Vec<u8>> {
        // Try exact match first
        if let Some(content) = self.raw_items.get(name) {
            return Some(content.clone());
        }

        // Try without leading path segments, requiring a path separator
        // before the basename to avoid partial matches (e.g. "old_chapter1.xhtml")
        let basename = name.rsplit('/').next().unwrap_or(name);
        let suffix = format!("/{}", basename);
        for (key, content) in &self.raw_items {
            if key == basename || key.ends_with(&suffix) || key.ends_with(name) {
                return Some(content.clone());
            }
        }

        None
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
            tracing::debug!("Processing doc: {} (id={}), content len={}", doc.name, doc.id, doc.content.len());
        }
        tracing::info!("Processing {} ordered documents", ordered_docs.len());

        for doc in ordered_docs {
            let title = self
                .toc_map
                .get(&doc.name)
                .cloned()
                .unwrap_or_else(|| "Unknown Section".to_string());

            // Content is already preprocessed during load()
            let fragment = Html::parse_document(&doc.content);
            let block_tags = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "div", "caption", "td", "th"];

            // Debug: count all elements in the parsed document
            let elem_count: usize = fragment.tree.values().filter(|n| n.as_element().is_some()).count();
            if elem_count == 0 {
                tracing::debug!("Doc '{}': HTML parse produced 0 elements from {} bytes", doc.name, doc.content.len());
            } else {
                tracing::debug!("Doc '{}': HTML parse found {} elements total", doc.name, elem_count);
            }

            let mut block_index = 0;
            let mut skipped_nested = 0;
            let mut empty_text = 0;
            let mut total_elements = 0;

            // Combined selector preserves document order (unlike per-tag iteration)
            let selector_str = block_tags.join(", ");
            if let Ok(selector) = Selector::parse(&selector_str) {
                for element in fragment.select(&selector) {
                    total_elements += 1;
                    // Check if it's a leaf block (no nested block children)
                    // Note: descendants() includes the element itself as the first item,
                    // so we skip(1) to only check actual descendants.
                    let has_nested_block = element.descendants().skip(1).any(|node| {
                        node.value().as_element().map_or(false, |e| {
                            block_tags.iter().any(|bt| *bt == e.name())
                        })
                    });

                    if has_nested_block {
                        skipped_nested += 1;
                        continue;
                    }

                    // Extract text, skipping <rt> (ruby text)
                    let text = Self::get_clean_text(&element);
                    let trimmed = text.trim();

                    if trimmed.is_empty() {
                        empty_text += 1;
                        continue;
                    }

                    // Determine page type from EPUB metadata
                    let page_type = if self.toc_pages.contains(&doc.name) {
                        "toc".to_string()
                    } else {
                        "content".to_string()
                    };

                    data.push(TranslationBlock {
                        file_id: doc.id.clone(),
                        file_name: doc.name.clone(),
                        title: title.clone(),
                        index: block_index,
                        src_text: trimmed.to_string(),
                        dst_text: None,
                        page_type,
                    });
                    block_index += 1;
                }
            }
            if total_elements > 0 || block_index > 0 {
                tracing::debug!(
                    "Doc '{}': {} elements found, {} skipped (nested), {} empty, {} extracted",
                    doc.name, total_elements, skipped_nested, empty_text, block_index
                );
            }
        }

        data
    }

    /// Extract text from element, ignoring <rt> (ruby annotation text)
    fn get_clean_text(element: &scraper::ElementRef) -> String {
        use ego_tree::iter::Edge;

        let mut text = String::new();
        let mut rt_depth = 0u32; // depth inside <rt> elements

        for edge in element.traverse() {
            match edge {
                Edge::Open(node) => {
                    if let Some(elem) = node.value().as_element() {
                        if elem.name() == "rt" {
                            rt_depth += 1;
                        }
                    }
                    if rt_depth == 0 {
                        if let Some(t) = node.value().as_text() {
                            text.push_str(&*t);
                        }
                    }
                }
                Edge::Close(node) => {
                    if let Some(elem) = node.value().as_element() {
                        if elem.name() == "rt" && rt_depth > 0 {
                            rt_depth -= 1;
                        }
                    }
                }
            }
        }

        // Don't trim here - preserve leading whitespace for indentation matching.
        // Callers should trim when needed.
        text
    }

    /// Preprocess XHTML content for html5ever (HTML5 parser).
    /// Converts self-closing non-void tags (e.g., `<script ... />`, `<style ... />`)
    /// into explicit open/close pairs, since HTML5 doesn't support self-closing
    /// for these elements.
    fn fix_xhtml_for_html5(content: &str) -> String {
        // Tags that are "raw text" or "RCDATA" in HTML5 and must not be self-closing
        static RE: OnceLock<Regex> = OnceLock::new();
        let re = RE.get_or_init(|| {
            Regex::new(r"(?i)<(script|style|textarea|title|iframe|noscript|noframes)\b([^>]*?)\s*/>")
                .unwrap()
        });
        re.replace_all(content, "<$1$2></$1>").to_string()
    }

    /// Normalize void elements to match html5ever serialization output.
    /// Converts `<br/>`, `<hr/>`, `<img .../>` etc. to `<br>`, `<hr>`, `<img ...>`.
    /// This ensures that element.html() output from scraper can be found in the source.
    fn normalize_void_elements(content: &str) -> String {
        static RE: OnceLock<Regex> = OnceLock::new();
        let re = RE.get_or_init(|| {
            Regex::new(r#"(?i)<(br|hr|img|input|meta|link|col|area|base|embed|param|source|track|wbr)\b([^>]*?)\s*/>"#)
                .unwrap()
        });
        re.replace_all(content, "<$1$2>").to_string()
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

        // Group blocks by file_id
        let mut by_file: HashMap<String, Vec<&TranslationBlock>> = HashMap::new();
        for block in data {
            if block.dst_text.is_some() {
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

            // html5ever reorders attributes during parsing, so element.html()
            // won't match the original source. We re-serialize the parsed document
            // to get canonical attribute order, then do matching on that.
            let fragment = Html::parse_document(&doc.content);
            let mut canonical_html = fragment.html();

            // html5ever converts <?xml ...?> to a comment and adds its own DOCTYPE.
            // Clean up: remove the <!--?xml ...?--> comment and the html5ever DOCTYPE.
            if let Ok(re) = Regex::new(r"<!--\?xml[^>]*\?-->") {
                canonical_html = re.replace_all(&canonical_html, "").to_string();
            }
            // Remove html5ever's DOCTYPE (we'll use the original preamble)
            if let Ok(re) = Regex::new(r"<!DOCTYPE[^>]*>") {
                canonical_html = re.replace(&canonical_html, "").to_string();
            }

            // Prepend original preamble (XML declaration + DOCTYPE) if present
            let preamble = if let Some(pos) = doc.content.find("<html") {
                doc.content[..pos].to_string()
            } else {
                String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE html>\n")
            };

            let mut new_content = canonical_html;

            let block_tags = [
                "p", "h1", "h2", "h3", "h4", "h5", "h6",
                "li", "blockquote", "div", "caption", "td", "th",
            ];

            // Collect all leaf block elements and their positions
            // (block_index, original_html, trimmed_text, raw_text_with_whitespace)
            let mut elements_info: Vec<(usize, String, String, String)> = Vec::new();
            let mut block_index = 0;

            // Combined selector preserves document order (unlike per-tag iteration)
            let selector_str = block_tags.join(", ");
            if let Ok(selector) = Selector::parse(&selector_str) {
                for element in fragment.select(&selector) {
                    // Check if it's a leaf block (no nested block children)
                    // descendants() includes the element itself, so skip(1)
                    let has_nested_block = element.descendants().skip(1).any(|node| {
                        node.value().as_element().map_or(false, |e| {
                            block_tags.iter().any(|bt| *bt == e.name())
                        })
                    });

                    if has_nested_block {
                        continue;
                    }

                    let raw_text = Self::get_clean_text(&element);
                    if !raw_text.trim().is_empty() {
                        let html = element.html();
                        elements_info.push((block_index, html, raw_text.trim().to_string(), raw_text));
                        block_index += 1;
                    }
                }
            }

            tracing::debug!(
                "Doc '{}': found {} leaf block elements for injection",
                doc.id, elements_info.len()
            );

            // Check if this document is a ToC page
            let is_toc_page = self.toc_pages.contains(&doc.name);

            // Apply translations (process in reverse to maintain positions)
            let mut match_failures = 0;
            for &block in blocks.iter().rev() {
                if let Some(translated) = &block.dst_text {
                    if let Some((_idx, ref original_html, _, ref raw_text)) = elements_info
                        .iter()
                        .find(|(idx, _, _, _)| *idx == block.index)
                    {
                        if !new_content.contains(original_html.as_str()) {
                            match_failures += 1;
                            if match_failures <= 5 {
                                let preview = if original_html.len() > 200 {
                                    &original_html[..200]
                                } else {
                                    original_html.as_str()
                                };
                                tracing::debug!(
                                    "Match failure in '{}' block {}: element.html() not found in source.\nelement.html(): {:?}",
                                    doc.id, block.index, preview
                                );
                            }
                            continue;
                        }

                        // ToC pages or blocks with page_type "toc": use "original / translation" format
                        let effective_toc = is_toc_page || block.page_type == "toc";
                        if effective_toc {
                            let toc_replaced = replace_tag_content_toc(original_html, &block.src_text, translated);
                            new_content = new_content.replacen(original_html, &toc_replaced, 1);
                            count += 1;
                        } else {
                            match mode {
                                crate::config::TranslationMode::Replace => {
                                    let replaced = replace_tag_content(original_html, translated, raw_text);
                                    new_content = new_content.replacen(original_html, &replaced, 1);
                                    count += 1;
                                }
                                crate::config::TranslationMode::Bilingual => {
                                    let new_tag =
                                        create_translation_tag(original_html, translated, raw_text, translation_gap);
                                    let replacement =
                                        format!("{}\n{}", original_html, new_tag);
                                    new_content =
                                        new_content.replacen(original_html, &replacement, 1);
                                    count += 1;
                                }
                            }
                        }
                    }
                }
            }
            if match_failures > 0 {
                tracing::warn!(
                    "Doc '{}': {} element.html() match failures (out of {} blocks)",
                    doc.id, match_failures, blocks.len()
                );
            }

            doc.content = format!("{}{}", preamble, new_content);
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
        for doc in &self.documents {
            // Find the matching raw item key
            for (key, _) in self.raw_items.clone() {
                if key == doc.name || key.ends_with(doc.name.rsplit('/').next().unwrap_or(&doc.name))
                {
                    self.raw_items
                        .insert(key, doc.content.as_bytes().to_vec());
                    break;
                }
            }
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
        let options =
            zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

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

    /// Apply format fixes (vertical->horizontal, reading direction, image styles)
    fn apply_format_fixes(&mut self) {
        format_fixer::apply_format_fixes(&mut self.raw_items, &mut self.documents);
    }

    // ── JSON I/O ─────────────────────────────────────────────────────────

    pub fn save_translation_data(
        data: &[TranslationBlock],
        output_path: &str,
    ) -> Result<()> {
        let json = serde_json::to_string_pretty(data)
            .context("Failed to serialize translation data")?;
        std::fs::write(output_path, json)
            .with_context(|| format!("Failed to write: {}", output_path))?;
        tracing::info!("Translation data saved to {}", output_path);
        Ok(())
    }

    pub fn load_translation_data(path: &str) -> Result<Vec<TranslationBlock>> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read: {}", path))?;
        let data: Vec<TranslationBlock> =
            serde_json::from_str(&content).context("Failed to parse translation data JSON")?;
        Ok(data)
    }
}

// ── Free helper functions (avoid borrow issues with &mut self) ───────────

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
                final_text,
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
    // Find the original text within the HTML and replace just the text,
    // preserving surrounding tags like <a href="..."> etc.
    if html.contains(original_text) {
        html.replacen(original_text, &toc_text, 1)
    } else {
        // Fallback: if exact text not found in HTML (e.g. split across child elements),
        // replace inner content of the outermost tag
        if let (Some(open_end), Some(close_start)) = (html.find('>'), html.rfind('<')) {
            if open_end < close_start {
                return format!(
                    "{}{}{}",
                    &html[..=open_end],
                    toc_text,
                    &html[close_start..]
                );
            }
        }
        html.to_string()
    }
}

/// Create a new translation tag to insert after the original
fn create_translation_tag(original_html: &str, translated: &str, raw_text: &str, gap: Option<&str>) -> String {
    let tag_name = original_html
        .trim_start_matches('<')
        .split(|c: char| c.is_whitespace() || c == '>')
        .next()
        .unwrap_or("p");

    let attrs_str = if let Some(open_end) = original_html.find('>') {
        let attrs_part = &original_html[tag_name.len() + 1..open_end];
        let id_re = Regex::new(r#"\s*id="[^"]*""#).unwrap_or(Regex::new("$^").expect("impossible"));
        let attrs = id_re.replace_all(attrs_part, "");
        // Add orion-translation class
        let class_re = Regex::new(r#"class="([^"]*)""#).unwrap_or(Regex::new("$^").expect("impossible"));
        let with_class = if class_re.is_match(&attrs) {
            class_re
                .replace(&attrs, r#"class="$1 orion-translation""#)
                .to_string()
        } else {
            format!("{} class=\"orion-translation\"", attrs.trim())
        };
        // Optionally add margin-bottom style for spacing
        if let Some(gap_value) = gap {
            let style_re = Regex::new(r#"style="([^"]*)""#).unwrap_or(Regex::new("$^").expect("impossible"));
            if style_re.is_match(&with_class) {
                style_re
                    .replace(&with_class, &format!("style=\"$1 margin-bottom:{};\"", gap_value))
                    .to_string()
            } else {
                format!("{} style=\"margin-bottom:{};\"", with_class.trim(), gap_value)
            }
        } else {
            with_class
        }
    } else if let Some(gap_value) = gap {
        format!(" class=\"orion-translation\" style=\"margin-bottom:{};\"", gap_value)
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

    format!("<{}{}>{}</{}>", tag_name, attrs_str, final_text, tag_name)
}
