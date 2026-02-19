use std::collections::HashMap;

use regex::Regex;

use super::handler::DocumentItem;

const SAFE_CSS: &str = r#"
/* Orion Image Fix */
img {
    max-width: 100% !important;
    height: auto !important;
    object-fit: contain;
}
svg {
    max-width: 100% !important;
    height: auto !important;
}
.orion-image-wrapper {
    width: 100% !important;
    text-align: center !important;
    margin: 0 !important;
    padding: 0 !important;
    page-break-inside: avoid;
    break-inside: avoid;
}
.orion-responsive-img {
    /* Mimic SVG preserveAspectRatio="xMidYMid meet":
       width:auto + max-width:100% handles landscape (width-constrained);
       height:auto + max-height:100vh handles portrait (height-constrained).
       Both can't be active simultaneously so aspect ratio is always preserved. */
    max-width: 100% !important;
    max-height: 100vh !important;
    width: auto !important;
    height: auto !important;
    object-fit: contain !important;
    display: block !important;
    margin: 0 auto !important;
}
"#;

/// Apply format fixes to the EPUB content:
/// 1. CSS: vertical-rl → horizontal-tb
/// 2. CSS: inject safe image styles
/// 3. OPF: set page-progression-direction to ltr
/// 4. HTML: simplify SVG-wrapped images to <img> tags
/// 5. HTML: fix inline vertical writing-mode styles
pub fn apply_format_fixes(
    raw_items: &mut HashMap<String, Vec<u8>>,
    documents: &mut [DocumentItem],
) {
    fix_css_writing_mode(raw_items);
    fix_opf_direction(raw_items);
    simplify_svg_images(documents);
    fix_html_inline_styles(documents);
}

/// Fix CSS: convert vertical-rl to horizontal-tb and inject safe image CSS
fn fix_css_writing_mode(raw_items: &mut HashMap<String, Vec<u8>>) {
    let vertical_re = Regex::new(
        r"(?i)((-webkit-|-epub-)?writing-mode\s*:\s*)vertical-[lr][rl]"
    );

    let mut css_fixed = false;
    let keys: Vec<String> = raw_items.keys().cloned().collect();

    for key in &keys {
        if key.ends_with(".css") || key.ends_with(".CSS") {
            if let Some(content) = raw_items.get(key) {
                if let Ok(css_str) = String::from_utf8(content.clone()) {
                    let mut new_css = css_str.clone();

                    // Fix vertical writing mode
                    if let Ok(ref re) = vertical_re {
                        new_css = re
                            .replace_all(&new_css, "${1}horizontal-tb")
                            .to_string();
                    }

                    // Add image fix CSS (only once, to the first CSS file)
                    if !css_fixed && !new_css.contains("Orion Image Fix") {
                        new_css.push_str(SAFE_CSS);
                        css_fixed = true;
                    }

                    if new_css != css_str {
                        raw_items.insert(key.clone(), new_css.into_bytes());
                    }
                }
            }
        }
    }
}

/// Fix OPF: set page-progression-direction to ltr
fn fix_opf_direction(raw_items: &mut HashMap<String, Vec<u8>>) {
    let keys: Vec<String> = raw_items.keys().cloned().collect();

    for key in &keys {
        if key.ends_with(".opf") {
            if let Some(content) = raw_items.get(key) {
                if let Ok(opf_str) = String::from_utf8(content.clone()) {
                    let mut new_opf = opf_str.clone();

                    // Replace existing page-progression-direction
                    if let Ok(re) = Regex::new(r#"page-progression-direction="[^"]*""#) {
                        if re.is_match(&new_opf) {
                            new_opf = re
                                .replace_all(&new_opf, r#"page-progression-direction="ltr""#)
                                .to_string();
                        } else {
                            // Insert page-progression-direction if not present
                            if let Ok(spine_re) = Regex::new(r#"<spine\b([^>]*?)>"#) {
                                new_opf = spine_re
                                    .replace(&new_opf, r#"<spine page-progression-direction="ltr"$1>"#)
                                    .to_string();
                            }
                        }
                    }

                    if new_opf != opf_str {
                        raw_items.insert(key.clone(), new_opf.into_bytes());
                    }
                }
            }
        }
    }
}

/// Simplify SVG-wrapped images to plain <img> tags.
/// Target: SVGs that contain only a single <image>, no complex shapes.
/// Matches Python fixer.py `simplify_svg_images()`.
fn simplify_svg_images(documents: &mut [DocumentItem]) {
    // Regex to find <svg>...</svg> blocks
    let svg_re = match Regex::new(r"(?si)<svg\b[^>]*>(.*?)</svg>") {
        Ok(r) => r,
        Err(_) => return,
    };
    // Regex to find <image> tags inside SVG
    let image_re = match Regex::new(r#"(?si)<image\b[^>]*/?\s*>"#) {
        Ok(r) => r,
        Err(_) => return,
    };
    // Regex to extract href from <image> (xlink:href or href)
    let href_re = match Regex::new(r#"(?:xlink:)?href="([^"]*)""#) {
        Ok(r) => r,
        Err(_) => return,
    };
    // Regex to extract width/height from <image> or SVG viewBox
    let img_width_re = Regex::new(r#"width="(\d+)""#).unwrap();
    let img_height_re = Regex::new(r#"height="(\d+)""#).unwrap();
    let viewbox_re = Regex::new(r#"viewBox="[\d.]+\s+[\d.]+\s+([\d.]+)\s+([\d.]+)""#).unwrap();
    // Complex SVG elements that indicate non-simple image wrapper
    let complex_re = match Regex::new(r"(?si)<(path|text|rect|circle|polygon)\b") {
        Ok(r) => r,
        Err(_) => return,
    };

    for doc in documents.iter_mut() {
        let original = doc.content.clone();
        let mut new_content = original.clone();
        let mut changed = false;

        // Find all SVG blocks (process in reverse to maintain positions)
        let matches: Vec<_> = svg_re.find_iter(&original).collect();
        for m in matches.into_iter().rev() {
            let svg_html = m.as_str();
            let inner = svg_re.captures(svg_html)
                .and_then(|c| c.get(1))
                .map(|m| m.as_str())
                .unwrap_or("");

            // Skip if contains complex elements
            if complex_re.is_match(inner) {
                continue;
            }

            // Count <image> tags - must be exactly 1
            let image_count = image_re.find_iter(inner).count();
            if image_count != 1 {
                continue;
            }

            // Extract image href
            let image_tag = match image_re.find(inner) {
                Some(m) => m.as_str(),
                None => continue,
            };
            let src = match href_re.captures(image_tag) {
                Some(c) => c.get(1).map(|m| m.as_str().to_string()),
                None => continue,
            };
            let src = match src {
                Some(s) => s,
                None => continue,
            };

            // Extract original dimensions from <image> width/height or SVG viewBox
            // This preserves the intrinsic aspect ratio for the replacement <img>
            let (img_w, img_h) = {
                // Try <image> width/height first
                let w = img_width_re.captures(image_tag).and_then(|c| c.get(1).map(|m| m.as_str().to_string()));
                let h = img_height_re.captures(image_tag).and_then(|c| c.get(1).map(|m| m.as_str().to_string()));
                if let (Some(w), Some(h)) = (w, h) {
                    (Some(w), Some(h))
                } else {
                    // Fall back to SVG viewBox
                    viewbox_re.captures(svg_html).map_or((None, None), |c| {
                        let vw = c.get(1).map(|m| m.as_str().to_string());
                        let vh = c.get(2).map(|m| m.as_str().to_string());
                        (vw, vh)
                    })
                }
            };

            // Build replacement with intrinsic dimensions + inline responsive style
            let dim_attrs = match (&img_w, &img_h) {
                (Some(w), Some(h)) => format!(r#" width="{}" height="{}""#, w, h),
                _ => String::new(),
            };
            let replacement = format!(
                r#"<div class="orion-image-wrapper"><img src="{}" class="orion-responsive-img"{} style="max-width:100%;max-height:100vh;width:auto;height:auto" alt="illustration"></div>"#,
                src, dim_attrs
            );

            new_content = new_content.replacen(svg_html, &replacement, 1);
            changed = true;
        }

        if changed {
            doc.content = new_content;
        }
    }
}

/// Fix inline vertical writing-mode styles in HTML body tags
fn fix_html_inline_styles(documents: &mut [DocumentItem]) {
    let vertical_style_re = match Regex::new(
        r"(?i)(writing-mode\s*:\s*)vertical-[lr][rl]"
    ) {
        Ok(r) => r,
        Err(_) => return,
    };

    for doc in documents.iter_mut() {
        if doc.content.contains("vertical-") {
            let new_content = vertical_style_re
                .replace_all(&doc.content, "${1}horizontal-tb")
                .to_string();
            if new_content != doc.content {
                doc.content = new_content;
            }
        }
    }
}
