mod epub;
mod txt;

pub use epub::{
    extract_attr, extract_epub_lines, extract_leaf_blocks_from_html, find_item_content,
    find_opf_path, fix_xhtml_for_html5, get_clean_text, normalize_void_elements, parse_opf_package,
    resolve_epub_href, resolve_spine_order, EpubLeafBlock, EpubPackage, EPUB_BLOCK_TAGS,
};
pub use txt::extract_txt_lines;
