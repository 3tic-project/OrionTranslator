use crate::llm::{canonical_key, contains_kana, is_pure_title, strip_affixes, TranslationEntry};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsStr;
use std::path::Path;

pub fn analyze_output_json(path: &Path) -> Result<()> {
    let content = std::fs::read_to_string(path)?;
    let entries: Vec<TranslationEntry> = serde_json::from_str(&content)?;
    analyze_entries(&entries, Some(path.display().to_string()));
    Ok(())
}

fn print_cluster_variants(
    variants: &[(&str, usize, usize, usize)],
    clusters: &BTreeMap<String, Vec<&TranslationEntry>>,
    limit: usize,
) {
    if variants.is_empty() {
        return;
    }
    println!("\n[canonical簇内别名变体] 样例（最多{}组）", limit);
    for (key, count, dstn, _infon) in variants.iter().take(limit) {
        println!("  {}: {}条（dst {}种）", key, count, dstn);
        if let Some(list) = clusters.get(*key) {
            let mut uniq = HashSet::new();
            for x in list {
                if uniq.insert((x.src.as_str(), x.dst.as_str(), x.info.as_str())) {
                    println!("    - {} -> {}   #{}", x.src, x.dst, x.info);
                }
            }
        }
    }
}

fn normalize_dst_base(dst: &str) -> String {
    let mut s = dst.replace('\u{3000}', "").trim().to_string();
    let suffixes = [
        "酱",
        "君",
        "前辈",
        "老师",
        "小姐",
        "先生",
        "学姐",
        "学长",
        "姐",
        "哥",
        "妹妹",
        "弟弟",
        "部长",
        "会长",
        "委员长",
        "店长",
        "课长",
        "社长",
        "监督",
        "大人",
    ];
    loop {
        let mut changed = false;
        for suf in suffixes {
            if s.ends_with(suf) && s.chars().count() > suf.chars().count() {
                s = s.strip_suffix(suf).unwrap_or(&s).to_string();
                changed = true;
                break;
            }
        }
        if !changed {
            break;
        }
    }
    s
}

pub fn analyze_characters_json(path: &Path) -> Result<()> {
    let content = std::fs::read_to_string(path)?;
    let characters: HashMap<String, crate::detector::CharacterInfo> =
        serde_json::from_str(&content)?;
    analyze_characters(&characters, Some(path.display().to_string()));
    Ok(())
}

pub fn analyze_output_dir(dir: &Path) -> Result<()> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension() != Some(OsStr::new("json")) {
            continue;
        }
        if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
            if name.ends_with("_output.json") || name.ends_with("_glossary.json") {
                files.push(path);
            }
        }
    }
    files.sort();

    if files.is_empty() {
        println!(
            "未在目录中找到 *_output.json 或 *_glossary.json: {}",
            dir.display()
        );
        return Ok(());
    }

    println!("\n===== 批量术语表质量分析: {} =====", dir.display());
    for f in &files {
        analyze_output_json(f)?;
    }
    Ok(())
}

pub fn analyze_characters_dir(dir: &Path) -> Result<()> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension() != Some(OsStr::new("json")) {
            continue;
        }
        if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
            if name.ends_with("_characters.json") {
                files.push(path);
            }
        }
    }
    files.sort();

    if files.is_empty() {
        println!("未在目录中找到 *_characters.json: {}", dir.display());
        return Ok(());
    }

    println!("\n===== 批量NER候选质量分析: {} =====", dir.display());
    for f in &files {
        analyze_characters_json(f)?;
    }
    Ok(())
}

fn analyze_entries(entries: &[TranslationEntry], label: Option<String>) {
    let total = entries.len();
    let label = label.unwrap_or_else(|| "(in-memory)".to_string());

    let mut dst_kana: Vec<(&str, &str, &str)> = Vec::new();
    let mut dst_space: Vec<(&str, &str, &str)> = Vec::new();
    let mut src_titles: Vec<(&str, &str, &str)> = Vec::new();
    let mut empty_dst: Vec<(&str, &str, &str)> = Vec::new();

    let mut by_src: HashMap<&str, Vec<&TranslationEntry>> = HashMap::new();
    for e in entries {
        by_src.entry(&e.src).or_default().push(e);
    }

    let mut dup_src = Vec::new();
    for (src, list) in &by_src {
        if list.len() > 1 {
            let mut uniq = HashSet::new();
            for x in list {
                uniq.insert((&x.dst, &x.info));
            }
            if uniq.len() > 1 {
                dup_src.push((*src, list.len(), uniq.len()));
            }
        }
    }
    dup_src.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));

    for e in entries {
        if e.dst.trim().is_empty() {
            empty_dst.push((e.src.as_str(), e.dst.as_str(), e.info.as_str()));
        }
        if e.dst.contains(' ') || e.dst.contains('\u{3000}') {
            dst_space.push((e.src.as_str(), e.dst.as_str(), e.info.as_str()));
        }
        if contains_kana(&e.dst) {
            dst_kana.push((e.src.as_str(), e.dst.as_str(), e.info.as_str()));
        }
        if is_pure_title(&e.src) {
            src_titles.push((e.src.as_str(), e.dst.as_str(), e.info.as_str()));
        }
    }

    let mut clusters: BTreeMap<String, Vec<&TranslationEntry>> = BTreeMap::new();
    for e in entries {
        clusters.entry(canonical_key(&e.src)).or_default().push(e);
    }

    let mut cluster_conflicts = Vec::new();
    let mut cluster_variants = Vec::new();
    for (key, list) in &clusters {
        if key.trim().is_empty() {
            continue;
        }
        let mut dsts = HashSet::<&str>::new();
        let mut infos = HashSet::<&str>::new();
        let mut base_dsts = HashSet::<String>::new();
        for x in list {
            dsts.insert(x.dst.as_str());
            infos.insert(x.info.as_str());
            base_dsts.insert(normalize_dst_base(&x.dst));
        }
        if dsts.len() > 1 || infos.len() > 1 {
            if infos.len() <= 1 && base_dsts.len() <= 1 {
                cluster_variants.push((key.as_str(), list.len(), dsts.len(), infos.len()));
                continue;
            }
            cluster_conflicts.push((key.as_str(), list.len(), dsts.len(), infos.len()));
        }
    }
    cluster_conflicts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    cluster_variants.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));

    println!("\n===== 术语表质量报告: {} =====", label);
    println!("总条目数: {}", total);
    println!("dst含假名: {}", dst_kana.len());
    println!("dst含空格: {}", dst_space.len());
    println!("dst为空: {}", empty_dst.len());
    println!("src为纯称谓: {}", src_titles.len());
    println!("src重复且冲突: {}", dup_src.len());
    println!("canonical簇内冲突: {}", cluster_conflicts.len());
    println!(
        "canonical簇内别名变体（同base dst且info一致）: {}",
        cluster_variants.len()
    );

    print_samples("dst含假名", &dst_kana, 12);
    print_samples("dst含空格", &dst_space, 12);
    print_samples("src为纯称谓", &src_titles, 12);
    print_dup_samples(&dup_src, entries, 12);
    print_cluster_conflicts(&cluster_conflicts, &clusters, 12);
    print_cluster_variants(&cluster_variants, &clusters, 12);
}

fn analyze_characters(
    characters: &HashMap<String, crate::detector::CharacterInfo>,
    label: Option<String>,
) {
    let label = label.unwrap_or_else(|| "(in-memory)".to_string());
    let total = characters.len();

    let mut title_like = Vec::new();
    let mut has_kana = Vec::new();
    let mut family_like = Vec::new();
    let mut very_short = Vec::new();

    let mut clusters: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for (name, info) in characters {
        let n = name.as_str();
        if is_pure_title(n) {
            title_like.push((n, info.count));
        }
        if contains_kana(n) {
            has_kana.push((n, info.count));
        }
        if n.ends_with('家') && n.chars().count() >= 2 {
            family_like.push((n, info.count));
        }
        if n.chars().count() <= 1 {
            very_short.push((n, info.count));
        }
        clusters.entry(canonical_key(n)).or_default().push(n);
    }

    title_like.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    has_kana.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    family_like.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    very_short.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));

    let mut multi_alias = Vec::new();
    for (key, names) in &clusters {
        let uniq: HashSet<&str> = names.iter().copied().collect();
        if uniq.len() >= 2 && !key.trim().is_empty() {
            multi_alias.push((key.as_str(), uniq.len()));
        }
    }
    multi_alias.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));

    println!("\n===== NER候选质量报告: {} =====", label);
    println!("候选总数: {}", total);
    println!("纯称谓/职务候选: {}", title_like.len());
    println!("包含假名候选: {}", has_kana.len());
    println!("疑似家庭/家族词（~家）: {}", family_like.len());
    println!("极短候选（<=1字）: {}", very_short.len());
    println!("canonical可归并簇（>=2别名）: {}", multi_alias.len());

    print_name_count_samples("纯称谓/职务候选", &title_like, 12);
    print_name_count_samples("包含假名候选", &has_kana, 12);
    print_name_count_samples("疑似家庭/家族词（~家）", &family_like, 12);
    print_name_count_samples("极短候选（<=1字）", &very_short, 12);

    if !multi_alias.is_empty() {
        println!("\n[canonical可归并簇] Top（最多12条）");
        for (k, n) in multi_alias.iter().take(12) {
            println!("  {}: {}个别名", k, n);
            if let Some(names) = clusters.get(*k) {
                let mut uniq: Vec<&str> = names.iter().copied().collect();
                uniq.sort();
                uniq.dedup();
                let preview = uniq.into_iter().take(8).collect::<Vec<_>>().join(" / ");
                println!("    - {}", preview);
            }
        }
    }
}

fn print_name_count_samples(title: &str, items: &[(&str, usize)], limit: usize) {
    if items.is_empty() {
        return;
    }
    println!("\n[{}] 样例（最多{}条）", title, limit);
    for (name, count) in items.iter().take(limit) {
        println!("  {} ({}次)", name, count);
    }
}

fn print_samples(title: &str, items: &[(&str, &str, &str)], limit: usize) {
    if items.is_empty() {
        return;
    }
    println!("\n[{}] 样例（最多{}条）", title, limit);
    for (src, dst, info) in items.iter().take(limit) {
        println!("  {} -> {}   #{}", src, dst, info);
    }
}

fn print_dup_samples(dup_src: &[(&str, usize, usize)], entries: &[TranslationEntry], limit: usize) {
    if dup_src.is_empty() {
        return;
    }
    let mut by_src: HashMap<&str, Vec<&TranslationEntry>> = HashMap::new();
    for e in entries {
        by_src.entry(&e.src).or_default().push(e);
    }

    println!("\n[src重复且冲突] 样例（最多{}组）", limit);
    for (src, count, uniq) in dup_src.iter().take(limit) {
        println!("  {}: {}条（{}种dst/info）", src, count, uniq);
        let list = &by_src[src];
        let mut seen = HashSet::new();
        for x in list {
            if seen.insert((&x.dst, &x.info)) {
                println!("    - {}   #{}", x.dst, x.info);
            }
        }
    }
}

fn print_cluster_conflicts(
    cluster_conflicts: &[(&str, usize, usize, usize)],
    clusters: &BTreeMap<String, Vec<&TranslationEntry>>,
    limit: usize,
) {
    if cluster_conflicts.is_empty() {
        return;
    }
    println!("\n[canonical簇内冲突] 样例（最多{}组）", limit);
    for (key, count, dstn, infon) in cluster_conflicts.iter().take(limit) {
        println!(
            "  {}: {}条（dst {}种 / info {}种）",
            key, count, dstn, infon
        );
        if let Some(list) = clusters.get(*key) {
            let mut uniq = HashSet::new();
            for x in list {
                if uniq.insert((x.src.as_str(), x.dst.as_str(), x.info.as_str())) {
                    let alias = strip_affixes(&x.src);
                    println!("    - {} ({}) -> {}   #{}", x.src, alias, x.dst, x.info);
                }
            }
        }
    }
}
