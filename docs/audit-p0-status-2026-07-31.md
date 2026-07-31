# 审计 P0/P1 落地状态（2026-07-31）

对照 `dist/audit` 基线结论，在当前 `main` 上的修复状态。

| 编号 | 问题 | 状态 | 相关 commit |
|------|------|------|-------------|
| P0-01 | EPUB HTML5 归一化写回未改章节 | **已修** | `f0c4470` 仅写 modified + void 还原 XHTML |
| P0-02 | Replace 抹平 inner HTML | **已修** | 同上，保留 ruby/em/a 等内联结构 |
| P0-03 | Orion 空 dst 术语被过滤 | **已修** | `c4d3abe` 人物候选注入 Orion prompt |
| P0-04 | 断点恢复整批重发 | **已修** | `44feccf` 只提交 pending indices |
| P0-05 | join/API 失败索引丢失 | **已修** | 同上，任务返回 pending 并 mark 失败 |
| P1-05 | AutoFixer 副作用 | **已修** | `44e5cb7` 通过项 `fix_safe` + 复检 |
| P1-08 | ner_batch_size/llm_workers=0 | **已修** | `65a6f60` `GlossaryConfig::validate` |

未在本轮处理的仍见审计文档：P1-01 格式修复默认开、P1-02 TXT 空白、P1-03 质检硬约束扩展、P1-06 共享 Client、P1-07 token 批、P1-09/10 等。

本地 `dist/audit` 下的 markdown 可手改状态表；该目录被 `.gitignore` 忽略，不进入仓库。
