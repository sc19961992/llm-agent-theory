# 《宏观规律何时存在？》中文版论文包

本目录是论文《宏观规律何时存在？有限随机系统中的压缩—闭合判据》的中文正式稿。正文保留从操作性基线、有限规模精确商与近似前沿到系统族判据的论证主干；完整证明、反例、误差传播、结构证书、扩展接口、范畴化组织和 Transformer 专用证明均置于附录。

## 目录

- `main.tex`：论文入口，使用 XeLaTeX。
- `macros.tex`：统一记号与定理环境。
- `sections/`：正文。
- `appendices/`：数学附录。
- `supplementary/`：附录 J 的独立编译入口与母稿迁移索引。Transformer 证明的唯一源码位于 `appendices/appendix_J_transformer_predictive_states.tex`；独立入口只包装同一文件，不维护重复副本。
- `references/references.bib`：正文实际引用的 BibTeX 条目。
- `references/citation_verification_log.md`：逐项引文核验记录。
- `ARXIV_SUBMISSION.md`：编译、打包和当前 arXiv 多语言政策说明。

## 编译

推荐在 TeX Live 2025 环境中运行：

```bash
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex
```

项目不依赖本机专有字体；中文排版由 `ctex` 的 TeX Live 字体配置处理。不要把 `.aux`、`.log`、`.out`、`.toc` 等中间文件放入 arXiv 上传包。

Transformer 附录 J 也可作为独立版本编译：

```bash
cd supplementary
xelatex transformer_supplement.tex
xelatex transformer_supplement.tex
```

## 版本边界

本文严格证明的是有限状态、一步、任务相对、最坏状态误差下的结果。标准 Borel 空间、随机粗粒化和连续时间只在讨论与附录中给出需要额外条件的推广；没有把这些推广伪装成有限主定理的无条件推论。

本目录是中文版源稿。根据 2026 年 7 月可查的 arXiv 多语言投稿政策，非英文稿件还须随附完整英文版，并在同一稿件中把英文版置于前面。完整英文翻译不在本次“中文版”交付范围内；详见 `ARXIV_SUBMISSION.md`。
