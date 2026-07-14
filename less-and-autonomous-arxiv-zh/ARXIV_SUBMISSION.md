# arXiv 编译与投稿说明

## 当前状态

本目录已经按多文件 XeLaTeX 项目组织，入口为根目录的 `main.tex`。中文版正文、附录、补充材料和 BibTeX 数据均随包提供。

## 语言政策

arXiv 的“Non-English submissions”页面目前说明：非英文投稿可以提交，但必须同时提供论文的完整英文版本；多语言稿应先排英文版、再排非英文版。因此，本目录可以作为中文版母稿和中文版本源文件，不能在缺少完整英文版时被标记为“已满足直接投稿条件”。英文标题与英文摘要不能替代完整英文正文。

核验入口：

- https://info.arxiv.org/help/faq/multilang.html
- https://info.arxiv.org/help/submit_tex.html
- https://info.arxiv.org/help/faq/texlive.html

## 上传前检查

1. 增加完整英文版，并使英文版位于中文版本之前。
2. 从项目根目录编译 `main.tex`，逐页检查最终 PDF。
3. 保留全部被 `\input` 或 `\bibliography` 引用的源文件。
4. 上传 `references/references.bib`；若自动 BibTeX 处理失败，可同时提供本地生成的 `main.bbl`。
5. 删除编译中间文件和未引用附件；不要上传本地生成的 PDF 作为 TeX 源稿的一部分。
6. 投稿元数据使用英文标题、英文摘要和拉丁字母作者名，并在 Comments 中注明包含中文版本。
7. 不使用 `\today`；版本日期由投稿记录给出。

## 编译器选择

项目使用 `ctexart` 与 XeLaTeX。arXiv 当前 TeX Live 页面列出 XeLaTeX 为支持的处理器。源码不按字体名称调用本机字体，也不依赖 `cleveref`，以避免服务器字体查询和 TeX Live 版本差异。
