# 引文直接核验记录

核验日期：2026-07-16。

本表覆盖 `references.bib` 中的全部条目。每项均对照出版社、学会、官方会议论文页或原始全文核验书目信息，并记录该来源在本文中实际支持的陈述范围。本文使用行随机、从左到右复合的记号；涉及的公式均按这一约定在文中重新陈述或证明。

| BibTeX key | 直接核验入口 | 与本文相关的原文内容及支持边界 |
|---|---|---|
| `KemenySnell1976` | [Springer 图书页](https://link.springer.com/book/9780387901923)，§6.3 | 核对作者、书名、1976 Springer 版和 ISBN。§6.3 支持有限链对全部初始分布成立的 lumpability 出块概率判据；原版 1960 年信息已在 BibTeX note 中标明。 |
| `Buchholz1994` | [Cambridge 论文页](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/exact-and-ordinary-lumpability-in-finite-markov-chains/2DC748F09D80BEEB03CCF18036E149D7)，[DOI](https://doi.org/10.2307/3215235) | 核对 *Journal of Applied Probability* 31(1), 59–75。其 ordinary lumpability 对应本文的出块概率条件；论文另行定义的 exact lumpability 是不同概念。 |
| `RogersPitman1981` | [Project Euclid 原文](https://doi.org/10.1214/aop/1176994363) | 核对 *Annals of Probability* 9(4), 573–582。支持带 link kernel 与相容初始分布的 Markov-function 充分框架；本文未将其表述为 strong lumpability 的必要充分条件。 |
| `GurvitsLedoux2005` | [Elsevier 原文](https://doi.org/10.1016/j.laa.2005.02.007) | 核对 *Linear Algebra and its Applications* 404, 85–117。支持有限链函数后 Markov 性、lumpability 与线性不变结构的背景。 |
| `LarsenSkou1991` | [Elsevier 原文](https://www.sciencedirect.com/science/article/pii/0890540191900306) | 核对 *Information and Computation* 94(1), 1–28。支持带标签概率系统中的 testing 与 probabilistic bisimulation；本文仅把它作为行为等价的邻近工作。 |
| `DerisaviEtAl2003` | [Elsevier 原文](https://www.sciencedirect.com/science/article/pii/S0020019003003430) | 核对 *Information Processing Letters* 87(6), 309–315。原文支持 ordinary-lumpable 分区的最粗解与稳定分区细化算法；主体为 CTMC，并说明方法可扩展到 DTMC。 |
| `CrutchfieldYoung1989` | [APS 原文](https://link.aps.org/doi/10.1103/PhysRevLett.63.105) | 核对 *Physical Review Letters* 63(2), 105–108。支持计算力学、预测等价状态与统计复杂度的历史入口。 |
| `ShaliziCrutchfield2001` | [Springer 论文页](https://link.springer.com/article/10.1023/A%3A1010388907793)，[arXiv 原文](https://arxiv.org/abs/cond-mat/9907176) | 核对 *Journal of Statistical Physics* 104, 817–879。支持 causal states 的预测充分性、最小性和唯一性；结论处于平稳过程、正则条件分布和几乎处处语境，且不保证状态空间有限。 |
| `Fritz2020` | [Advances in Mathematics/DOI](https://doi.org/10.1016/j.aim.2020.107239)，[arXiv 原文](https://arxiv.org/abs/1908.07021) | 核对 *Advances in Mathematics* 370, 107239。支持 Markov category、kernel、条件独立与充分统计的统一语言；不承担本文压缩前沿或 $\Gamma_N$ 结论。 |
| `VaswaniEtAl2017` | [NeurIPS 官方论文页](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)，[arXiv 原文](https://arxiv.org/abs/1706.03762) | 核对作者、标题、会议和 5998–6008 页。只支持 Transformer 架构与 decoder 自回归背景；附录 J 的递推表示和闭合定理由本文自行证明。 |
| `Kallenberg2021` | [Springer 图书页](https://link.springer.com/book/10.1007/978-3-030-61871-1) | 核对第三版、2021 年、DOI 与 ISBN。支持标准 Borel kernel、正则条件概率、disintegration 及 Borel–Cantelli 等概率论背景；Borel 商的 smoothness 另引 Kechris。 |
| `EthierKurtz1986` | [Wiley 图书页](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316658) | 核对作者、1986 年首版、ISBN 与 DOI。支持一般连续时间 Markov 过程、生成元定义域、强连续性和 martingale problem 背景；有限矩阵 Duhamel 界由本文直接证明。 |
| `Dobrushin1956` | [SIAM/DOI 原文](https://doi.org/10.1137/1101006) | 核对原始论文题名、期刊、卷期页。本文的最大行 TV 量对应原文收缩范数 $N(P)$；原文以互补量 $1-N(P)$ 表示 ergodic coefficient，正文已明确现代记号差异。 |
| `GaubertQu2015` | [Springer/DOI](https://doi.org/10.1007/s00020-014-2193-2)，[arXiv 原文](https://arxiv.org/abs/1307.4649) | 核对 *Integral Equations and Operator Theory* 81(1), 127–150。直接支持最大行 TV 量的现代收缩系数表述；本文另给有限情形的自足证明。 |
| `Koopman1931` | [PNAS/PMC 原文](https://pmc.ncbi.nlm.nih.gov/articles/PMC1076052/) | 核对 *PNAS* 17(5), 315–318。原文讨论确定性 Hamiltonian 系统在函数空间中的线性演化；本文把有限随机核的观测算子作为直接线性扩展并自行证明其性质。 |
| `MichelSiegle2025` | [Elsevier 原文](https://www.sciencedirect.com/science/article/pii/S0166531624000695) | 核对 *Performance Evaluation* 167 (2025), article 102464。支持 DTMC/CTMC 聚合中的瞬态误差传播、稳态残差及 lumpability 比较；不承担本文任务相对前沿或 $\Gamma_N$ 判据。 |
| `GeigerEtAl2015` | [arXiv 原文及期刊信息](https://arxiv.org/abs/1304.6603)，[DOI](https://doi.org/10.1109/TAC.2014.2364971) | 核对 *IEEE Transactions on Automatic Control* 60(4), 1010–1022。直接研究固定较小状态数下的 Markov 聚合和 KL divergence rate 目标；其平稳 KL 度量不同于本文的任务保真、最坏行 TV 度量。 |
| `BanischLima2015` | [arXiv 原文](https://arxiv.org/abs/1209.3902)，[DOI](https://doi.org/10.1142/S0219525915500113) | 核对 *Advances in Complex Systems* 18(03n04), article 1550011。支持对称网络上的 voter/agent-based Markov 模型以 automorphism 轨道聚合后保持 Markov 性；正文的一般群作用上界另有自足证明。 |
| `LittmanSuttonSingh2001` | [NeurIPS 官方原文](https://proceedings.neurips.cc/paper_files/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf)，[官方论文页](https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html) | 核对 NIPS 14, 1555–1561。论文首页列有 Littman、Sutton、Singh 三位作者；官方 HTML 元数据遗漏 Singh。原文支持用未来的动作条件预测表示状态及其递推思想。 |
| `CoverThomas2006` | [Wiley 图书页](https://onlinelibrary.wiley.com/doi/book/10.1002/047174882X) | 核对第二版版权年 2006、DOI 与 ISBN；平台的 2005 日期是在线记录时间。Lemma 11.6.1 支持自然对数和半 $\ell^1$ TV 约定下的 Pinsker 界；Jensen 步骤在本文中直接给出。 |
| `Kechris1995` | [Springer 图书页](https://link.springer.com/book/10.1007/978-1-4612-4190-4)，§18 | 核对 *Graduate Texts in Mathematics* 156、1995 年、DOI 与 ISBN。§18 的 smooth Borel equivalence relation 理论支持“一般 Borel 等价关系未必有由标准 Borel 值域实现的分类映射”这一边界。 |

## 引用范围与术语约定

- 本文使用的有限链条件是 Kemeny–Snell 的 lumpability，现代常称 strong lumpability；在 Buchholz 的术语中对应 ordinary lumpability。Buchholz 的 exact lumpability 是另一条件。
- 单链最粗 ordinary lumping、稳定分区细化、Dobrushin 收缩、causal-state 最小性和固定预算 Markov 聚合均已有直接文献坐标。本文在正文中明确标出这些既有结果的适用范围。
- 本文的复杂度--闭合误差前沿采用任务保真约束和最坏行 TV 缺陷；Geiger 等采用平稳 KL divergence rate。两者属于相关优化结构，目标函数与量词不同。
- 附录 J 的定理适用于有限自回归生成器，Transformer 是目标应用之一，具体模型仍需独立验证 T1--T3。Vaswani 等只承担架构背景；KL 到加权 TV 的传递以及 $\Gamma_N$ 上界均以本文列出的假设和证明为准。
- 标准 Borel kernel 的背景由 Kallenberg 支持；一般 Borel 等价关系的 smoothness 边界由 Kechris 支持。附录 H 只证明固定良好因子的接口结论。
- 本文的主要数学对象包括可辨微观基线、任务保真的共同精确商、复杂度--闭合误差前沿、纤维预测直径以及系统族指标 $\Gamma_N$。所有定理的证明均在正文或附录中给出，文献用于定位既有定义、结果与邻近研究。
