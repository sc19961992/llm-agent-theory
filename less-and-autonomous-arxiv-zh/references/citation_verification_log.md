# 引文直接核验记录

核验日期：2026-07-15。

本表只收录当前论文实际需要或附录可能调用的文献。元数据通过出版社页面、DOI 注册元数据、学会论文页、arXiv 原始记录或机构原始档案逐项核对；没有从母稿的关键词清单直接复制。论文中的数学式统一换成本文的行随机、顺序复合约定，未照搬采用列向量约定的公式。

| BibTeX key | 直接核验入口 | 核验内容与本文允许支持的陈述 |
|---|---|---|
| `KemenySnell1976` | [Springer 图书页](https://link.springer.com/book/9780387901923)；书中 §6.3 | 作者、书名、1976 Springer 重印信息、ISBN；给定分区对所有初始分布成立的有限链 lumpability 纤维判据。原版为 1960，已在条目 note 中标出。 |
| `Buchholz1994` | [Cambridge 论文页](https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/exact-and-ordinary-lumpability-in-finite-markov-chains/2DC748F09D80BEEB03CCF18036E149D7)；[DOI](https://doi.org/10.2307/3215235) | 1994 年、卷期 31(1)、页 59–75。Buchholz 的 ordinary lumpability 对应本文使用的出块概率条件；exact lumpability 是另一条件，本文没有混称。 |
| `RogersPitman1981` | [Project Euclid/DOI](https://doi.org/10.1214/aop/1176994363) | 作者、题名、Annals of Probability 9(4), 573–582；link-kernel 条件是 Markov-function 的充分框架，不被写成 strong lumpability 的必要充分条件。 |
| `GurvitsLedoux2005` | [Elsevier/DOI](https://doi.org/10.1016/j.laa.2005.02.007) | Linear Algebra and its Applications 404, 85–117；支持函数后 Markov 性、强/弱 lumpability 与不变空间背景。 |
| `LarsenSkou1991` | [Elsevier 论文页](https://www.sciencedirect.com/science/article/pii/0890540191900306) | Information and Computation 94(1), 1–28；只用于概率互模拟背景，没有把带动作模型当成一般 Markov 链聚合的唯一来源。 |
| `DerisaviEtAl2003` | [Elsevier 论文页](https://www.sciencedirect.com/science/article/pii/S0020019003003430) | Information Processing Letters 87(6), 309–315；支持稳定分区细化与最优 lumping 的算法背景。论文主体写 CTMC，并说明方法可延伸到 DTMC。 |
| `CrutchfieldYoung1989` | [APS 论文页](https://link.aps.org/doi/10.1103/PhysRevLett.63.105) | Physical Review Letters 63(2), 105–108；用于计算力学和预测状态的历史入口。 |
| `ShaliziCrutchfield2001` | [Springer 论文页](https://link.springer.com/article/10.1023/A%3A1010388907793)；[arXiv 原始记录](https://arxiv.org/abs/cond-mat/9907176) | Journal of Statistical Physics 104, 817–879；支持 causal states 的预测充分性、递推与适当意义下的最小性。本文没有省略其平稳性、正则条件分布及几乎处处语境后声称无条件有限状态。 |
| `Fritz2020` | [arXiv 原始记录](https://arxiv.org/abs/1908.07021)；[DOI](https://doi.org/10.1016/j.aim.2020.107239) | Advances in Mathematics 370, 107239；作为 Markov categories、随机 kernels、条件独立与充分性的主引文。未把 Baez 的相关工作误列为这篇 Markov-category 主文献。 |
| `VaswaniEtAl2017` | [NeurIPS 论文页](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)；[arXiv 原始记录](https://arxiv.org/abs/1706.03762) | 作者、标题、会议和页码；只用于 Transformer 架构背景，不用于支持本文预测状态定理。后者由补充材料自行证明。 |
| `Kallenberg2021` | [Springer 图书页](https://link.springer.com/book/10.1007/978-3-030-61871-1) | 第三版、2021、DOI/ISBN；用于标准 Borel 空间、kernel、disintegration 等测度论背景。未据此声称任意可测商都良好。 |
| `EthierKurtz1986` | [Wiley 图书页](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316658) | 1986 首版、作者、书名、ISBN/DOI；用于一般连续时间 Markov 过程、生成元和 martingale problem 的背景。有限矩阵指数等式由本文直接证明。 |
| `Dobrushin1956` | [SIAM/DOI](https://doi.org/10.1137/1101006) | 原始论文题名、期刊、卷期页；用于 Dobrushin 系数的历史归属。 |
| `GaubertQu2015` | [Springer/DOI](https://doi.org/10.1007/s00020-014-2193-2)；[arXiv](https://arxiv.org/abs/1307.4649) | 现代 TV 收缩系数表述、卷页和 DOI；本文的有限收缩不等式另有自足证明。 |
| `Koopman1931` | [PNAS/PMC 原文页](https://pmc.ncbi.nlm.nih.gov/articles/PMC1076052/) | PNAS 17(5), 315–318，DOI；只用于 Koopman 观测算子的历史背景。有限点乘代数与分区的等价由本文证明，不归给该文。 |

## 术语与原创性边界

- 本文条件是 Kemeny–Snell 的 lumpability，现代常称 strong lumpability，在 Buchholz 的术语中对应 ordinary lumpability。正文不再把 “strong / ordinary” 写成两种并列条件。
- Rogers–Pitman 只被描述为带 link kernel、对相容初始分布的充分框架。
- causal states 被描述为最粗的预测充分划分；任何其他预测充分统计量须细化它。它不保证状态空间有限。
- 给定分区的精确纤维判据、稳定分区细化、Dobrushin 收缩和 causal-state 最小性不作为本文原创定理声称。
- 本文可辨认的新贡献集中在操作性微观基线、严格填充不变性组织、任务保真系统族指标 `Γ_N`、该指标的充要刻画、统一故障—修复正例、近似证书的组合，以及不同证书逻辑的严格拆分。

## 未纳入参考文献表的母稿关键词

母稿还提到 Dynkin、Blackwell、Le Cam、Baez 等。它们与本文有历史或类比关系，但当前正文没有依赖相应具体结论，故没有为了显得文献丰富而把它们堆入参考文献表。若未来新增统计实验比较或原始 Dynkin 判据的实质讨论，应先写清参数族、决策问题、初始分布量词和公式约定，再加入逐项核验后的引用。
