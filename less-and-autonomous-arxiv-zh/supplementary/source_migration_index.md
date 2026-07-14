# 母稿迁移索引

本文档记录两份内部母稿如何迁移到《少而自治》论文包。它是编辑追踪文件，不是论文参考文献，也不主张母稿中的每一句话都进入正式稿。正式论文的定义、定理编号和证明以本目录中的 LaTeX 文件为准。

## 迁移原则

- 正文只保留问题、定义、主判据、非平凡正例、构造路线和结论边界；可独立证明移入附录。
- 同一数学内容若在两份母稿重复，以“统一证明框架”为证明底稿，以“核心结论直观版”为解释与 Transformer 构造底稿；正式稿只保留一份定理陈述。
- “涌现”“规律”“自治”均改成可检验的操作性命题，不保留无法由定理推出的本体论或普遍性表述。
- Transformer 结果采用分布加权 TV；它不被写成正文最坏情形 $\Gamma_N$ 判据的字面特例。两者的对齐条件单列在 `S1_transformer_predictive_states.tex`。
- 标准 Borel 空间、随机粗粒化和连续时间不进入有限主定理，只放在讨论与推广附录，并明确额外条件。

## 母稿一：`涌现构建闭合-统一证明框架.md`

| 母稿单元 | 新位置 | 处理 |
|---|---|---|
| 标题“涌现、构建与闭合：一个统一证明框架” | `main.tex`、`README.md` | 改名为“少而自治”，把范围收窄到有限随机系统与任务相对判据；原题不保留。 |
| 阅读导引：问题、非平凡内容与证明结构 | `sections/01_introduction.tex` | 压缩为研究问题、贡献与路线，不在正文预告全部技术细节。 |
| 宏观与微观先由信息关系区分 | `sections/02_operational_microstates.tex`；`appendices/appendix_B_operational_baseline.tex` | 正文给操作性定义，规范构造与填充不变性证明移入附录。 |
| 贯穿全文的非平凡正例 | `sections/05_fault_repair_example.tex`；`appendices/appendix_G_examples_counterexamples.tex` | 正文保留故障数链及最小性结论；辅助计算与对照例移入附录。 |
| 可检验的系统族充要条件 | `sections/04_main_criterion.tex`；`appendices/appendix_D_approximate_theory.tex` | 作为正文中心定理；优化达到、等价方向和技术引理分置正文短证与附录全证。 |
| Transformer 构造例与一般框架的关系 | `sections/06_construction_routes.tex`；`supplementary/S1_transformer_predictive_states.tex` | 正文只说明结论和非字面特例关系；三个假设、加权定理与完整证明进入补充材料。 |
| 残余对称证书与相形成被严格拆开 | `sections/06_construction_routes.tex`；`appendices/appendix_F_structural_certificates.tex` | 正文保留“证书”和“形成机制”不同；概率事件与证明放附录。 |
| 单个有限系统能够严格保证什么 | `sections/03_closure_and_fibers.tex`、`sections/04_main_criterion.tex`；附录 C、D | 去除与后文重复的清单；精确结论归附录 C，近似优化归附录 D。 |
| 时间尺度的位置 | `sections/07_verification_and_limits.tex`、`sections/08_discussion.tex`；`appendices/appendix_E_composition_time.tex` | 一步主定理与多步外推分开；固定时长、增长时长和收缩条件不混写。 |
| 证明依赖顺序 | `sections/01_introduction.tex`；`main.tex` 的文件次序 | 改成一段路线说明；不重复列出所有引理。 |
| 0. 文献坐标 | `sections/01_introduction.tex`；`references/references.bib`；`references/citation_verification_log.md` | 只保留实际用到且已核验的学术坐标；母稿关键词清单不直接转成参考文献。 |
| 1. 有限 Markov 记号 | `sections/02_operational_microstates.tex`；`appendices/appendix_A_markov_tv.tex` | 正文只留必要记号；TV、通道复合与收缩事实移到附录 A。 |
| 1.1 操作性微观呈现 | `sections/02_operational_microstates.tex`；`appendices/appendix_B_operational_baseline.tex` | 保留探针、允许动力学与规范微观核；完整等价证明放附录。 |
| 1.2 可交换故障—修复链 | `sections/05_fault_repair_example.tex`；`appendices/appendix_G_examples_counterexamples.tex` | 作为纵贯正文的非平凡正例；参数退化与最小性细节放附录。 |
| 2. 无范畴核心版 | `sections/03_closure_and_fibers.tex`、`sections/04_main_criterion.tex`；附录 C、D | 拆成“闭合缺陷”和“系统族判据”两段，避免范畴语言遮蔽主线。 |
| 2.1 闭合粗粒化判据 | `sections/03_closure_and_fibers.tex`；`appendices/appendix_C_exact_theory.tex` | 正文给纤维判据；双向证明与精确 lumpability 关系放附录 C。 |
| 2.2 观测代数生成粗粒化 | `appendices/appendix_C_exact_theory.tex` | 整体移出正文，作为最小闭合商的构造工具。 |
| 2.3 闭合代数与自治商 | `appendices/appendix_C_exact_theory.tex` | 与 2.2 合并陈述，删去重复定义。 |
| 2.4 任务相对的最小闭合商 | `sections/04_main_criterion.tex`；`appendices/appendix_C_exact_theory.tex` | 正文只用其规范性结论；精确构造和唯一性证明入附录。 |
| 2.5 多层复合与有限终止 | `appendices/appendix_E_composition_time.tex` | 全部移出正文；与近似复合统一记号。 |
| 3. 非平凡压缩的来源机制 | `sections/06_construction_routes.tex` | 压缩为对称、守恒和预测状态三条候选构造路线。 |
| 3.1 对称性保证严格压缩 | `sections/06_construction_routes.tex`；`appendices/appendix_F_structural_certificates.tex` | 正文给充分证书；群作用、近似等变和任务不变条件入附录。 |
| 3.1.1 对称破缺的严格角色与有限系统障碍 | `appendices/appendix_F_structural_certificates.tex`；`sections/08_discussion.tex` | 只作为候选分区形成机制；不写成闭合的替代证明。 |
| 3.2 守恒任务达到最小可能复杂度 | `sections/06_construction_routes.tex`；`appendices/appendix_F_structural_certificates.tex` | 正文留结论，最小性证明入附录；与故障数正例去重。 |
| 3.3 预测状态与 Koopman 表示的边界 | `sections/06_construction_routes.tex`、`sections/08_discussion.tex`；`supplementary/S1_transformer_predictive_states.tex` | 预测状态形成 Transformer 补充材料；Koopman 只用于讨论概念边界，不宣称等价。 |
| 4. 范畴化版本 | `appendices/appendix_I_categories.tex` | 完整移出正文；作为组织语言，不增加有限主定理的数学强度。 |
| 5. 近似闭合、系统族与扩展 | `sections/03_closure_and_fibers.tex`、`sections/04_main_criterion.tex`；附录 D、E、H | 有限近似理论留主线，空间/时间推广移附录。 |
| 5.1 有限状态到标准 Borel 空间 | `sections/08_discussion.tex`；`appendices/appendix_H_extensions.tex` | 只给带正则条件的推广框架；不写成有限证明的自动延伸。 |
| 5.2 随机 coarse-graining kernel | `sections/08_discussion.tex`；`appendices/appendix_H_extensions.tex` | 作为另一模型类；确定性分区主定理不由它替代。 |
| 5.3 有限 Markov 近似闭合理论 | `sections/03_closure_and_fibers.tex`；`appendices/appendix_D_approximate_theory.tex` | 正文保留闭合缺陷与纤维直径，完整优化理论入附录。 |
| 5.3.1 TV 距离与收缩引理 | `appendices/appendix_A_markov_tv.tex` | 集中为预备事实，正文只引用。 |
| 5.3.2 近似动力学因子 | `sections/03_closure_and_fibers.tex`；`appendices/appendix_D_approximate_theory.tex` | 正文给定义，等价表达与稳定性入附录。 |
| 5.3.3 最优宏观 kernel 存在 | `sections/03_closure_and_fibers.tex`；`appendices/appendix_D_approximate_theory.tex` | 正文只陈述有限维紧致性结论；完整证明入附录。 |
| 5.3.4 确定性粗粒化的纤维中心公式 | `sections/03_closure_and_fibers.tex`；`appendices/appendix_D_approximate_theory.tex` | 纤维直径进入可检验指标；中心半径公式与常数关系入附录。 |
| 5.3.5 两层复合与误差证书范畴 | `appendices/appendix_E_composition_time.tex`；`appendices/appendix_I_categories.tex` | 数值误差复合与范畴解释分开，删除双份证明。 |
| 5.3.6 多层空间复合 | `appendices/appendix_E_composition_time.tex` | 移出正文，保留有限层误差和。 |
| 5.3.7 同一因子的多步时间误差 | `appendices/appendix_E_composition_time.tex`；`sections/07_verification_and_limits.tex` | 附录给线性与收缩界，正文只强调适用条件。 |
| 5.3.8 确定性粗粒化的路径误差 | `appendices/appendix_E_composition_time.tex` | 与状态边缘误差明确区分；不把状态收缩误用到已输出轨迹。 |
| 5.3.9 任务相对复杂度—误差曲线 | `sections/04_main_criterion.tex`；`appendices/appendix_D_approximate_theory.tex` | 正文用其定义 $\Gamma_N$，更多 Pareto/容限形式入附录。 |
| 5.4 系统族的低复杂度近似闭合 | `sections/04_main_criterion.tex`；`appendices/appendix_D_approximate_theory.tex` | 作为充要条件主定理；量词与优化达到完整写明。 |
| 5.4.1 Transformer 分布加权构造例 | `sections/06_construction_routes.tex`；`supplementary/S1_transformer_predictive_states.tex` | 从一般证明中拆出，自包含重证；新增与 $\Gamma_N$ 对齐的三项条件。 |
| 5.4.2 残余对称充分证书 | `sections/06_construction_routes.tex`；`appendices/appendix_F_structural_certificates.tex` | 正文保留证书形式；内缺陷、外分离和压缩条件入附录。 |
| 5.4.3 近似残余相与闭合证书 | `appendices/appendix_F_structural_certificates.tex` | 事件逻辑全部放附录，保持“相形成”和“闭合”两条逻辑独立。 |
| 5.5 估计动力学下的证书稳健性 | `sections/07_verification_and_limits.tex`；`appendices/appendix_D_approximate_theory.tex` | 正文给可检验含义，固定基线下的扰动常数与限制入附录。 |
| 5.6 单步 Markov 到连续时间半群 | `sections/08_discussion.tex`；`appendices/appendix_H_extensions.tex` | 仅作带生成元/半群兼容条件的推广，不进入主定理。 |
| 5.7 点乘代数到 $\sigma$-代数 | `sections/08_discussion.tex`；`appendices/appendix_H_extensions.tex` | 与标准 Borel 推广合并，删除重复动机。 |
| 6. 结论层级与严格边界 | `sections/07_verification_and_limits.tex`、`sections/08_discussion.tex` | 改写为“已证、需验证、未主张”三层。 |
| 6.1 已证明的结论 | `sections/07_verification_and_limits.tex` | 与正文定理去重，只保留审计式摘要。 |
| 6.2 本文不主张的结论 | `sections/07_verification_and_limits.tex`、`sections/08_discussion.tex` | 保留并收紧；状态数不等于描述长度、计算代价、因果自治或低维几何。 |
| 6.3 各机制在统一框架中的位置 | `sections/06_construction_routes.tex` | 改为候选构造路线，不把充分机制写成必要条件。 |
| 6.4 完整逻辑链 | `sections/01_introduction.tex`、`sections/08_discussion.tex` | 前置为简洁路线，结尾不重复全部公式。 |

## 母稿二：`涌现构建闭合-核心结论直观版.md`

| 母稿单元 | 新位置 | 处理 |
|---|---|---|
| 标题“从底层动力学到上层规律……” | `main.tex`、`README.md` | 不沿用；标题改为有限、任务相对且不过度声称的“少而自治”。 |
| 认识论立场 | `sections/01_introduction.tex` | 只保留操作主义方法说明；删除“世界上的规律都……”一类不可由本文验证的普遍断言。 |
| 核心数学结论 | `main.tex` 摘要；`sections/01_introduction.tex`、`sections/04_main_criterion.tex` | 由正式 $\Gamma_N$ 定理替代，避免把条件性存在写成无条件涌现。 |
| 四种表述 | `main.tex` 摘要；`sections/01_introduction.tex` | 合并为一条正式摘要和一段直观解释，不在论文中重复四遍同一结论。 |
| 1. 一般问题 | `sections/01_introduction.tex` | 作为问题动机，删去与定义章节重复的公式。 |
| 1.1 任务相对复杂度 | `sections/02_operational_microstates.tex`、`sections/04_main_criterion.tex` | 用操作性微观核和相对状态数正式化；强调基线必须预先声明。 |
| 2. 单个有限系统 | `sections/03_closure_and_fibers.tex`；附录 C、D | 分成精确与近似结果；“最优存在”不写成“容易找到”。 |
| 3. 规模增长的系统族 | `sections/04_main_criterion.tex` | 由 $\Gamma_N\to0$ 的充要条件替代口语化定义。 |
| 4. 低复杂度构建来源 | `sections/06_construction_routes.tex` | 仅作为候选来源，不宣称总能成功。 |
| 4.1 任务本身守恒 | `sections/06_construction_routes.tex`；`appendices/appendix_F_structural_certificates.tex` | 与统一母稿 3.2 去重。 |
| 4.2 残余对称商 | `sections/06_construction_routes.tex`；`appendices/appendix_F_structural_certificates.tex` | 与统一母稿 3.1、5.4.2 去重。 |
| 4.3 预测状态 | `sections/06_construction_routes.tex`；`supplementary/S1_transformer_predictive_states.tex` | 正文一段，详细构造入补充材料。 |
| 5. 预测状态为何闭合 | `supplementary/S1_transformer_predictive_states.tex` | 重写为“递推 + 同一 token 推前”的严格证明。 |
| 6. 三个条件 | `supplementary/S1_transformer_predictive_states.tex` | 统一为 T1 语言结构、T2 总体学习、T3 状态可读出。 |
| 6.1 统一记号与系统边界 | `supplementary/S1_transformer_predictive_states.tex` 的“系统边界与统一记号” | 保留有限词表、有限时域、EOS、分布与 Markov 完整状态；删去重复说明。 |
| 6.2 低复杂度递推预测状态 | 同上 T1 | 保留读出误差、精确递推与相对状态数；基线改为正文操作性基线。 |
| 6.3 学会总体下一 token 规律 | 同上 T2 | 保留总体 KL 与 teacher-forcing/rollout 边界；经验测试成功不直接替代。 |
| 6.4 预测状态由完整状态确定 | 同上 T3 | 保留“完整历史可构造、指定神经层需验证”的强度区分。 |
| 7. Transformer 自回归预测状态定理 | `supplementary/S1_transformer_predictive_states.tex` | 改名为分布加权闭合定理，避免单独使用“涌现”造成过度解释。 |
| 7.1 距离与宏观转移 | 同上“系统边界”“加权 TV 定理” | 统一 $d_{\nu_N}$、$Q_q$、$K_N$、$R_N$、$F_N$、$L_N$ 记号。 |
| 7.2 定理 | 同上定理 `transformer-weighted-closure` | 保留 $\eta_N+\sqrt{\delta_N/2}$ 两个误差界和复杂度界。 |
| 7.3 证明 | 同上完整证明 | 保留逐历史 Pinsker、Jensen、三角不等式和确定性推前收缩四步，压缩重复文字。 |
| 7.4 精确情形与近似递推 | 同上 remark 与 proposition `transformer-approx-recursion` | 保留全支撑限定和 $\zeta_N$；causal-state 长篇动机不重复进入正式证明。 |
| 8. 从一步预测到整段生成 | `supplementary/S1_transformer_predictive_states.tex`；`appendices/appendix_E_composition_time.tex` | Transformer 特定固定时长边缘界入 S1；一般多步与路径界入附录 E。 |
| 8.1 上层模拟器如何运行 | `sections/06_construction_routes.tex`；S1 的 $R_N,F_N,L_N$ 定义 | 不保留操作步骤列表；由 kernel 定义直接表达。 |
| 8.2 宏观状态边缘多步误差 | S1 proposition `transformer-fixed-horizon`；`appendices/appendix_E_composition_time.tex` | 补足 rollout 分布前提；状态边缘与 token 轨迹明确分开。 |
| 9. 最简可行规律存在 | `sections/04_main_criterion.tex`；`appendices/appendix_D_approximate_theory.tex` | 由有限分区优化达到与 $\Gamma_N$ 判据统一处理，不在 S1 重证。 |
| 10. 对应真实 Transformer | `sections/07_verification_and_limits.tex`；`supplementary/S1_transformer_predictive_states.tex` | 分成形式边界、部署核验和定理边界，不作为无条件现实结论。 |
| 10.1 固定推理过程有限 Markov 化 | S1“系统边界与统一记号”；`sections/07_verification_and_limits.tex` | 保留有限精度、有限历史与完整状态条件；连续/无界环境列为范围外。 |
| 10.2 最终输出任务通道 | `sections/07_verification_and_limits.tex`；`appendices/appendix_E_composition_time.tex` | 归入一般任务通道与后处理，不与下一 token 定理混为同一任务。 |
| 10.3 代理误差相加 | `sections/07_verification_and_limits.tex`；`appendices/appendix_D_approximate_theory.tex` | 归入核估计/部署失配的三角不等式；避免 S1 主线膨胀。 |
| 11. 结论层级与边界 | `sections/07_verification_and_limits.tex`；S1“结论边界” | 去重后分别保留一般理论与 Transformer 专项边界。 |
| 11.1 一般有限系统 | `sections/07_verification_and_limits.tex` | 以正式稿定理编号重写，不复制旧表。 |
| 11.2 Transformer 自回归情形 | S1“结论边界”；`sections/07_verification_and_limits.tex` | 区分已证的条件结论和仍需验证的 T1--T3。 |
| 11.3 行为规律、内部表征与低维几何 | S1 T3 与“结论边界”；`sections/08_discussion.tex` | 保留三层区分：行为商不推出指定层读出，更不推出低维流形。 |
| 结语 | `sections/08_discussion.tex`；S1 开头与结论边界 | 不照录传播性口号；由条件、结论和未证内容共同收束。 |

## 明确删除或降级的内容

1. 删除任何“有限复杂系统必然有简单宏观规律”的无条件句式；正文只证明 $\Gamma_N\to0$ 与指定低复杂度闭合性质等价，并给出满足条件的系统族。
2. “最简”仅指预先声明候选类与复杂度下最小值达到，不等于最短程序、最易解释、最易计算或可被高效搜索。
3. “自治”仅指声明 kernel、任务和误差范数下的转移闭合，不自动等于因果干预自治。
4. Transformer 的行为级预测状态不自动成为指定神经层中的显式表征；更不自动推出 hidden-state 低维流形。
5. teacher-forcing 分布上的平均误差不外推到自由生成的最坏历史；固定时长外推需要逐时刻 rollout 证书，增长时长需要误差和或收缩条件。
6. 标准 Borel、随机粗粒化、连续时间和范畴化只提供推广或组织方式，不提升有限主定理的适用范围。
