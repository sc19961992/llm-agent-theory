# 宏观规律的生成、维持与有向演化

## 有限微观状态系统中的闭合、扰动恢复与移动平衡

> 研究草稿 v0.2

---

## 摘要

设微观状态为 $X_t$，粗粒化后的宏观状态为

$$
Y_t=q(X_t).
$$

固定环境和系统参数时，微观转移矩阵 $K_\theta$ 若满足

$$
K_\theta Q_q\simeq Q_qL_\theta,
$$

便生成宏观状态的运动方程 $L_\theta$。这给出宏观规律的第一部分：微观动力推动宏观状态运动。

环境中的宏观规律还会经历两种运动。第一种是扰动恢复：在同一环境下制备一组未扰动系统和一组轻微扰动系统，两组规律参数的平均差随时间衰减。第二种是有向演化：未扰动系统的规律参数沿一个预先选定、具有物理意义的方向量持续变化。扰动恢复描述规律对偏离的响应；有向演化描述规律参考轨迹自身的移动。

本文建立这三种运动的统一形式。当规律参数的更新由已保留的宏观变量决定时，联合闭合误差与原有闭合误差相等。未扰动轨迹确定参考路径；配对扰动实验进一步确定路径附近的响应矩阵。当局部响应可逐步复合且响应矩阵一致收缩时，微小扰动的平均响应便衰减。若方向量在宏观规律适用域内具有统一的正增量下界，并且适用域沿该方向具有有限边界，则规律在有限期望时间内到达边界。

故障—修复系统给出完整例子。固定损伤和修复能力时，部件的微观跳变精确生成故障数的宏观生灭方程；修复反馈使系统回到稳定平衡；累积损伤推动平衡点持续移动；有限修复能力使目标故障率在临界损伤后失去可维持性。这个模型把宏观规律的生成、维持、有向演化和条件失效放入同一组方程。

**关键词：** 宏观闭合；粗粒化；扰动恢复；移动平衡；有向演化；任务失效

---

## 1. 宏观规律也会运动

### 1.1 三种运动

一条宏观规律至少涉及两个层次。

第一层是宏观状态的运动。微观状态 $X_t$ 按 $K_\theta$ 演化，宏观状态 $Y_t=q(X_t)$ 按 $L_\theta$ 演化：

$$
K_\theta Q_q\simeq Q_qL_\theta.
$$

第二层是规律参数 $\theta_t$ 的运动。它包含两个可以分别测量的部分。

给定未扰动参考轨迹 $\theta_t^0$，在时刻 $t$ 对另一组系统施加小扰动

$$
\theta_t^\delta=\theta_t^0+\delta.
$$

若两组系统随后经历相同环境，并满足

$$
\left\|
\mathbb E\theta_{t+h}^\delta
-
\mathbb E\theta_{t+h}^0
\right\|
\le
C\rho^h\|\delta\|,
\qquad 0<\rho<1,
$$

则外加扰动的平均响应逐渐消失。本文称这一性质为**扰动恢复**。

再选择一个表示演化方向的物理量

$$
\Psi(\theta,e).
$$

例如，它可以表示累积损伤、资源消耗或外部驱动强度。若未扰动参考轨迹满足

$$
\mathbb E[
\Psi(\theta_{t+1}^0,e_{t+1})
-
\Psi(\theta_t^0,e_t)
\mid\mathcal F_t]
\ge \nu>0,
$$

其中 $\mathcal F_t$ 表示时刻 $t$ 以前已经发生的历史。该条件表示宏观规律沿 $\Psi$ 的正方向演化。

三种运动可以同时存在：

$$
\boxed{
\begin{aligned}
&L_{\theta_t} &&\text{推动宏观状态},\\
&\theta_t^\delta-\theta_t^0 &&\text{描述扰动恢复},\\
&\Psi(\theta_t^0,e_t) &&\text{描述规律的有向演化}.
\end{aligned}
}
$$

### 1.2 时间窗口的位置

设 $\mathcal D$ 是当前宏观规律的适用域。规律轨迹首次离开该区域的时刻为

$$
\tau_{\mathcal D}
:=
\inf\{t\ge0:(\theta_t,e_t)\notin\mathcal D\}.
$$

这个时刻给出规律的有效时间窗口。轨迹 $\theta_t$ 和方向量 $\Psi$ 则给出规律怎样变化、朝哪里变化以及怎样到达边界。

### 1.3 本文的问题

本文依次回答四个问题：

1. 微观动力如何生成当前宏观方程？
2. 怎样测量宏观规律对扰动的恢复？
3. 怎样测量未扰动规律自身的有向变化？
4. 哪些条件使稳定演化最终到达失效边界？

全文考察固定任务和粗粒化变量 $q$ 下，规律参数随时间变化的情形。

---

## 2. 一个贯穿全文的物理图像

先考虑 $N$ 个可以正常或故障的部件。令

$$
x_i=
\begin{cases}
0,&\text{部件 }i\text{ 正常},\\
1,&\text{部件 }i\text{ 故障},
\end{cases}
$$

微观状态为

$$
X=(x_1,\ldots,x_N)\in\{0,1\}^N.
$$

宏观变量取故障比例

$$
y=\frac1N\sum_{i=1}^N x_i.
$$

设 $\lambda(z)$ 是故障率，$\mu$ 是修复率，$z$ 是累积损伤。大系统极限下，故障比例满足

$$
\dot y
=
\lambda(z)(1-y)-\mu y.
$$

系统根据当前故障比例调节修复能力：

$$
\dot\mu
=
\kappa(y-y_*),
\qquad
0\le\mu\le\mu_{\max}.
$$

损伤持续积累：

$$
\dot z=\varepsilon,
\qquad
\varepsilon>0.
$$

这三个方程已经包含全文的主线。

- 固定 $(\mu,z)$ 时，微观部件跳变生成 $y$ 的宏观运动。
- 固定 $z$ 时，反馈使 $(y,\mu)$ 回到稳定平衡。
- $z$ 增长时，稳定平衡沿确定方向移动。
- $\mu_{\max}$ 给出系统的最大维持能力。

固定 $z$ 后，希望维持 $y=y_*$ 所需的修复能力为

$$
\mu^*(z)
=
\lambda(z)\frac{1-y_*}{y_*}.
$$

当

$$
\mu^*(z)<\mu_{\max},
$$

在 $\varepsilon$ 足够小、初态靠近平衡且仍有容量裕度时，系统便能跟随这条移动平衡。随着 $z$ 增长，$\mu^*(z)$ 逐渐接近容量上限。越过临界点后，目标平衡失去可行性，故障比例转而上升。

后文先建立一般框架，再回到这个模型给出完整证明。

---

## 3. 从微观动力到演化中的宏观方程

### 3.1 固定参数下的闭合

设 $B$ 是有限微观状态集，$Y$ 是有限宏观状态集。满射

$$
q:B\to Y
$$

把微观状态映到宏观状态。若任务变量 $g:B\to O$ 可以写成

$$
g=r\circ q,
$$

则 $q$ 保留了任务所需的信息。

本节先采用离散时间。给定规律参数 $\theta$ 和环境 $e$，微观 Markov 转移矩阵记为 $K_{\theta,e}$，候选宏观转移矩阵记为 $L_{\theta,e}$。

用 $Q_q$ 表示映射 $q$ 对应的确定性转移矩阵。定义一步闭合误差

$$
\varepsilon_{\mathrm{cl}}(\theta,e)
:=
d_\infty(
K_{\theta,e}Q_q,
Q_qL_{\theta,e}
),
$$

其中

$$
d_\infty(P,P')
:=
\max_x
\frac12\sum_z|P(x,z)-P'(x,z)|.
$$

对单个概率分布，相同距离简记为

$$
\|\mu-\nu\|_{\mathrm{TV}}
:=
\frac12\sum_z|\mu(z)-\nu(z)|.
$$

联合状态含连续变量时，总变差改用对可测事件的上确界；后面的乘积分解与结论保持相同。

误差为零时，同一宏观状态内的所有微观状态都给出相同的下一宏观状态分布。误差很小时，$L_{\theta,e}$ 在一步尺度上近似描述粗粒化后的运动。

### 3.2 规律参数化

选择一个低维宏观方程族

$$
(\theta,e)\longmapsto L_{\theta,e}.
$$

参数差异应对应可辨的宏观方程差异。本文在所研究的局部区域内采用条件

$$
c\|\theta-\theta'\|
\le
d_\infty(
L_{\theta,e},
L_{\theta',e}
)
\le
C\|\theta-\theta'\|,
\qquad c,C>0.
$$

左侧保证不同参数对应可区分的宏观方程；右侧保证参数与方程同步连续变化。这里“最优”指使闭合误差最小的参数。本文进一步假设：对每个被研究的微观转移矩阵，最优参数唯一，并随环境连续变化。这样便得到一条连续可辨的规律轨迹

$$
t\longmapsto L_{\theta_t,e_t}.
$$

### 3.3 参数演化与联合闭合

规律参数和环境按照当前宏观状态更新：

$$
(\theta_{t+1},e_{t+1})
\sim
M(Y_t,\theta_t,e_t).
$$

更新规则 $M$ 由已保留的变量 $(Y_t,\theta_t,e_t)$ 决定。对应同一宏观状态的微观状态因而使用同一套参数更新规则。给定当前联合状态后，参数—环境更新与微观转移条件独立。

定义联合微观状态和联合宏观状态

$$
\widetilde X_t=(X_t,\theta_t,e_t),
\qquad
\widetilde Y_t=(q(X_t),\theta_t,e_t).
$$

联合微观核 $\widetilde K$ 由 $K_{\theta,e}$ 与 $M(q(x),\theta,e)$ 的乘积定义；联合宏观核 $\widetilde L$ 则由 $L_{\theta,e}$ 与同一个 $M(y,\theta,e)$ 的乘积定义。这里 $M$ 读取更新前的宏观状态。

**定理 1（参数演化保持原有闭合误差）。**
在上述条件下，联合系统从 $\widetilde X_t$ 到 $\widetilde Y_t$ 的一步闭合误差满足

$$
d_\infty(
\widetilde KQ_{\widetilde q},
Q_{\widetilde q}\widetilde L
)
=
\sup_{\theta,e}
\varepsilon_{\mathrm{cl}}(\theta,e).
$$

这里

$$
\widetilde q(x,\theta,e)
=
(q(x),\theta,e).
$$

**证明。** 固定当前状态 $(x,\theta,e)$，记 $y=q(x)$。微观更新后投影到下一宏观状态的分布为

$$
(K_{\theta,e}Q_q)(x,y')\,
M(y,\theta,e;\theta',e').
$$

宏观模型给出的分布为

$$
L_{\theta,e}(y,y')\,
M(y,\theta,e;\theta',e').
$$

两式含有相同的参数—环境更新因子 $M$。对 $(\theta',e')$ 求和后，二者的总变差距离恰好等于

$$
\left\|
(K_{\theta,e}Q_q)(x,\cdot)
-
L_{\theta,e}(q(x),\cdot)
\right\|_{\mathrm{TV}}.
$$

再对 $x,\theta,e$ 取上确界得到结论。 $\square$

这个定理说明，规律参数随时间变化时，原有闭合误差仍然控制宏观方程的瞬时准确度。参数更新若进一步读取微观细节，相应细节需要进入宏观状态。

### 3.4 有限时间预测

考虑一列预先给定的微观转移矩阵 $K_t$ 和宏观转移矩阵 $L_t$，并设

$$
d_\infty(K_tQ_q,Q_qL_t)
\le\varepsilon_t.
$$

则

$$
d_\infty(
K_0\cdots K_{T-1}Q_q,
Q_qL_0\cdots L_{T-1}
)
\le
\sum_{t=0}^{T-1}\varepsilon_t.
$$

证明来自逐项展开：在第 $t$ 步插入误差

$$
K_tQ_q-Q_qL_t,
$$

其余 Markov 转移保持总变差距离不增。特别地，当每一步误差都不超过 $\varepsilon$ 时，$T$ 步终端误差不超过 $T\varepsilon$。

---

## 4. 分别测量扰动恢复与有向演化

### 4.1 未扰动轨迹确定参考路径

设未扰动数据给出规律参数轨迹

$$
r_0,r_1,\ldots,r_T.
$$

对任意矩阵序列 $A_t$，定义路径附近的更新

$$
F_t^A(\theta)
:=
r_{t+1}
+
A_t(\theta-r_t).
$$

所有这些更新都满足

$$
F_t^A(r_t)=r_{t+1}.
$$

因此，同一条参考路径可以对应不同的邻域响应。取 $A_t=0$ 时，扰动一步消失；取 $A_t=I$ 时，扰动保持；取 $A_t=2I$ 时，扰动放大。

**命题 2（同一参考路径对应任意局部响应）。**
若在时刻 $t$ 施加扰动 $\delta$，且迭代轨迹留在所定义的参数域内，则对 $1\le h\le T-t$ 有

$$
\theta_{t+h}^\delta-r_{t+h}
=
A_{t+h-1}\cdots A_t\delta.
$$

**证明。** 第一步有

$$
F_t^A(r_t+\delta)-r_{t+1}
=
A_t\delta.
$$

逐步代入便得到矩阵乘积。 $\square$

参考路径给出规律怎样移动；配对扰动给出规律怎样响应偏离。

### 4.2 配对扰动实验

设规律参数空间为 $\mathbb R^p$。在同一时刻制备一组对照系统和 $p$ 组扰动系统：

$$
\theta_t^0=r_t,
\qquad
\theta_t^{(j)}=r_t+\delta_j,
\qquad j=1,\ldots,p.
$$

各组随后经历相同的环境变化和反馈规则。路径附近的平均更新写成

$$
\mathbb E[
\theta_{t+1}\mid\theta_t
]
=
r_{t+1}
+
A_t(\theta_t-r_t)
+
O(\|\theta_t-r_t\|^2).
$$

这里 $A_t$ 是平均更新在参考轨迹处对 $\theta_t$ 的导数。

定义一步平均响应

$$
\Delta_{t,j}
:=
\mathbb E\theta_{t+1}^{(j)}
-
\mathbb E\theta_{t+1}^{0}.
$$

当平均更新在所用扰动范围内为线性时，

$$
\Delta_{t,j}
=
A_t\delta_j.
$$

令

$$
D=[\delta_1,\ldots,\delta_p],
\qquad
\Delta_t=[\Delta_{t,1},\ldots,\Delta_{t,p}].
$$

**定理 3（配对扰动确定局部响应矩阵）。**
若平均更新在扰动邻域内为线性，并取 $p$ 个线性无关的扰动方向，使方阵 $D$ 可逆，则

$$
\boxed{
A_t=\Delta_tD^{-1}.
}
$$

**证明。** 各列满足 $\Delta_{t,j}=A_t\delta_j$，合并后得到

$$
\Delta_t=A_tD.
$$

右乘 $D^{-1}$ 即得结论。 $\square$

对一般光滑更新，取 $D=dD_0$，其中 $D_0$ 固定且可逆；令 $d\to0$，估计量 $\Delta_tD^{-1}$ 收敛到导数 $A_t$。有限扰动产生的二阶误差见附录 A。

在线性响应极限中，当

$$
\|A_t\|\le\rho<1
$$

成立时，一步局部平均响应满足

$$
\limsup_{\|\delta\|\to0}
\frac{
\left\|
\mathbb E\theta_{t+1}^\delta
-
\mathbb E\theta_{t+1}^0
\right\|
}{\|\delta\|}
\le
\rho.
$$

对一般随机更新，这就是配对实验直接给出的结论。若规律参数按确定性光滑递推演化，各步轨迹留在同一局部邻域，且 $\|A_{t+j}\|\le\rho<1$ 一致成立，则链式法则进一步给出

$$
\limsup_{\|\delta\|\to0}
\frac{
\left\|
\theta_{t+h}^\delta
-
\theta_{t+h}^0
\right\|
}{\|\delta\|}
\le
\rho^h.
$$

这给出可逐步复合的局部扰动恢复。实验调节部分规律参数时，扰动方向张成相应的可控子空间即可。

### 4.3 测量参考轨迹的方向

选择一个预先给定的方向量

$$
\Psi(\theta,e).
$$

在对照组中测量

$$
J_t
:=
\mathbb E[
\Psi(\theta_{t+1}^0,e_{t+1})
-
\Psi(\theta_t^0,e_t)
\mid\mathcal F_t].
$$

条件

$$
J_t\ge\nu>0
$$

表示规律沿 $\Psi$ 的正方向持续演化。方向量应由具体物理问题预先指定；故障—修复系统中可以取

$$
\Psi=\lambda(z),
$$

即损伤导致的故障压力。

恢复和演化由两组互补的量给出：

$$
\boxed{
\begin{aligned}
\text{恢复：}\quad&
\mathbb E\theta_{t+h}^\delta
-
\mathbb E\theta_{t+h}^0,\\
\text{演化：}\quad&
\mathbb E\!\left[
\Psi(\theta_{t+1}^0,e_{t+1})
-
\Psi(\theta_t^0,e_t)
\mid\mathcal F_t
\right].
\end{aligned}
}
$$

---

## 5. 沿移动稳定支的演化与退出

### 5.1 移动稳定支

考虑快变量 $x$ 和慢变量 $s$：

$$
\dot x=f(x,s),
\qquad
\dot s=\varepsilon v(s),
\qquad
0<\varepsilon\ll1.
$$

对每个固定 $s$，设方程

$$
f(x,s)=0
$$

有一个平衡点 $x_*(s)$。这些平衡点组成一条平衡支。

设线性化矩阵

$$
A(s)
:=
D_xf(x_*(s),s)
$$

在区间 $s\in[s_0,s_c)$ 上一致稳定。具体地，存在连续正定矩阵 $P(s)$ 和常数 $a>0$，使

$$
A(s)^\top P(s)+P(s)A(s)
\le
-aI.
$$

再假设存在 $0<m<M<\infty$ 使

$$
mI\le P(s)\le MI,
$$

并且 $f$ 在稳定支的统一邻域内二阶连续可微，$x_*',P',v$ 以及 $f$ 的一、二阶导数在该邻域内一致有界。

**定理 4（移动稳定支的跟踪）。**
存在常数 $C_1,C_2,\gamma>0$。当 $\varepsilon$ 足够小、初态靠近稳定支时，在 $s(t)<s_c$ 的整个时间内，

$$
\boxed{
\|x(t)-x_*(s(t))\|
\le
C_1e^{-\gamma t}
\|x(0)-x_*(s_0)\|
+
C_2\varepsilon.
}
$$

若 $x^\delta(t)$ 与 $x^0(t)$ 经历同一条 $s(t)$，初始差为 $\delta$，且两条轨迹始终留在同一个足够小的稳定邻域内，则

$$
\boxed{
\|x^\delta(t)-x^0(t)\|
\le
C_1e^{-\gamma t}\|\delta\|.
}
$$

第一式说明系统跟随移动平衡，第二式说明外加扰动相对参考轨迹衰减。

**证明。** 令

$$
e(t)
:=
x(t)-x_*(s(t)).
$$

在稳定支附近展开 $f$：

$$
\dot e
=
A(s)e
+
R(e,s)
-
\varepsilon v(s)x_*'(s),
$$

其中

$$
\|R(e,s)\|
\le
C\|e\|^2.
$$

取

$$
W(e,s)=e^\top P(s)e.
$$

沿轨迹求导。线性项由稳定条件给出 $-a\|e\|^2$；非线性项为 $O(\|e\|^3)$；稳定支移动带来的项为 $O(\varepsilon\|e\|)$；$P(s)$ 的变化带来 $O(\varepsilon\|e\|^2)$。在足够小的邻域内，

$$
\dot W
\le
-cW+C_0\varepsilon^2.
$$

积分得到

$$
W(t)
\le
e^{-ct}W(0)
+
\frac{C_0}{c}\varepsilon^2.
$$

由于 $W$ 与 $\|e\|^2$ 在整个区间上一致等价，第一式成立。

再比较 $x^\delta$ 与 $x^0$。两条轨迹共享同一个 $s(t)$，稳定支移动项在相减时消失。若邻域半径为 $r$，中值定理给出

$$
\|R(e^\delta,s)-R(e^0,s)\|
\le Cr\|e^\delta-e^0\|.
$$

取 $r$ 足够小，这一项可由稳定线性项吸收。对差

$$
d=x^\delta-x^0
$$

作同样估计得到

$$
\frac{d}{dt}\bigl(d^\top P(s)d\bigr)
\le
-c_1d^\top P(s)d.
$$

积分后得到第二式。 $\square$

这一定理给出一个直接的物理图像：系统在横向上回到稳定支，同时沿纵向参数 $s$ 持续前进。

### 5.2 到达边界的时间

若存在 $v_0>0$ 使

$$
v(s)\ge v_0,
$$

则 $s(t)$ 单调增加。从 $s_0$ 到 $s_c$ 的时间为

$$
\boxed{
\tau_c
=
\frac1\varepsilon
\int_{s_0}^{s_c}
\frac{ds}{v(s)}.
}
$$

稳定支在 $s_c$ 处终止、失去稳定性或超过维持能力时，$\tau_c$ 给出到达该边界的时间；一般情形下，积分有限恰好对应有限时间到达。时间尺度由方向速度 $v(s)$ 和到边界的距离共同决定。

### 5.3 随机规律轨迹

有限 Markov 系统中的规律参数可以随机变化。记 $\Psi_t=\Psi(\theta_t,e_t)$，设初态位于规律适用域 $\mathcal D$，$\tau_{\mathcal D}$ 是首次离开时间。若 $\Psi_t$ 适应历史 $\mathcal F_t$ 且可积，并且

$$
\mathbb E[
\Psi_{t+1}-\Psi_t
\mid\mathcal F_t]
\ge\nu>0,
\qquad \text{在事件 }\{t<\tau_{\mathcal D}\}\text{ 上},
$$

并且从 $\mathcal D$ 内状态一步可达的全部状态都满足

$$
\Psi\le\Psi_{\max},
$$

则有

$$
\boxed{
\mathbb E\tau_{\mathcal D}
\le
\frac{
\Psi_{\max}-\mathbb E\Psi_0
}{\nu}.
}
$$

**证明。** 对停止过程逐步求和：

$$
\mathbb E\Psi_{n\wedge\tau_{\mathcal D}}
-\mathbb E\Psi_0
=
\sum_{t=0}^{n-1}
\mathbb E\!\left[
\mathbf 1_{\{t<\tau_{\mathcal D}\}}
(\Psi_{t+1}-\Psi_t)
\right].
$$

漂移条件于是给出

$$
\mathbb E\Psi_{n\wedge\tau_{\mathcal D}}
-
\mathbb E\Psi_0
\ge
\nu\,
\mathbb E(n\wedge\tau_{\mathcal D}).
$$

左侧不超过 $\Psi_{\max}-\mathbb E\Psi_0$。令 $n\to\infty$ 即得结论。 $\square$

统一的正增量下界和沿该方向的有限边界共同给出有限离开时间。

---

## 6. 故障—修复系统

本节改用连续时间。固定参数时，以微观生成元 $\mathcal G_{\mu,z}$ 和宏观生成元 $\bar{\mathcal G}_{\mu,z}$ 代替第 3 节的转移矩阵；关系

$$
\mathcal G_{\mu,z}Q_q
=
Q_q\bar{\mathcal G}_{\mu,z}
$$

保证相应转移半群精确闭合。

### 6.1 微观模型与精确宏观方程

系统由 $N$ 个部件组成。每个正常部件以速率 $\lambda(z)$ 故障，每个故障部件以速率 $\mu$ 修复。令

$$
k=\sum_{i=1}^N x_i
$$

为故障部件数。

固定 $(\mu,z)$ 后，若当前有 $k$ 个故障，则

$$
k\longrightarrow k+1
\quad\text{的速率为}\quad
(N-k)\lambda(z),
$$

$$
k\longrightarrow k-1
\quad\text{的速率为}\quad
k\mu.
$$

**命题 5（故障数精确闭合）。**
映射

$$
q_N(x)=\sum_{i=1}^Nx_i
$$

把 $2^N$ 个微观状态精确压缩为 $N+1$ 个宏观状态。固定 $(\mu,z)$ 时，故障数本身构成 Markov 生灭过程，其转移速率就是上面两式。

**证明。** 任意具有 $k$ 个故障的微观状态都含有 $N-k$ 个正常部件和 $k$ 个故障部件。下一次增加或减少一个故障的总速率由 $k$ 完全决定。 $\square$

这一步实现了第一种运动：微观部件的故障—修复跳变生成故障数的宏观运动方程。

### 6.2 大系统极限与反馈修复

定义故障比例

$$
y_N=\frac{k}{N}.
$$

在速率局部 Lipschitz、相关轨迹有界等标准条件下，若初始故障比例收敛，则在 $N\to\infty$ 的密度极限中，随机生灭过程在任意有限时间区间上依概率一致收敛到确定性方程 [9]：

$$
\dot y
=
\lambda(z)(1-y)-\mu y.
$$

设目标故障比例为 $y_*\in(0,1)$。修复系统按

$$
\dot\mu
=
\kappa(y-y_*),
\qquad
\kappa>0,
$$

调节修复能力。到达 $0$ 或 $\mu_{\max}$ 后，令指向区间外的变化率变为零。将这一规则写成

$$
F_\mu(y)
=
\begin{cases}
\max\{\kappa(y-y_*),0\},&\mu=0,\\
\kappa(y-y_*),&0<\mu<\mu_{\max},\\
\min\{\kappa(y-y_*),0\},&\mu=\mu_{\max}.
\end{cases}
$$

于是 $0\le\mu(t)\le\mu_{\max}$。

损伤变量满足

$$
\dot z=\varepsilon,
\qquad
z(0)=z_0,
\qquad
\varepsilon>0.
$$

于是完整宏观方程为

$$
\boxed{
\begin{aligned}
\dot y
&=
\lambda(z)(1-y)-\mu y,\\
\dot\mu
&=
F_\mu(y),\\
\dot z
&=
\varepsilon.
\end{aligned}
}
$$

### 6.3 固定损伤时的恢复

固定 $z$ 后，目标平衡为

$$
y=y_*,
\qquad
\mu=\mu^*(z)
:=
\lambda(z)\frac{1-y_*}{y_*}.
$$

当

$$
0<\mu^*(z)<\mu_{\max},
$$

这个平衡位于可调范围内部。

在平衡点附近，$(y,\mu)$ 方程的线性化矩阵为

$$
J_z
=
\begin{pmatrix}
-(\lambda(z)+\mu^*(z))&-y_*\\
\kappa&0
\end{pmatrix}.
$$

它的迹和行列式分别为

$$
\operatorname{tr}J_z
=
-(\lambda(z)+\mu^*(z))<0,
$$

$$
\det J_z
=
\kappa y_*>0.
$$

因此两个特征值的实部都为负，平衡局部指数稳定。对修复能力施加小扰动

$$
\mu(t_0)\longmapsto\mu(t_0)+\delta
$$

后，扰动组与对照组在相同损伤路径下指数靠近。这实现了第二种运动：宏观规律对偏离具有恢复能力。

### 6.4 损伤增长时的稳定跟踪

现在让 $z$ 缓慢增长。设 $\lambda$ 光滑且

$$
\lambda'(z)>0.
$$

移动平衡

$$
\bigl(y_*,\mu^*(z)\bigr)
$$

随 $z$ 沿确定方向移动。取一个损伤区间 $I=[z_0,z_1]$，并假设存在裕度 $\eta>0$，使

$$
\eta
\le
\mu^*(z)
\le
\mu_{\max}-\eta,
\qquad z\in I.
$$

**定理 6（反馈系统跟随移动平衡）。**
存在常数 $C_1,C_2,\gamma>0$。当 $\varepsilon$ 足够小、初态靠近移动平衡，并且

$$
C_1E_0+C_2\varepsilon<\eta,
$$

时，在 $z(t)\in I$ 的整个时间内，

$$
\boxed{
|y(t)-y_*|
+
|\mu(t)-\mu^*(z(t))|
\le
C_1e^{-\gamma t}E_0
+
C_2\varepsilon,
}
$$

其中

$$
E_0
=
|y(0)-y_*|
+
|\mu(0)-\mu^*(z_0)|.
$$

两组经历同一 $z(t)$ 的系统若初态相差 $\delta$，还满足

$$
\boxed{
\left\|
\begin{pmatrix}
y^\delta(t)-y^0(t)\\
\mu^\delta(t)-\mu^0(t)
\end{pmatrix}
\right\|
\le
C_1e^{-\gamma t}\|\delta\|.
}
$$

**证明。** 这个系统符合定理 4。取快变量

$$
x=(y,\mu),
$$

慢变量

$$
s=z.
$$

固定 $z$ 时的平衡为

$$
x_*(z)=(y_*,\mu^*(z)).
$$

上一节已经证明线性化矩阵 $J_z$ 稳定。对每个 $z\in I$，令 $P(z)$ 解

$$
J_z^\top P(z)+P(z)J_z=-I.
$$

区间 $I$ 紧，且平衡始终与容量边界保持距离，因此 $P(z)$ 光滑、统一正定，稳定常数可以在整个区间上一致选取。误差小于裕度 $\eta$ 又保证边界规则不会被触发。将定理 4 应用于这条稳定支，便得到两式。 $\square$

该定理同时表达两件事：扰动造成的横向偏差指数衰减，累积损伤使平衡点沿纵向持续移动。

### 6.5 有向变化与容量边界

宏观方程中的故障压力为

$$
\Psi=\lambda(z).
$$

沿未扰动参考轨迹，

$$
\frac{d\Psi}{dt}
=
\lambda'(z)\varepsilon
>0.
$$

因此损伤推动宏观规律有向演化。系统为了维持 $y_*$，需要不断提高

$$
\mu^*(z)
=
\lambda(z)\frac{1-y_*}{y_*}.
$$

假设方程

$$
\mu^*(z_c)=\mu_{\max}
$$

有解。由于 $\lambda'(z)>0$，该解唯一，记为临界损伤 $z_c$。当 $z_c\ge z_0$ 时，系统到达该点的时间为

$$
\tau_c
=
\frac{z_c-z_0}{\varepsilon}.
$$

这个时间来自规律沿损伤方向的运动。越过 $z_c$ 后，维持 $y=y_*$ 所需的修复率超过容量。此时的饱和边界平衡为

$$
\mu=\mu_{\max},
\qquad
\bar y(z)
=
\frac{\lambda(z)}{\lambda(z)+\mu_{\max}}
>y_*.
$$

固定 $z>z_c$ 后，可行平衡位于这条饱和支上；损伤继续增长时，$\bar y(z)$ 随之上升。下面的失效时间界直接使用容量上限，因而不依赖慢变轨迹是否跟随这条边界支。

### 6.6 容量达到上限后的任务失效

取任务失效阈值

$$
y_f\in(y_*,1).
$$

若存在 $z_f$ 使

$$
c_f
:=
\lambda(z_f)(1-y_f)
-
\mu_{\max}y_f
>0,
$$

则对所有

$$
z\ge z_f,
\qquad
y\le y_f,
\qquad
\mu\le\mu_{\max},
$$

都有

$$
\dot y
=
\lambda(z)(1-y)-\mu y
\ge c_f.
$$

**定理 7（有限维持能力给出有限失效时间）。**
令

$$
t_f
=
\frac{(z_f-z_0)_+}{\varepsilon}.
$$

若系统在 $t_f$ 时尚未越过 $y_f$，则它在此后至多经过

$$
\boxed{
\frac{y_f-y(t_f)}{c_f}
}
$$

的时间到达任务边界。

**证明。** 在 $y\le y_f$ 的时间内，$\dot y\ge c_f$。对时间积分即可。 $\square$

故障—修复模型由此完成整条理论链：

$$
\boxed{
\begin{gathered}
\text{部件跳变}
\longrightarrow
\text{故障数闭合},\\
\text{修复反馈}
\longrightarrow
\text{扰动恢复},\\
\text{累积损伤}
\longrightarrow
\text{移动平衡},\\
\text{有限修复能力且损伤达到 }z_f
\longrightarrow
\text{任务失效}.
\end{gathered}
}
$$

### 6.7 四种机制组合

| 修复反馈 $\kappa$ | 损伤速度 $\varepsilon$ | 系统行为 |
|---:|---:|---|
| $0$ | $0$ | 修复率固定，其扰动持续存在 |
| $>0$ | $0$ | 系统恢复扰动，参考平衡固定 |
| $0$ | $>0$ | 规律沿损伤方向变化，修复率扰动持续存在 |
| $>0$ | $>0$ | 系统跟随移动平衡；若定理 7 的条件成立，则有限时间内越过任务边界 |

---

## 7. 物理含义与适用范围

### 7.1 恢复、方向和退出承担不同作用

恢复描述扰动组向对照组靠近：

$$
\|\mathbb E\theta_{t+h}^\delta
-
\mathbb E\theta_{t+h}^0\|
\longrightarrow0.
$$

方向描述对照组自身的移动：

$$
\mathbb E(\Psi_{t+1}-\Psi_t)
>0.
$$

退出描述规律轨迹到达适用域边界：

$$
\tau_{\mathcal D}
=
\inf\{t:(\theta_t,e_t)\notin\mathcal D\}.
$$

三个条件分别回答“偏差是否消失”“参考轨迹朝哪里移动”和“轨迹何时到达边界”。

下面两个过程进一步区分“具有方向”和“到达边界”。

第一，有限格点区间上的对称随机游走以两个端点为吸收边界；它在内部具有零条件平均方向：

$$
\mathbb E(\theta_{t+1}-\theta_t\mid\theta_t)=0,
$$

它仍在有限时间到达吸收边界，尽管条件平均方向为零。

第二，令

$$
d_{t+1}=\frac{1+d_t}{2},
\qquad 0\le d_0<1.
$$

$d_t$ 单调增加并趋近 $1$。若规律适用域包含极限点 $d=1$，轨迹始终留在适用域内。统一的正增量下界和沿该方向的有限边界共同给出第 5.3 节中的有限退出结论。

### 7.2 本文研究的规律层次

本文固定任务 $g$、宏观映射 $q$ 和观察尺度，研究宏观方程 $L_{\theta_t,e_t}$ 随物理时间的变化。所有结论都相对于这些选择；宏观表示的重组与空间尺度流留给后续研究。

### 7.3 与相关研究的关系

给定粗粒化映射以后，有限 Markov 链何时保持 Markov 性属于经典聚合理论 [1,2]。近似理论进一步研究粗粒化误差及其随时间的传播 [3,4]。本文在这些闭合工具上研究随时间变化的宏观方程族 $L_{\theta_t,e_t}$。

反馈控制和系统生物学研究系统怎样在参数变化下维持输出 [5,6]。相关研究也指出，未扰动轨迹和模型参数之间可能存在多种等价解释 [7,8]。本文采用配对扰动实验，直接测量参考轨迹附近的响应矩阵，并把这一步与任务相对闭合连接起来。

---

## 8. 结论

本文把有效宏观规律表示为一条由微观动力学支持的宏观方程轨迹。

$$
\boxed{
\begin{aligned}
K_{\theta_t,e_t}Q_q
&\simeq
Q_qL_{\theta_t,e_t},
\\[0.3em]
\limsup_{\|\delta\|\to0}
\frac{
\|\mathbb E\theta_{t+h}^\delta
-\mathbb E\theta_{t+h}^0\|
}{\|\delta\|}
&\le
\rho^h,
\\[0.3em]
\mathbb E(\Psi_{t+1}-\Psi_t\mid\mathcal F_t)
&\ge
\nu>0.
\end{aligned}
}
$$

第一式说明当前宏观方程具有微观依据。第二式说明在可逐步复合的局部响应条件下，微小扰动衰减。第三式说明未扰动参考轨迹沿预先选定的物理方向演化。

当冻结系统沿一条一致稳定的平衡支演化、参数变化足够慢时，系统可以跟随这条移动稳定支。若维持目标所需的能力达到上限，该目标维持支便到达容量边界。规律轨迹到达这个边界的时刻构成有效时间窗口。

故障—修复模型把这条逻辑具体化：部件跳变生成故障数方程，修复反馈维持目标故障比例，累积损伤推动所需修复能力上升；若损伤继续增长并满足定理 7 的条件，系统将在有限时间内越过任务边界。

宏观规律由此呈现一个完整的动力过程：

$$
\boxed{
\text{生成}
\longrightarrow
\text{维持}
\longrightarrow
\text{有向演化}
\longrightarrow
\text{条件失效}.
}
$$

---

## 附录 A：规律参数和响应矩阵的误差

正文定理 3 假设规律参数可以准确测量。本附录说明闭合误差和有限数据误差怎样传递到响应矩阵估计。

### A.1 从转移矩阵估计规律参数

设真实规律参数为 $\theta$，并满足

$$
d_\infty(KQ_q,Q_qL_{\theta,e})
\le\varepsilon.
$$

由数据估计得到投影转移矩阵 $\widehat P$，设

$$
d_\infty(\widehat P,KQ_q)
\le s.
$$

把参数限制在一个紧区域 $\Theta$ 内，并假设第 3.2 节的参数化上下界在该区域成立。取达到下式最小值的 $\widehat\theta\in\Theta$：

$$
d_\infty(
\widehat P,
Q_qL_{\vartheta,e}
)
$$

并设真实参数 $\theta\in\Theta$。由最优性，

$$
d_\infty(
\widehat P,
Q_qL_{\widehat\theta,e}
)
\le
s+\varepsilon.
$$

三角不等式先给出

$$
d_\infty(
Q_qL_{\widehat\theta,e},
Q_qL_{\theta,e}
)
\le
2(s+\varepsilon).
$$

由于 $q$ 满射，$Q_q$ 会逐行列出 $L$ 的全部行，因此

$$
d_\infty(Q_qL,Q_qL')
=
d_\infty(L,L').
$$

再利用规律参数化的下界

$$
c\|\widehat\theta-\theta\|
\le
d_\infty(
L_{\widehat\theta,e},
L_{\theta,e}
),
$$

得到

$$
\boxed{
\|\widehat\theta-\theta\|
\le
\frac{2(s+\varepsilon)}{c}.
}
$$

参数反演的稳定常数 $1/c$ 控制误差放大。

### A.2 响应矩阵的误差

允许平均响应含有二阶项：

$$
\mathbb E\theta_{t+1}^{(j)}
-
\mathbb E\theta_{t+1}^{0}
=
A_t\delta_j
+
R_t(\delta_j),
$$

$$
\|R_t(\delta_j)\|
\le
\frac H2\|\delta_j\|^2.
$$

设每组参数均值的总估计误差不超过 $h_j$，对照组误差不超过 $h_0$。下文对矩阵采用 Frobenius 范数，对 $D^{-1}$ 采用算子范数。令

$$
\widehat A
=
[\widehat\Delta_1,\ldots,\widehat\Delta_p]D^{-1}.
$$

则

$$
\boxed{
\|\widehat A-A_t\|
\le
\left[
\sum_{j=1}^p
\left(
\frac H2\|\delta_j\|^2+h_j+h_0
\right)^2
\right]^{1/2}
\|D^{-1}\|.
}
$$

若采用等幅正交扰动 $\delta_j=de_j$，并把闭合误差、转移数据误差和组间均值误差分别记为 $\varepsilon,s,\zeta$，且每个扰动组与对照组都满足

$$
h_j,h_0
\le
\frac{2(\varepsilon+s)}c+\zeta,
$$

则上式化为

$$
\|\widehat A-A_t\|
\le
\sqrt p
\left[
\frac H2d
+
\frac{4(\varepsilon+s)}{cd}
+
\frac{2\zeta}{d}
\right].
$$

较大的扰动增强信号，同时增加非线性误差；较小的扰动降低非线性误差，同时放大统计误差。这个式子给出选择扰动幅度的直接依据。

---

## 参考文献

1. J. G. Kemeny and J. L. Snell, *Finite Markov Chains*, Springer, 1976.
2. P. Buchholz, “Exact and Ordinary Lumpability in Finite Markov Chains,” *Journal of Applied Probability* 31, 59–75, 1994. [DOI](https://doi.org/10.2307/3215235)
3. G. Bian and A. Abate, “On the Relationship between Bisimulation and Trace Equivalence in an Approximate Probabilistic Context,” FoSSaCS, 2017. [DOI](https://doi.org/10.1007/978-3-662-54458-7_19)
4. B. Hilder and U. Sharma, “Quantitative Coarse-Graining of Markov Chains,” *SIAM Journal on Mathematical Analysis* 56, 913–954, 2024. [DOI](https://doi.org/10.1137/22M1473996)
5. O. Karin, A. Swisa, B. Glaser, Y. Dor, and U. Alon, “Dynamical Compensation in Physiological Circuits,” *Molecular Systems Biology* 12, 886, 2016. [DOI](https://doi.org/10.15252/msb.20167216)
6. C. Briat, A. Gupta, and M. Khammash, “Antithetic Integral Feedback Ensures Robust Perfect Adaptation in Noisy Biomolecular Networks,” *Cell Systems* 2, 15–26, 2016. [DOI](https://doi.org/10.1016/j.cels.2016.01.004)
7. E. D. Sontag, “Dynamic Compensation, Parameter Identifiability, and Equivariances,” *PLOS Computational Biology* 13, e1005447, 2017. [DOI](https://doi.org/10.1371/journal.pcbi.1005447)
8. A. F. Villaverde and J. R. Banga, “Dynamical Compensation and Structural Identifiability of Biological Models,” *PLOS Computational Biology* 13, e1005878, 2017. [DOI](https://doi.org/10.1371/journal.pcbi.1005878)
9. S. N. Ethier and T. G. Kurtz, *Markov Processes: Characterization and Convergence*, Wiley, 1986. [DOI](https://doi.org/10.1002/9780470316658)
