# 跨层权重复合：为什么 $W_Q^2 \cdot W_O^1$ 构成直接连接

## 1. 单层残差结构

第 $\ell$ 层的输出（写入残差流）：

$$x^{(\ell)} = x^{(\ell-1)} + \underbrace{\text{Attn}_\ell(x^{(\ell-1)})}_{h^{(\ell)}_{attn}} + \underbrace{\text{FFN}_\ell(x^{(\ell-1)})}_{h^{(\ell)}_{ffn}}$$

其中 Attention 的输出：

$$h^{(\ell)}_{attn} = \text{Concat}(\text{head}_1, \dots, \text{head}_H) \cdot W_O^\ell$$

单个头的运算（简化写）：

$$\text{head}_h = \text{Softmax}\left(\frac{Q_h K_h^T}{\sqrt{d_k}}\right) \cdot V_h$$

其中：

$$Q_h = x \cdot W_Q^{\ell,h}, \quad K_h = x \cdot W_K^{\ell,h}, \quad V_h = x \cdot W_V^{\ell,h}$$

---

## 2. 展开：Layer 1 写入残差流

Layer 1 的 Attention 写入：

$$h^{(1)}_{attn} = \sum_h \text{head}_h^{(1)} \cdot W_O^{1,h}$$

残差流在 Layer 1 之后：

$$x^{(1)} = x^{(0)} + h^{(1)}_{attn} + h^{(1)}_{ffn}$$

> 残差流 = Embedding + Attn₁ 的贡献 + FFN₁ 的贡献

---

## 3. Layer 2 读取残差流（分配律）

Layer 2 的 Query 投影：

$$Q^{(2)} = x^{(1)} \cdot W_Q^2$$

把 $x^{(1)}$ 展开：

$$Q^{(2)} = \left(x^{(0)} + h^{(1)}_{attn} + h^{(1)}_{ffn}\right) \cdot W_Q^2$$

**分配律**：

$$\boxed{Q^{(2)} = \underbrace{x^{(0)} \cdot W_Q^2}_{\text{来自 Embedding}} + \underbrace{h^{(1)}_{attn} \cdot W_Q^2}_{\text{来自 Layer1 Attn}} + \underbrace{h^{(1)}_{ffn} \cdot W_Q^2}_{\text{来自 Layer1 FFN}}}$$

第三项展开就是关键：

$$h^{(1)}_{attn} \cdot W_Q^2 = \left(\sum_h \text{head}_h^{(1)} \cdot W_O^{1,h}\right) \cdot W_Q^2$$

$$= \sum_h \text{head}_h^{(1)} \cdot \left(W_O^{1,h} \cdot W_Q^2\right)$$

---

## 4. 核心：$W_O^1 \cdot W_Q^2$ 是跨层"虚拟直接连接"

```
表面路径：
  Layer1 Head → [d_head] → W_O¹ → 残差流 [d_model] → W_Q² → Layer2 Q [d_head]

展开后：
  Layer1 Head → [W_O¹ · W_Q²] → Layer2 Q [d_head]
                ↑______________↑
                一个矩阵，残差流"消掉了"
```

$W_O^{1} \cdot W_Q^2$ 的形状：

$$W_O^1 \in \mathbb{R}^{d_{head} \times d_{model}}, \quad W_Q^2 \in \mathbb{R}^{d_{model} \times d_{head}}$$

$$W_O^1 \cdot W_Q^2 \in \mathbb{R}^{d_{head} \times d_{head}}$$

> 它把 Layer 1 某个 head 的输出空间**直接映射**到 Layer 2 某个 head 的 Query 空间，不需要经过残差流（残差流只是载体，不是变换）。

---

## 5. 一般化：任意两层之间

Layer $\ell$ 的 Attention 写入残差流的是：

$$h^{(\ell)}_{attn} = \sum_h \text{head}_h^{(\ell)} \cdot W_O^{\ell,h}$$

Layer $\ell'$（$\ell' > \ell$）的 Q / K / V 都能读取它：

$$Q^{(\ell')} = x^{(\ell'-1)} \cdot W_Q^{\ell'} = \cdots + h^{(\ell)}_{attn} \cdot W_Q^{\ell'} + \cdots$$

因为残差流只加不减，**所有早期层的写入都在**：

$$x^{(\ell'-1)} = x^{(0)} + \sum_{k=1}^{\ell'-1} \left(h^{(k)}_{attn} + h^{(k)}_{ffn}\right)$$

所以任意层对 $(\ell, \ell')$ 的跨层连接：

| 连接 | 矩阵 | 含义 |
|---|---|---|
| $W_O^\ell \cdot W_Q^{\ell'}$ | $d_{head} \times d_{head}$ | Headℓ 输出 → Headℓ' 的 Query |
| $W_O^\ell \cdot W_K^{\ell'}$ | $d_{head} \times d_{head}$ | Headℓ 输出 → Headℓ' 的 Key |
| $W_O^\ell \cdot W_V^{\ell'}$ | $d_{head} \times d_{head}$ | Headℓ 输出 → Headℓ' 的 Value |

---

## 6. 更完整的路径展开（Anthropic 风格）

Layer 2 的 Attention score（QK 内积）：

$$\text{score}^{(2)} = Q^{(2)} \cdot (K^{(2)})^T$$

把两者的分配律展开都代入：

$$\text{score}^{(2)} = \left(\sum_a c_a \cdot W_Q^2\right) \cdot \left(\sum_b c_b \cdot W_K^2\right)^T$$

$$= \sum_{a,b} \left(c_a \cdot W_Q^2\right) \cdot \left(c_b \cdot W_K^2\right)^T$$

每一项 $c_a \cdot W_Q^2 \cdot (W_K^2)^T \cdot c_b^T$ 就是一个**路径**：

> "残差流上的组件 $c_a$ 通过 Q 路由读取组件 $c_b$ 通过 K 路由"

---

## 7. 范畴论总结

| 范畴论概念 | Transformer 对应 |
|---|---|
| **对象** | 残差流空间 $\mathbb{R}^{d_{model}}$ |
| **态射** | 每层的参数矩阵 $W_Q, W_K, W_V, W_O, W_1, W_2$ |
| **恒等态射** | 残差连接（$x \mapsto x$） |
| **态射复合** | 跨层矩阵乘积，如 $W_O^\ell \cdot W_Q^{\ell'}$ |
| **分配律保证** | 残差流的加法结构 + 矩阵乘法对加法的线性 |

> 只要残差流维度统一 + 残差连接存在，任意两层的权重矩阵就可以通过矩阵乘法直接复合。复合成的新态射仍然是 $\mathbb{R}^{d_{model}}$ 上的线性映射。
