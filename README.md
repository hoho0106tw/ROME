# ROME Sequential Feedback Editing 說明

本次實驗的目標，是針對模型上一次推論錯誤的輸入樣本進行權重修正。  
模型原本在輸入 `x` 時輸出錯誤答案：

```text
ŷ_old = cervical spondylosis
```

但該樣本的正確答案，也就是本次 editing 的 target，應該是：

```text
t = radiculopathy
```

因此，本次方法不是讓模型學習錯誤答案，而是使用正確答案 `radiculopathy` 作為最佳化目標，透過 loss 反向傳播更新模型權重，使模型下次對相同輸入能輸出正確答案。

---

## Before Feedback

```text
GT:      radiculopathy
Pred:    cervical spondylosis
Correct: NO
```

更新前模型可表示為：

```math
f_{\theta_{old}}(x) = \hat{y}_{old}
```

其中：

```math
\hat{y}_{old} = \text{cervical spondylosis}
```

但正確答案為：

```math
t = \text{radiculopathy}
```

因此：

```math
f_{\theta_{old}}(x) = \hat{y}_{old} \neq t
```

---

## Target 定義

本次 editing 的 target 是正確答案：

```math
t = \text{radiculopathy}
```

模型更新的目標是讓：

```math
f_{\theta_{new}}(x) = t
```

也就是：

```math
f_{\theta_{new}}(x) = \text{radiculopathy}
```

---

## Loss Function

為了提高模型在輸入 `x` 時產生 target `t` 的機率，定義 negative log-likelihood loss：

```math
\mathcal{L}(\theta)
=
-\log P_{\theta}(t \mid x)
```

其中：

```math
P_{\theta}(t \mid x)
```

代表模型在參數 `θ` 下，對輸入 `x` 產生正確答案 `t` 的條件機率。

若 target `t` 被 tokenizer 拆成多個 token：

```math
t = (t_1, t_2, \dots, t_n)
```

則 loss 可展開為：

```math
\mathcal{L}(\theta)
=
-\sum_{i=1}^{n}
\log P_{\theta}(t_i \mid x, t_{<i})
```

其中：

```math
t_{<i} = (t_1, t_2, \dots, t_{i-1})
```

這表示模型要在每一個生成位置上，提高正確 target token 的機率。

---

## Weight Update

權重更新使用 masked gradient descent。  
先計算 target loss：

```math
\mathcal{L}(\theta_{old})
=
-\log P_{\theta_{old}}(t \mid x)
```

再根據 loss 對模型權重計算梯度：

```math
\nabla_{\theta}\mathcal{L}(\theta_{old})
```

由於本次只希望保留 allowed update token ids 對應的梯度分量，因此加入 mask：

```math
\theta_{new}
=
\theta_{old}
-
\eta \,
\mathrm{Mask}
\left(
\nabla_{\theta}
\mathcal{L}(\theta_{old})
\right)
```

其中：

```math
\eta
```

是 learning rate。

```math
\mathrm{Mask}(\cdot)
```

代表只保留允許更新的梯度分量，降低非目標 token 對權重更新的影響。

權重實際變化量為：

```math
\Delta \theta
=
\theta_{new} - \theta_{old}
```

因此：

```math
\Delta \theta
=
-
\eta \,
\mathrm{Mask}
\left(
\nabla_{\theta}
\mathcal{L}(\theta_{old})
\right)
```

每一步 log 中的 `update norm` 可以理解為：

```math
\|\Delta \theta\|
```

也就是該步驟模型權重實際被修改的幅度。

---

## Loss 收斂結果

本次 feedback editing 共進行 30 steps。  
loss 從一開始的：

```math
\mathcal{L}(\theta_0) = 2.705394
```

下降到最後的：

```math
\mathcal{L}(\theta_{30}) = 0.000008
```

可表示為：

```math
2.705394 \rightarrow 0.000008
```

loss 收斂過程如下：

```text
Step  1 | loss = 2.705394
Step  2 | loss = 2.517541
Step  3 | loss = 2.329761
Step  4 | loss = 2.142031
Step  5 | loss = 1.954342
Step  6 | loss = 1.766719
Step  7 | loss = 1.579223
Step  8 | loss = 1.392003
Step  9 | loss = 1.205338
Step 10 | loss = 1.019844
Step 11 | loss = 0.836810
Step 12 | loss = 0.658733
Step 13 | loss = 0.490302
Step 14 | loss = 0.339123
Step 15 | loss = 0.214707
Step 16 | loss = 0.123979
Step 17 | loss = 0.066164
Step 18 | loss = 0.033412
Step 19 | loss = 0.016326
Step 20 | loss = 0.007838
Step 21 | loss = 0.003730
Step 22 | loss = 0.001767
Step 23 | loss = 0.000836
Step 24 | loss = 0.000395
Step 25 | loss = 0.000186
Step 26 | loss = 0.000090
Step 27 | loss = 0.000045
Step 28 | loss = 0.000024
Step 29 | loss = 0.000013
Step 30 | loss = 0.000008
```

因為 loss 定義為：

```math
\mathcal{L}(\theta)
=
-\log P_{\theta}(t \mid x)
```

所以當 loss 持續下降時，代表：

```math
P_{\theta}(t \mid x)
```

正在提高。

當最後：

```math
\mathcal{L}(\theta_{30}) \approx 0
```

表示：

```math
P_{\theta_{30}}(t \mid x) \approx 1
```

也就是模型在更新後，已經高度傾向於對輸入 `x` 輸出 target：

```math
t = \text{radiculopathy}
```

---

## After Feedback

更新後模型輸出為：

```text
GT:      radiculopathy
Pred:    radiculopathy
Correct: YES
```

也就是：

```math
f_{\theta_{new}}(x) = t
```

更具體地：

```math
f_{\theta_{new}}(x)
=
\text{radiculopathy}
```

---

## 總結

本次 editing 可以簡化表示為：

```text
更新前：
x → cervical spondylosis

target：
t = radiculopathy

loss：
L(θ) = -log Pθ(t | x)

權重更新：
θ_new = θ_old - η Mask(∇θ L(θ_old))

更新後：
x → radiculopathy
```

因此，本次方法的核心是：

```math
\hat{y}_{old}
=
\text{cervical spondylosis}
\neq
t
=
\text{radiculopathy}
```

透過最小化：

```math
-\log P_{\theta}(t \mid x)
```

並更新權重：

```math
\theta_{new}
=
\theta_{old}
-
\eta \,
\mathrm{Mask}
\left(
\nabla_{\theta}
\left[
-\log P_{\theta_{old}}(t \mid x)
\right]
\right)
```

最終讓模型由：

```math
f_{\theta_{old}}(x)
=
\text{cervical spondylosis}
```

修正為：

```math
f_{\theta_{new}}(x)
=
\text{radiculopathy}
```

loss 從：

```math
2.705394
```

收斂到：

```math
0.000008
```

表示模型對正確 target `radiculopathy` 的生成機率已經大幅提升，權重修正成功。
