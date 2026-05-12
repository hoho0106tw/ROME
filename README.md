# ROME Sequential Feedback Editing 說明

本次實驗的目標，是針對模型上一次推論錯誤的輸入樣本進行權重修正。

模型原本在輸入 `x` 時，輸出錯誤答案：

```text
Pred = cervical spondylosis
```

但正確答案，也就是本次 feedback editing 的 target，是：

```text
t = radiculopathy
```

因此，本次更新的目的不是讓模型學習錯誤答案，而是讓模型學習正確 target：

```text
target = radiculopathy
```

---

## Before Feedback

```text
GT:      radiculopathy
Pred:    cervical spondylosis
Correct: NO
```

也就是說，更新前模型的輸出可以表示為：

$$
f_{\theta_{old}}(x) = \text{cervical spondylosis}
$$

但我們希望模型輸出的正確 target 是：

$$
t = \text{radiculopathy}
$$

因此更新前模型是錯誤的：

$$
f_{\theta_{old}}(x) \neq t
$$

---

## Target 定義

本次 editing 的目標答案為：

$$
t = \text{radiculopathy}
$$

我們希望更新權重後，模型對同一個輸入 `x` 的輸出變成：

$$
f_{\theta_{new}}(x) = t
$$

也就是：

$$
f_{\theta_{new}}(x) = \text{radiculopathy}
$$

---

## Loss Function

為了讓模型更傾向輸出正確 target `t`，使用 negative log-likelihood loss：

$$
\mathcal{L}(\theta) = -\log P_{\theta}(t \mid x)
$$

其中：

$$
P_{\theta}(t \mid x)
$$

代表模型在輸入 `x` 時，輸出 target `t` 的機率。

在本次實驗中：

$$
t = \text{radiculopathy}
$$

所以 loss 可以理解為：

$$
\mathcal{L}(\theta) = -\log P_{\theta}(\text{radiculopathy} \mid x)
$$

當 loss 越小，代表模型越傾向輸出：

```text
radiculopathy
```

---

## 基本反向傳播與權重更新

模型會先計算 loss：

$$
\mathcal{L}(\theta)
$$

接著用微分計算 loss 對權重的梯度：

$$
\nabla_{\theta}\mathcal{L}(\theta)
$$

這個梯度代表：

```text
如果要讓 loss 下降，權重應該往哪個方向修改
```

因此使用梯度下降更新權重：

$$
\theta_{new}
=
\theta_{old}
-
\eta \nabla_{\theta}\mathcal{L}(\theta_{old})
$$

其中：

$$
\theta_{old}
$$

是更新前的模型權重。

$$
\theta_{new}
$$

是更新後的模型權重。

$$
\eta
$$

是 learning rate。

$$
\nabla_{\theta}\mathcal{L}(\theta_{old})
$$

是 loss 對模型權重的梯度。

---

## 加入 Mask 的權重更新

本次更新有使用 allowed token mask，因此只保留允許更新的梯度部分。

權重更新可寫成：

$$
\theta_{new}
=
\theta_{old}
-
\eta \, \mathrm{Mask}
\left(
\nabla_{\theta}\mathcal{L}(\theta_{old})
\right)
$$

也就是：

```text
先計算 loss
再反向傳播得到梯度
接著用 Mask 保留需要的梯度
最後更新模型權重
```

權重變化量可以表示為：

$$
\Delta \theta
=
\theta_{new} - \theta_{old}
$$

因此：

$$
\Delta \theta
=
-
\eta \, \mathrm{Mask}
\left(
\nabla_{\theta}\mathcal{L}(\theta_{old})
\right)
$$

log 中的 `update norm` 可以理解為：

$$
\|\Delta \theta\|
$$

也就是每一步權重實際被修改的幅度。

---

## Loss 收斂結果

本次 feedback editing 的 loss 從：

$$
2.705394
$$

下降到：

$$
0.000008
$$

可表示為：

$$
2.705394 \rightarrow 0.000008
$$

loss 過程如下：

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

$$
\mathcal{L}(\theta) = -\log P_{\theta}(t \mid x)
$$

所以當 loss 下降時，代表：

$$
P_{\theta}(t \mid x)
$$

正在上升。

也就是模型越來越傾向輸出正確 target：

```text
radiculopathy
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

$$
f_{\theta_{new}}(x) = \text{radiculopathy}
$$

因此模型從原本的錯誤輸出：

```text
cervical spondylosis
```

被修正為正確 target：

```text
radiculopathy
```

---

## 總結

本次方法可以簡化為：

```text
更新前：
x → cervical spondylosis

正確 target：
t = radiculopathy

loss：
L(θ) = -log Pθ(t | x)

權重更新：
θ_new = θ_old - η Mask(∇θ L(θ_old))

更新後：
x → radiculopathy
```

其中最重要的是：

$$
t = \text{radiculopathy}
$$

也就是本次 feedback editing 的 target 是正確答案 `radiculopathy`。

最終 loss 從：

$$
2.705394
$$

下降到：

$$
0.000008
$$

表示模型對 target `radiculopathy` 的生成機率大幅提高，權重修正成功。
