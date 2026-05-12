# ROME Sequential Feedback Editing 說明

本次實驗的目標，是針對模型上一次推論錯誤的輸入樣本進行權重修正。

模型原本輸出錯誤答案：

```text
Pred = cervical spondylosis
```

但正確答案，也就是本次 feedback editing 的 target，是：

```text
target = radiculopathy
```

因此，本次方法不是讓模型學習錯誤答案，而是把正確答案 `radiculopathy` 當作 target，透過 loss 反向傳播更新模型權重，使模型下次對同一個輸入能輸出正確答案。

---

## Before Feedback

```text
GT:      radiculopathy
Pred:    cervical spondylosis
Correct: NO
```

更新前模型可以表示為：

```text
Model(x; θ_old) = cervical spondylosis
```

其中：

```text
x = 上一次輸入
θ_old = 更新前的模型權重
```

但本次真正希望模型輸出的 target 是：

```text
t = radiculopathy
```

所以更新前的狀態是：

```text
Model(x; θ_old) ≠ t
```

也就是模型答錯了。

---

## Target 定義

本次 editing 的 target 是正確答案：

```text
t = radiculopathy
```

我們希望權重更新後，模型變成：

```text
Model(x; θ_new) = radiculopathy
```

也就是：

```text
Model(x; θ_new) = t
```

其中：

```text
θ_new = 更新後的模型權重
```

---

## Loss Function

為了讓模型更傾向輸出 target `radiculopathy`，使用 negative log-likelihood loss。

簡單公式如下：

```text
L(θ) = -log Pθ(t | x)
```

其中：

```text
L(θ)
```

代表模型目前的 loss。

```text
Pθ(t | x)
```

代表模型在輸入 `x` 時，輸出 target `t` 的機率。

在本次實驗中：

```text
t = radiculopathy
```

所以 loss 可以寫成：

```text
L(θ) = -log Pθ(radiculopathy | x)
```

意思是：

```text
如果模型輸出 radiculopathy 的機率越高，loss 就越低。
如果模型輸出 radiculopathy 的機率越低，loss 就越高。
```

因此，降低 loss 的目的就是提高模型輸出正確 target 的機率。

---

## 權重更新公式

本次使用梯度下降更新模型權重。

簡單公式如下：

```text
θ_new = θ_old - η × gradient
```

其中：

```text
θ_old
```

代表更新前的模型權重。

```text
θ_new
```

代表更新後的模型權重。

```text
η
```

代表 learning rate，也就是每次更新的步伐大小。

```text
gradient
```

代表 loss 對模型權重的梯度，也就是模型應該往哪個方向修改權重，才能讓 loss 下降。

更完整地寫，可以表示為：

```text
θ_new = θ_old - η × ∇θ L(θ_old)
```

其中：

```text
∇θ L(θ_old)
```

代表在目前權重 `θ_old` 下，loss 對模型權重的梯度。

---

## 加入 Mask 的權重更新

本次實驗有使用 allowed token mask，因此不是所有梯度都直接拿來更新，而是只保留允許更新的部分。

公式可以簡化成：

```text
θ_new = θ_old - η × Mask(∇θ L(θ_old))
```

其中：

```text
Mask(...)
```

代表只保留 allowed update token ids 對應的梯度，避免不相關 token 對權重造成太多影響。

也就是說，本次權重更新流程可以理解為：

```text
1. 設定 target = radiculopathy
2. 計算 loss = -log Pθ(radiculopathy | x)
3. 反向傳播取得 gradient
4. 使用 Mask 保留允許更新的梯度
5. 更新模型權重 θ
```

---

## 權重變化量

每一步實際修改的權重差異可以表示為：

```text
Δθ = θ_new - θ_old
```

因為：

```text
θ_new = θ_old - η × Mask(∇θ L(θ_old))
```

所以：

```text
Δθ = -η × Mask(∇θ L(θ_old))
```

log 裡面的 `update norm` 可以理解為：

```text
||Δθ||
```

也就是每一步模型權重實際被修改的幅度。

---

## Loss 收斂結果

本次 feedback editing 共進行 30 steps。

loss 從：

```text
2.705394
```

下降到：

```text
0.000008
```

可以表示為：

```text
loss: 2.705394 → 0.000008
```

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

這表示模型對 target `radiculopathy` 的預測機率逐步提高。

因為 loss 是：

```text
L(θ) = -log Pθ(radiculopathy | x)
```

所以：

```text
loss 越大  → 模型越不確定 radiculopathy
loss 越小  → 模型越傾向輸出 radiculopathy
loss 接近 0 → 模型幾乎非常確定輸出 radiculopathy
```

本次 loss 從 `2.705394` 收斂到 `0.000008`，代表模型已經成功學到這次 target。

---

## After Feedback

更新後模型輸出：

```text
GT:      radiculopathy
Pred:    radiculopathy
Correct: YES
```

也就是：

```text
Model(x; θ_new) = radiculopathy
```

模型從原本的錯誤輸出：

```text
cervical spondylosis
```

被修正為正確 target：

```text
radiculopathy
```

---

## 總結

本次方法可以簡化成：

```text
更新前：
x → cervical spondylosis

正確 target：
t = radiculopathy

loss：
L(θ) = -log Pθ(t | x)

權重更新：
θ_new = θ_old - η × Mask(∇θ L(θ_old))

更新後：
x → radiculopathy
```

其中最重要的是：

```text
target = radiculopathy
```

而不是錯誤答案 `cervical spondylosis`。

最終 loss：

```text
2.705394 → 0.000008
```

表示模型對正確 target `radiculopathy` 的生成機率大幅提高，因此本次 feedback editing 的權重修正成功。
