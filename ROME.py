#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import os
from datetime import datetime

# =========================
# config
# =========================
MODEL_NAME = "m42-health/Llama3-Med42-8B"

OUTPUT_MODEL_BASE = "/workspace/med42_awq_project/models/med42-8b-feedback-radiculopathy-fp32-lmhead"
OUTPUT_MODEL = OUTPUT_MODEL_BASE + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_LABEL = "radiculopathy"
WRONG_LABEL = "stroke"

FEEDBACK_STEPS = 20
FEEDBACK_LR = 1e-4

# =========================
# single SOAP data
# =========================
SOAP = {
    "S": """
20220112 for 2nd opinion of MRI reports from MMH.
20170529 still toes numbness.
20170501 improved hand numbness.
20170410 first visit:
Palpation of Rt posterior neck swelling mass.
Bil. hand redness and numbness.
Also bil. leg numbness, Rt peroneal territory.
P.H. nil.
Allergy: NKA.
Occupation: house chore.
""",

    "O": """
20191224 brain and C spine MRI at MMH:
A small old lacune at anterior limb of right internal capsule.
Marginal spur and mild thecal compression.

Consciousness: clear.
Pupil: 3/3 mm, LR +/+.
VF: intact.
EOM: no limitation.
No facial palsy.
No tongue deviation.
M.P. 5/5.
DTR: ++/++.
Sensory: symmetric.
FNF: no dysmetria.
Gait: intact.
Romberg sign: negative.

Lab:
HbA1c 5.6%, AC 91.

NCV WNL.
SSEP poor bil. P1.

X-ray:
Mild cervical spondylosis, with compromising right neural foramen between C5/C6.
""",

    "A": """
Numbness of upper limbs.
""",

    "P": """
Encourage stroke prevention.
""",

    "PRIMARY_DIAGNOSIS": "radiculopathy"
}

# =========================
# utils
# =========================
def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def normalize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


def match_label(pred):
    """
    已移除 LABELS。
    只判斷 target label / wrong label。
    其他輸出原樣 normalize 後回傳，方便 debug。
    """

    pred = normalize(pred)

    if TARGET_LABEL in pred:
        return TARGET_LABEL

    if WRONG_LABEL in pred:
        return WRONG_LABEL

    return pred if pred else "unknown"


# =========================
# prompt
# =========================
def build_prompt(row):
    parts = []

    for col in ["S", "O", "A", "P"]:
        val = str(row[col]).strip()
        if val and val != "nan":
            parts.append(f"{col}: {val}")

    text = " ".join(parts)

    prompt = f"""
You are a clinical diagnosis classifier.

Task:
Given a SOAP note, output the primary diagnosis.

Rules:
- Answer with EXACTLY one diagnosis label
- Do NOT explain
- Do NOT output anything else

SOAP:
{text}

Answer:
"""

    return prompt.strip()


# =========================
# inference
# =========================
def infer(model, tokenizer, prompt):
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True)

    if "Answer:" in text:
        text = text.split("Answer:")[-1]

    return text.strip()


# =========================
# get lm_head
# =========================
def get_lm_head(model):
    if hasattr(model, "get_output_embeddings"):
        lm_head = model.get_output_embeddings()
        if lm_head is not None:
            return lm_head

    if hasattr(model, "lm_head"):
        return model.lm_head

    raise ValueError("Cannot find lm_head / output embeddings.")


# =========================
# target feedback update
# =========================
def target_feedback_update(
    model,
    tokenizer,
    prompt,
    target_label,
    wrong_label="stroke",
    steps=20,
    lr=1e-4
):
    """
    Target feedback:
    - 整個模型使用 FP32
    - 只更新 lm_head
    - 只允許 target_label + wrong_label token rows 更新
    - loss / gradient NaN 時 rollback
    """

    print("\n===== TARGET FEEDBACK FP32 VERSION =====")
    print("Target label:", target_label)
    print("Wrong label :", wrong_label)

    # 凍結全部參數
    for param in model.parameters():
        param.requires_grad = False

    lm_head = get_lm_head(model)

    # 只轉 dtype，不強制移動 device
    lm_head.to(torch.float32)

    for param in lm_head.parameters():
        param.requires_grad = True

    backup_weight = lm_head.weight.detach().clone()

    trainable_params = [p for p in lm_head.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=0.0,
        eps=1e-8
    )

    model.train()

    prompt_text = prompt.rstrip()
    target_text = " " + target_label.strip()

    prompt_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False
    ).input_ids.to(DEVICE)

    full_text = prompt_text + target_text

    full_inputs = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False
    ).to(DEVICE)

    input_ids = full_inputs["input_ids"]
    attention_mask = full_inputs["attention_mask"]

    labels = input_ids.clone()

    prompt_len = prompt_ids.shape[1]
    labels[:, :prompt_len] = -100

    target_token_ids = tokenizer(
        target_text,
        add_special_tokens=False
    ).input_ids

    wrong_text = " " + wrong_label.strip()
    wrong_token_ids = tokenizer(
        wrong_text,
        add_special_tokens=False
    ).input_ids

    allowed_token_ids = list(dict.fromkeys(target_token_ids + wrong_token_ids))

    print("Prompt tokens:", prompt_len)
    print("Total tokens :", input_ids.shape[1])
    print("Target tokens:", input_ids.shape[1] - prompt_len)

    print("\nTarget token ids:")
    for tid in target_token_ids:
        print("TARGET:", tid, "=>", tokenizer.decode([tid]))

    print("\nWrong token ids:")
    for tid in wrong_token_ids:
        print("WRONG :", tid, "=>", tokenizer.decode([tid]))

    print("\nAllowed update token ids:", allowed_token_ids)

    success = True

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        before_weight = lm_head.weight.detach().clone()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )

        loss = outputs.loss

        if not torch.isfinite(loss):
            print(f"Step {step + 1}/{steps} | loss = {loss.item()}")
            print("Loss became NaN/Inf. Rollback lm_head.")
            success = False
            break

        loss.backward()

        grad_norm_before = 0.0
        grad_norm_after = 0.0

        if hasattr(lm_head, "weight") and lm_head.weight.grad is not None:
            grad = lm_head.weight.grad

            if not torch.isfinite(grad).all():
                print("Gradient became NaN/Inf. Rollback lm_head.")
                success = False
                break

            grad_norm_before = grad.norm().item()

            mask = torch.zeros(
                grad.shape[0],
                device=grad.device,
                dtype=torch.bool
            )

            for tid in allowed_token_ids:
                if tid < grad.shape[0]:
                    mask[tid] = True
                else:
                    print(f"Warning: token id {tid} is out of vocab range.")

            # 只保留 target / wrong token rows 的 gradient
            grad[~mask, :] = 0

            grad_norm_after = grad.norm().item()

        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=0.05
        )

        optimizer.step()

        with torch.no_grad():
            update_norm = (lm_head.weight.detach() - before_weight).norm().item()

        print(
            f"Step {step + 1}/{steps} | "
            f"loss = {loss.item():.6f} | "
            f"grad before mask = {grad_norm_before:.6f} | "
            f"grad after mask = {grad_norm_after:.6f} | "
            f"update norm = {update_norm:.8f}"
        )

    if not success:
        with torch.no_grad():
            lm_head.weight.copy_(backup_weight)
        print("lm_head rollback done.")

    model.eval()

    if success:
        print("===== TARGET FEEDBACK DONE =====\n")
    else:
        print("===== TARGET FEEDBACK FAILED, MODEL RESTORED =====\n")

    return success


# =========================
# evaluate single case
# =========================
def evaluate_single(model, tokenizer, row, title):
    gt = normalize(str(row["PRIMARY_DIAGNOSIS"]))
    prompt = build_prompt(row)

    pred_raw = infer(model, tokenizer, prompt)
    pred = match_label(pred_raw)

    print(f"\n===== {title} =====")
    print("GT:", gt)
    print("Pred:", pred)
    print("Raw:", pred_raw)

    if pred == gt:
        print("Correct: YES")
    else:
        print("Correct: NO")

    return pred, pred_raw


# =========================
# save model
# =========================
def save_feedback_model(model, tokenizer, output_dir):
    print("\n===== SAVING FEEDBACK MODEL =====")
    print("Save path:", output_dir)

    os.makedirs(output_dir, exist_ok=False)

    print("Saving model weights...")

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="2GB"
    )

    print("Saving tokenizer...")

    tokenizer.save_pretrained(output_dir)

    print("\nSaved feedback model to:", output_dir)

    print("\nFiles in saved directory:")
    files = sorted(os.listdir(output_dir))

    for f in files:
        path = os.path.join(output_dir, f)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f" - {f} ({size_mb:.2f} MB)")

    has_weight_file = any(
        f.endswith(".safetensors") or f == "pytorch_model.bin"
        for f in files
    )

    has_index_file = any(
        f.endswith(".index.json")
        for f in files
    )

    has_tokenizer_file = any(
        f in ["tokenizer.json", "tokenizer.model"]
        for f in files
    )

    has_tokenizer_config = "tokenizer_config.json" in files

    if not (has_weight_file or has_index_file):
        raise RuntimeError(
            "No model weight file found after saving. "
            "Expected .safetensors, .index.json, or pytorch_model.bin."
        )

    if not has_tokenizer_file:
        raise RuntimeError(
            "No tokenizer file found after saving. "
            "Expected tokenizer.json or tokenizer.model."
        )

    if not has_tokenizer_config:
        raise RuntimeError(
            "No tokenizer_config.json found after saving."
        )

    print("\nReload testing tokenizer from saved directory...")

    _ = AutoTokenizer.from_pretrained(
        output_dir,
        use_fast=True,
        trust_remote_code=True
    )

    print("Tokenizer reload OK.")
    print("Weight files detected. Save looks OK.")
    print("===== SAVE DONE =====\n")


# =========================
# repair tokenizer only
# =========================
def repair_tokenizer_if_needed(base_model_name, output_dir):
    """
    如果某個已儲存的 feedback model 資料夾缺 tokenizer，
    可以用這個函數從 base model 補 tokenizer。
    """

    print("\n===== REPAIR TOKENIZER =====")
    print("Base tokenizer:", base_model_name)
    print("Target dir     :", output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        use_fast=True,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.save_pretrained(output_dir)

    print("Tokenizer repaired.")
    print("===== REPAIR DONE =====\n")


# =========================
# print SOAP
# =========================
def print_soap(row):
    print("===== SOAP INPUT =====")
    print("S:", row["S"])
    print("O:", row["O"])
    print("A:", row["A"])
    print("P:", row["P"])


# =========================
# main
# =========================
def main():
    tokenizer = None
    model = None

    try:
        print("Output model path:")
        print(OUTPUT_MODEL)

        print("\nLoading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast=True,
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float32

        print("Using dtype:", dtype)
        print("Device:", DEVICE)

        print("\nLoading model...")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        print_soap(SOAP)

        prompt = build_prompt(SOAP)

        # 1. feedback 前推論
        before_pred, before_raw = evaluate_single(
            model,
            tokenizer,
            SOAP,
            title="BEFORE FEEDBACK"
        )

        # 2. target feedback
        feedback_success = target_feedback_update(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            target_label=TARGET_LABEL,
            wrong_label=WRONG_LABEL,
            steps=FEEDBACK_STEPS,
            lr=FEEDBACK_LR
        )

        # 3. feedback 後推論
        after_pred, after_raw = evaluate_single(
            model,
            tokenizer,
            SOAP,
            title="AFTER FEEDBACK"
        )

        # 4. 只有真的變成 TARGET_LABEL 才存
        if feedback_success and after_pred == TARGET_LABEL:
            save_feedback_model(
                model=model,
                tokenizer=tokenizer,
                output_dir=OUTPUT_MODEL
            )
        else:
            print("\n===== DO NOT SAVE MODEL =====")
            print("feedback_success:", feedback_success)
            print("before_pred:", before_pred)
            print("before_raw:", before_raw)
            print("after_pred:", after_pred)
            print("after_raw:", after_raw)
            print("Target label:", TARGET_LABEL)
            print("Model was NOT saved because feedback did not produce target label.")

    finally:
        print("\nCleaning memory...")

        try:
            del model
        except Exception:
            pass

        try:
            del tokenizer
        except Exception:
            pass

        clean_memory()


# =========================
# run
# =========================
if __name__ == "__main__":
    main()


# In[ ]:




