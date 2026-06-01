"""PyTorch LLaMA-3.1-8B MRPC baseline (Phase 1).

Zero-shot prompt-classification: compare logit("Yes") vs logit("No") at the
answer position. Predict label=1 (paraphrase) if Yes-logit > No-logit.
"""
import argparse
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "NousResearch/Meta-Llama-3.1-8B"

PROMPT = (
    "Are these two sentences paraphrases of each other?\n"
    "Sentence 1: {s1}\n"
    "Sentence 2: {s2}\n"
    "Answer (Yes or No):"
)


def load_yes_no_token_ids(tokenizer):
    yes_id = tokenizer(" Yes", add_special_tokens=False).input_ids
    no_id  = tokenizer(" No",  add_special_tokens=False).input_ids
    assert len(yes_id) == 1 and len(no_id) == 1, (yes_id, no_id)
    return yes_id[0], no_id[0]


def estimate_prior(model, tokenizer, yes_id, no_id, device="cuda:0"):
    """Contextual calibration (Zhao et al. 2021): probe the model with
    content-free placeholders to estimate P(Yes) / P(No) priors, then
    subtract from each example's logits."""
    nulls = ["", "N/A", "[MASK]"]
    biases = []
    with torch.no_grad():
        for null in nulls:
            prompt = PROMPT.format(s1=null, s2=null)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = model(**inputs)
            last = out.logits[0, -1]
            biases.append((float(last[yes_id].item()),
                           float(last[no_id].item())))
    yes_bias = float(np.mean([b[0] for b in biases]))
    no_bias  = float(np.mean([b[1] for b in biases]))
    print(f"  contextual calibration over {len(nulls)} null prompts:")
    for null, (yb, nb) in zip(nulls, biases):
        print(f"    null={null!r:>8s}  yes={yb:.3f}  no={nb:.3f}")
    print(f"    avg yes_bias={yes_bias:.3f}  no_bias={no_bias:.3f}  "
          f"diff={yes_bias-no_bias:+.3f}")
    return yes_bias, no_bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="if set, score only the first N dev examples")
    ap.add_argument("--split", default="validation",
                    choices=["validation", "train", "test"])
    args = ap.parse_args()

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    yes_id, no_id = load_yes_no_token_ids(tokenizer)
    print(f"  yes_id={yes_id} no_id={no_id}")

    print(f"Loading model (fp16) onto cuda:0...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0")
    model.eval()
    print(f"  model loaded in {time.time()-t0:.1f}s")

    print(f"Estimating contextual-calibration prior...")
    yes_bias, no_bias = estimate_prior(model, tokenizer, yes_id, no_id)

    print(f"Loading MRPC ({args.split} split)...")
    ds = load_dataset("nyu-mll/glue", "mrpc")[args.split]
    if args.limit:
        ds = ds.select(range(args.limit))
    print(f"  {len(ds)} examples; label distribution: "
          f"label=1 (para): {sum(1 for r in ds if r['label']==1)}, "
          f"label=0 (not):  {sum(1 for r in ds if r['label']==0)}")

    n_correct_raw = n_correct_cal = 0
    n_total = 0
    confusion_raw = np.zeros((2, 2), dtype=np.int64)
    confusion_cal = np.zeros((2, 2), dtype=np.int64)
    yes_logits, no_logits = [], []
    t_start = time.time()

    with torch.no_grad():
        for i, row in enumerate(ds):
            prompt = PROMPT.format(s1=row["sentence1"], s2=row["sentence2"])
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            out = model(**inputs)
            last_logits = out.logits[0, -1]  # next-token logits at end of prompt
            yl = float(last_logits[yes_id].item())
            nl = float(last_logits[no_id].item())
            true = int(row["label"])
            pred_raw = 1 if yl > nl else 0
            pred_cal = 1 if (yl - yes_bias) > (nl - no_bias) else 0
            confusion_raw[true][pred_raw] += 1
            confusion_cal[true][pred_cal] += 1
            n_correct_raw += (pred_raw == true)
            n_correct_cal += (pred_cal == true)
            n_total += 1
            yes_logits.append(yl)
            no_logits.append(nl)
            if (i + 1) % 50 == 0:
                acc_raw = n_correct_raw / n_total
                acc_cal = n_correct_cal / n_total
                rate = n_total / (time.time() - t_start)
                print(f"  [{i+1:4d}/{len(ds)}] raw_acc={acc_raw:.4f} "
                      f"cal_acc={acc_cal:.4f}  rate={rate:.2f} ex/s")

    elapsed = time.time() - t_start

    def _report(name, n_correct, confusion):
        acc = n_correct / n_total
        tp, fp = confusion[1][1], confusion[0][1]
        fn = confusion[1][0]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        print(f"  --- {name} ---")
        print(f"    accuracy: {acc:.4f} ({n_correct}/{n_total})")
        print(f"    precision (1): {prec:.4f}  recall (1): {rec:.4f}  F1 (1): {f1:.4f}")
        print(f"    confusion[true][pred]:")
        print(f"      label=0:  pred=0 {confusion[0][0]:4d}  pred=1 {confusion[0][1]:4d}")
        print(f"      label=1:  pred=0 {confusion[1][0]:4d}  pred=1 {confusion[1][1]:4d}")

    print(f"\n=== MRPC {args.split} (zero-shot prompt, LLaMA-3.1-8B base) ===")
    print(f"  total examples: {n_total}")
    print(f"  yes_bias={yes_bias:.3f}  no_bias={no_bias:.3f}  diff={yes_bias-no_bias:+.3f}")
    print(f"  Yes-logit  mean={np.mean(yes_logits):.3f} std={np.std(yes_logits):.3f}")
    print(f"  No-logit   mean={np.mean(no_logits):.3f} std={np.std(no_logits):.3f}")
    _report("Raw (no calibration)", n_correct_raw, confusion_raw)
    _report("Contextual calibration", n_correct_cal, confusion_cal)
    print(f"  total time: {elapsed:.1f}s  rate={n_total/elapsed:.2f} ex/s")


if __name__ == "__main__":
    main()
