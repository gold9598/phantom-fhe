"""Pre-capture all 408 MRPC PyTorch reference activations in a STANDALONE
process (no FHE engine present), so the PT-ref model and the CKKS engine never
coexist on the 32GB GPU (that collision is what OOM'd the sweep after the reboot
wiped the /tmp PT-ref cache). Mirrors mrpc_sweep.py's _ptref_cached exactly.
Caches to /tmp (where mrpc_sweep reads) AND to a PERSISTENT backup so a future
reboot can restore without re-capturing.
"""
import os, sys, shutil
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/build/lib")
sys.path.insert(0, "/home/yongwoo-oh/phantom-fhe/python/llm_project")
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from helpers.pytorch_ref import capture_pytorch_ref

PERSIST = "/home/yongwoo-oh/mrpc_campaign/ptref"
os.makedirs(PERSIST, exist_ok=True)
PROMPT_FMT = ("Are these two sentences paraphrases of each other?\n"
              "Sentence 1: {s1}\nSentence 2: {s2}\nAnswer (Yes or No):")

tok = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3.1-8B")
ds = load_dataset("nyu-mll/glue", "mrpc")["validation"]
print(f"pre-capturing 408 PT-refs (standalone, no FHE engine)...", flush=True)

done = 0
for idx in range(408):
    row = ds[idx]
    tids = tok(PROMPT_FMT.format(s1=row["sentence1"], s2=row["sentence2"])).input_ids
    fn = f"mrpc_ptref_idx{idx}_n{len(tids)}.npz"
    tmp, per = f"/tmp/{fn}", f"{PERSIST}/{fn}"
    if os.path.exists(per):                       # persistent backup exists
        if not os.path.exists(tmp):
            shutil.copy(per, tmp)                 # restore to /tmp for the sweep
        done += 1
        continue
    ref, prenorm, yes, no = capture_pytorch_ref(tids)
    np.savez(tmp, ref=ref, prenorm=prenorm, yes=np.float64(yes), no=np.float64(no))
    shutil.copy(tmp, per)                          # persist backup
    done += 1
    if idx % 20 == 0 or idx == 407:
        print(f"  {done}/408 (idx={idx} n={len(tids)})", flush=True)

print(f"DONE: {done}/408 PT-refs in /tmp + {PERSIST}", flush=True)
