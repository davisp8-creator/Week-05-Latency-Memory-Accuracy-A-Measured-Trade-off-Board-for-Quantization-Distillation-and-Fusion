# Running the notebook on a Colab T4 GPU

This produces the graded results, tagged `t4gpu`, and at the end offers to write an **Intel-VM vs Colab-T4
comparison** into `README.md` and push it to GitHub with a fine-grained token. The VM's CPU results
(`results/*_cpu.*`) are never overwritten; they become the left-hand column of the comparison, and the
Excel workbook (`results/Week05_Benchmark_Workbook.xlsx`) is rebuilt with both machines in it.

## 1. One-time: create a fine-grained token

GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** → Generate new token:

- **Repository access:** *Only select repositories* → this Week-05 repo
- **Permissions:** Repository permissions → **Contents: Read and write** (nothing else)
- **Expiration:** short (7–30 days); revoke it when the assignment is done

Optional, recommended: in Colab open the 🔑 **Secrets** panel, add a secret named `GITHUB_TOKEN` with the token
as its value, and turn on *Notebook access*. The notebook then uses it automatically and never asks you to paste it.
Without the secret, you get a hidden prompt at the end (the token isn't echoed or saved).

## 2. Runtime and notebook

**Runtime → Change runtime type → T4 GPU → Save.** Upload `Assignment_05_Tradeoff_Board_Colab.ipynb` (File → Upload notebook,
or open it from GitHub). The notebook's first code cell, **Colab Bootstrap**, clones this repo once and changes into it, so
there is nothing to type. It is safe to re-run: when `env_setup.py` already exists in the working directory it does nothing,
so **Run all** never clones twice or nests folders.

Results land in the clone's `results/` folder, which is what makes the end-of-run push possible, and the workbook step
imports `scripts/build_workbook.py` from the clone. Do not add your own `!git clone` / `%cd` cell: an unguarded one clones
again on every Run all and descends one folder deeper each time. If that already happened, use
**Runtime → Disconnect and delete runtime**, then reconnect and Run all.

## 3. Run all

**Runtime → Run all.** Step 1 should print:

```text
[setup] host=colab-t4 (Colab (T4 GPU))  platform=colab  python=3.12.x  torch=2.x.x+cu12x  device=cuda  gpu=Tesla T4 (14.7 GB, sm_75)  results tag='t4gpu'
```

Check for **`host=colab-t4`**, **`gpu=Tesla T4`** and **`results tag='t4gpu'`**. Step 8 also prints how much
GPU memory is allocated before the memory pass and how much of it belongs to tensors Python can still reach. A
leftover of roughly 10-20 MB with almost no reachable tensors is library workspace (cuBLAS), not a leak; the board
reports each config's peak above its own floor, so it is excluded. A large reachable-tensor figure means a model or
dataset stayed on the GPU and needs parking before the pass. If you see:

| Message | Meaning | Fix |
|---|---|---|
| `Colab is on a CPU-only runtime` | Step 2 was skipped | Change runtime type to T4 GPU, then Run all again |
| `nvidia-smi sees a GPU ... but torch cannot use it` | A CPU torch was installed over Colab's CUDA build | Runtime → Disconnect and delete runtime, then run again without installing anything |
| `GPU is <name>, not a T4` | Colab assigned a different GPU | Fine; results are tagged `gpu` so they aren't mistaken for T4 numbers |
| `not running inside a git clone` | The Colab Bootstrap cell was deleted or skipped | Restore it (rebuild with `python scripts/build_notebook.py`), or Disconnect and delete runtime, then Run all |

Don't `pip install -r requirements.txt` on Colab. It pins the VM's CPU torch and would hide the T4 from torch.

Expect about 15–25 minutes in total (training dominates; latency timing is a few minutes).

## 4. The end-of-run prompts (Step 14)

```text
Write these results into README.md (Intel VM vs Colab T4 comparison)? [y/N] y
[publish] README.md updated (results section only)
Commit and push README.md + results/*_t4gpu* to GitHub? [y/N] y
[publish] GitHub token (hidden; blank = use existing git credentials): ••••••••
[publish] pulling origin/main ...
[publish] pushing to origin/main ...
[publish] done: README.md and N result path(s) pushed to origin/main
```

What the push does, in order:

1. Commits `results_t4gpu.json`, `manifest_t4gpu.json`, the two `*_t4gpu.csv` run files, the workbook and `figs_workbook/`.
2. Pulls `origin/main` with a rebase. If the VM pushed a newer CPU run after you cloned, it comes in here.
3. Re-renders the README results section from **all** results now in the clone (VM + this T4 run) and commits it.
4. Pushes.

The token is sent per command as an HTTP auth header. It is never printed, never written to `.git/config`, never
added to the remote URL, and never saved in the notebook. Answer **N** to either question to skip; nothing is
pushed without your second yes.

> The workbook pushed from Colab is built from the results in the clone at that moment. If the VM's `*_cpu` results are
> already committed, the workbook contains both machines and the `VM_vs_GPU` sheet is populated.

## 5. If you skipped the push

- **Later from Colab** (same session): `!python scripts/update_readme.py --push`
- **From the VM:** download the zip from Step 13, unzip it into `results/`, then
  `python scripts/build_workbook.py` and `python scripts/update_readme.py --push --tag t4gpu`.
  It uses the VM's normal git credentials if you leave the token prompt blank.
- **Re-render only, no git:** `python scripts/update_readme.py`

## What the comparison shows

| Table | Contents |
|---|---|
| Hardware | Device (CPU model / VM product vs T4 + VRAM + compute capability), host class, platform, torch build, total runtime |
| Board | Accuracy (mean ± seed std), p50 / p95 latency, model size and peak GPU memory, for every config on every machine |
| Workbook `VM_vs_GPU` | The same config on both machines: accuracy difference (should be ~0) and the latency speed-up |

These models are tiny, so a T4 will **not** always beat a modern CPU on batch-size-1 latency: kernel-launch overhead
dominates. That is a finding about *this* workload, not a bug.
