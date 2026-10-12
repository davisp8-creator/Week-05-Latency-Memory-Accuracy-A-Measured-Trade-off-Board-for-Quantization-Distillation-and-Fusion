# Step 1: Environment Setup
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(ROOT))
from env_setup import setup, hardware_snapshot

ENV = setup(require_t4=False)  # set require_t4=True to refuse anything but a Colab T4
if not (ROOT / ".git").exists():
    print(
        "[setup] note: not running inside a git clone of the repo, so the last step cannot write "
        "README.md or push. On Colab: !git clone <repo>, %cd into it, then Run all (COLAB.md)."
    )


# Step 2: Imports, Single-Source Constants & Hardware-Tagged Output Paths
# Install required libraries: run the installer for your OS (or `python env_setup.py`);
# on Colab, Step 1 installs anything missing and leaves the CUDA torch build alone.

# Import necessary libraries
import copy
import gc
import io
import json
import math
import os
import platform
import random
import time

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

T_START = time.perf_counter()
IN_NOTEBOOK = "ipykernel" in sys.modules
QUICK = os.environ.get("BOARD_QUICK") == "1"  # smoke test only: results NOT meaningful
CFG = dict(
    n_train=800 if QUICK else 8000,
    n_test=400 if QUICK else 2000,
    epochs=2 if QUICK else 15,
    seeds=[0, 1] if QUICK else [0, 1, 2],
    trials=2 if QUICK else 5,
    n_iter=20 if QUICK else 300,
    n_warm=5 if QUICK else 50,
    mem_iters=5 if QUICK else 20,
    batch=64,
    lr=1e-3,
    data_seed=1234,
    kd_alpha=0.5,
    kd_T=2.0,
    cpu_threads=2,  # Colab has 2 vCPUs; pinning keeps the CPU rows comparable across machines
    img_noise=0.12,
    cue_depth=0.22,
    cue_noise_img=0.10,
    cue_noise_note=0.12,
)
RUN_TAG = ENV["tag"] + ("_quick" if QUICK else "")
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device(ENV["device"])
HAS_CUDA = DEVICE.type == "cuda"
torch.set_num_threads(CFG["cpu_threads"])
print(f"tag={RUN_TAG}  device={DEVICE}  host={ENV['host_class']}" + ("  [QUICK smoke-test mode]" if QUICK else ""))
print(CFG)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Step 3: Synthetic Evaluation Set (Generated Here, One Fixed Seed for Every Configuration)
ACCEPT = ["receiver accepted this load and signed", "consignee signed for this delivery no issues",
          "dock confirmed receipt of this freight", "receiving signed off on this shipment"]
REFUSE = ["receiver refused this load at the dock", "consignee rejected this delivery would not sign",
          "dock turned this freight away refused", "receiving denied this shipment and would not sign"]
DECOY = ["last week receiver refused a different load", "yesterday consignee rejected another delivery",
         "earlier dock accepted a different load", "before that receiving signed another shipment"]
FILLER = ["traffic was heavy on route", "weather clear all day", "arrived at gate around noon", "driver on hos break earlier",
          "trailer number checked at gate", "fuel stop at last exit", "waited in line at scale", "appointment was confirmed by phone"]
MAXLEN = 24
WORDS = sorted({w for s in ACCEPT + REFUSE + DECOY + FILLER for w in s.split()})
VOCAB = {w: i + 1 for i, w in enumerate(WORDS)}  # 0 = pad
CLASSES = [f"{c}/{o}" for c in ["intact", "crushed", "wet"] for o in ["accepted", "refused"]]


def make_image(rng, cue):
    img = 0.85 + CFG["img_noise"] * rng.standard_normal((64, 64))
    img[2:8, :] = 0.55                                                # header bar
    for r in range(12, 60, 6):                                        # printed text lines
        length, start = rng.integers(20, 56), rng.integers(2, 8)
        img[r:r + 2, start:start + length] = 0.35
    for _ in range(2):                                                # distractor stamps on every page
        y, x = rng.integers(10, 50, 2)
        img[y:y + 5, x:x + 5] = 0.5

    def band(depth):                                                  # diagonal scratch / crush line
        y, x = rng.integers(8, 40, 2)
        for k in range(14):
            img[y + k, x + k:x + k + 2] -= depth

    def blob(amp):                                                    # soft water stain
        yy, xx = np.mgrid[0:64, 0:64]
        cy, cx = rng.integers(8, 56, 2)
        img[:] -= amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 40.0)

    d = CFG["cue_depth"]
    if cue == 1:
        band(d)                                                       # crushed
    elif cue == 2:
        for _ in range(2):
            blob(d * 0.7)                                             # wet
    if cue != 1 and rng.random() < 0.5:
        band(d * 0.4)                                                 # decoy: faint scratch
    if cue != 2 and rng.random() < 0.5:
        blob(d * 0.3)                                                 # decoy: faint smudge
    return img.clip(0, 1).astype(np.float32)


def make_note(rng, outcome):
    parts = [(ACCEPT if outcome == 0 else REFUSE)[rng.integers(4)]]
    if rng.random() < 0.6:
        parts.append(DECOY[rng.integers(4)])
    parts += [FILLER[i] for i in rng.choice(8, rng.integers(0, 3), replace=False)]
    rng.shuffle(parts)
    ids = [VOCAB[w] for w in " ".join(parts).split()][:MAXLEN]
    return ids + [0] * (MAXLEN - len(ids))


def make_split(n, seed):
    rng = np.random.default_rng(seed)
    cond, outc = rng.integers(0, 3, n), rng.integers(0, 2, n)
    shown_c = np.where(rng.random(n) < CFG["cue_noise_img"], rng.integers(0, 3, n), cond)
    said_o = np.where(rng.random(n) < CFG["cue_noise_note"], 1 - outc, outc)
    imgs = np.stack([make_image(rng, c) for c in shown_c])[:, None]
    toks = np.array([make_note(rng, o) for o in said_o])
    return torch.tensor(imgs), torch.tensor(toks), torch.tensor(cond * 2 + outc)


Xtr = make_split(CFG["n_train"], CFG["data_seed"])
Xte = make_split(CFG["n_test"], CFG["data_seed"] + 1)
Xtr = [t.to(DEVICE) for t in Xtr]
print("train", tuple(Xtr[0].shape), "test", tuple(Xte[0].shape), "classes", len(CLASSES), "vocab", len(VOCAB) + 1)


# Step 4: Models (Written from Scratch so Every Variant Is Inspectable)
class Block(nn.Module):
    def __init__(s, d, h, attn):
        super().__init__()
        s.h, s.attn = h, attn
        s.qkv, s.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x, mask):  # mask: (B, N) bool, True = real token
        B, N, D = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, N, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        if s.attn == "softmax":
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        else:  # linear attention: cost grows linearly with sequence length
            q = F.elu(q) + 1
            k = (F.elu(k) + 1) * mask[:, None, :, None].to(q.dtype)
            kv = torch.einsum("bhnd,bhne->bhde", k, v)
            z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(2)) + 1e-6)
            a = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)
        x = x + s.o(a.transpose(1, 2).reshape(B, N, D))
        return x + s.ff(s.n2(x))


def pool(x, mask):
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / m.sum(1).clamp(min=1)


class TextEnc(nn.Module):
    def __init__(s, d, h, L, attn):
        super().__init__()
        s.emb = nn.Embedding(len(VOCAB) + 1, d, padding_idx=0)
        s.pos = nn.Parameter(torch.zeros(MAXLEN, d))
        s.blocks = nn.ModuleList([Block(d, h, attn) for _ in range(L)])

    def forward(s, tok):
        m = tok != 0
        x = s.emb(tok) + s.pos
        for b in s.blocks:
            x = b(x, m)
        return pool(x, m)


class ImgEnc(nn.Module):
    def __init__(s, d):
        super().__init__()
        c = max(8, d // 4)
        s.net = nn.Sequential(
            nn.Conv2d(1, c, 3, 2, 1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.Conv2d(c, 2 * c, 3, 2, 1), nn.BatchNorm2d(2 * c), nn.ReLU(),
            nn.Conv2d(2 * c, d, 3, 2, 1), nn.BatchNorm2d(d), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(s, img):
        return s.net(img)


class LateFusion(nn.Module):  # separate encoders, features concatenated before the head
    def __init__(s, d=64, h=4, L=2, attn="softmax"):
        super().__init__()
        s.t, s.i = TextEnc(d, h, L, attn), ImgEnc(d)
        s.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, len(CLASSES)))

    def forward(s, img, tok):
        return s.head(torch.cat([s.t(tok), s.i(img)], -1))


class EarlyFusion(nn.Module):  # image patches become tokens in the same transformer as the words
    def __init__(s, d=64, h=4, L=2):
        super().__init__()
        s.emb = nn.Embedding(len(VOCAB) + 1, d, padding_idx=0)
        s.proj = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, d))
        s.pos_t, s.pos_i = nn.Parameter(torch.zeros(MAXLEN, d)), nn.Parameter(torch.zeros(64, d))
        s.blocks = nn.ModuleList([Block(d, h, "softmax") for _ in range(L)])
        s.head = nn.Linear(d, len(CLASSES))

    def forward(s, img, tok):
        B = tok.shape[0]
        p = img.squeeze(1).unfold(1, 8, 8).unfold(2, 8, 8).reshape(B, 64, 64)
        x = torch.cat([s.proj(p) + s.pos_i, s.emb(tok) + s.pos_t], 1)
        m = torch.cat([torch.ones(B, 64, dtype=torch.bool, device=tok.device), tok != 0], 1)
        for b in s.blocks:
            x = b(x, m)
        return s.head(pool(x, m))


ARCH = {
    "late": lambda: LateFusion(64, 4, 2, "softmax"),
    "linear": lambda: LateFusion(64, 4, 2, "linear"),
    "early": lambda: EarlyFusion(64, 4, 2),
    "early2x": lambda: EarlyFusion(64, 4, 2),
    "student_kd": lambda: LateFusion(16, 2, 1, "softmax"),
    "student_nokd": lambda: LateFusion(16, 2, 1, "softmax"),
}
n_params = lambda m: sum(p.numel() for p in m.parameters())
for k, f in ARCH.items():
    print(f"{k:13s} params = {n_params(f()):,}")


# Step 5: Training Loop and Knowledge Distillation
@torch.no_grad()
def predict_acc(model, data, device, dtype=torch.float32):
    model.eval()
    img, tok, y = [t.to(device) for t in data]
    correct = 0
    for i in range(0, len(y), 256):
        out = model(img[i:i + 256].to(dtype), tok[i:i + 256])
        correct += (out.argmax(-1) == y[i:i + 256]).sum().item()
    return correct / len(y)


def train(name, seed, teacher=None, epochs=None):
    epochs = epochs or CFG["epochs"]
    set_seed(seed)
    m = ARCH[name]().to(DEVICE)
    img, tok, y = Xtr
    opt = torch.optim.AdamW(m.parameters(), lr=CFG["lr"], weight_decay=0.01)
    steps = epochs * math.ceil(len(y) / CFG["batch"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["lr"], total_steps=steps)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(y), generator=g).to(DEVICE)
        for i in range(0, len(y), CFG["batch"]):
            idx = perm[i:i + CFG["batch"]]
            out = m(img[idx], tok[idx])
            loss = F.cross_entropy(out, y[idx])
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(img[idx], tok[idx])
                T = CFG["kd_T"]
                kd = F.kl_div(F.log_softmax(out / T, -1), F.softmax(t_out / T, -1), reduction="batchmean") * T * T
                loss = (1 - CFG["kd_alpha"]) * loss + CFG["kd_alpha"] * kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return m.eval()


# Step 6: Train Every Configuration and Measure Accuracy
# key: (label, where it runs, reference config for deltas)
INT8_REF = "baseline_fp32" if not HAS_CUDA else "cpu_baseline_fp32"
CONFIGS = {
    "baseline_fp32": ("Baseline: late fusion, softmax attention, FP32", "primary", None),
    "baseline_fp16": ("Precision: FP16 weights (GPU only)", "primary", "baseline_fp32"),
    "linear_attn": ("Architecture variant: linear attention", "primary", "baseline_fp32"),
    "early_fusion": ("Fusion variant: early fusion, same budget", "primary", "baseline_fp32"),
    "early_fusion_2x": ("Fusion variant: early fusion, 2x epochs", "primary", "baseline_fp32"),
    "student_kd": ("Distilled student (KD from baseline)", "primary", "baseline_fp32"),
    "student_nokd": ("Student, labels only (KD control)", "primary", "baseline_fp32"),
    "cpu_baseline_fp32": ("Baseline on CPU, FP32", "cpu", None),
    "cpu_baseline_int8": ("Quantized: dynamic INT8 on CPU", "cpu", INT8_REF),
}
if not HAS_CUDA:  # no GPU: primary device is the CPU, so the CPU baseline row would duplicate baseline_fp32
    CONFIGS.pop("baseline_fp16")
    CONFIGS.pop("cpu_baseline_fp32")

models, acc_rows = {}, []


def add_acc(key, seed, model, device, dtype=torch.float32):
    a = predict_acc(model, Xte, device, dtype)
    n = len(Xte[2])
    acc_rows.append(dict(tag=RUN_TAG, config=key, seed=seed, accuracy=a, n_test=n, ci95=1.96 * math.sqrt(a * (1 - a) / n)))
    print(f"{key:20s} seed {seed}: acc {a:.4f}")


T_TRAIN = time.perf_counter()
for s in CFG["seeds"]:
    base = train("late", s)
    models[("late", s)] = base
    models[("linear", s)] = train("linear", s)
    models[("early", s)] = train("early", s)
    models[("early2x", s)] = train("early2x", s, epochs=2 * CFG["epochs"])
    models[("student_kd", s)] = train("student_kd", s, teacher=base)
    models[("student_nokd", s)] = train("student_nokd", s)
    add_acc("baseline_fp32", s, base, DEVICE)
    if HAS_CUDA:
        add_acc("baseline_fp16", s, copy.deepcopy(base).half(), DEVICE, torch.float16)
    add_acc("linear_attn", s, models[("linear", s)], DEVICE)
    add_acc("early_fusion", s, models[("early", s)], DEVICE)
    add_acc("early_fusion_2x", s, models[("early2x", s)], DEVICE)
    add_acc("student_kd", s, models[("student_kd", s)], DEVICE)
    add_acc("student_nokd", s, models[("student_nokd", s)], DEVICE)
    cpu_base = copy.deepcopy(base).cpu().eval()
    if HAS_CUDA:
        add_acc("cpu_baseline_fp32", s, cpu_base, "cpu")
    q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(cpu_base), {nn.Linear}, dtype=torch.qint8)
    models[("int8", s)] = q
    add_acc("cpu_baseline_int8", s, q, "cpu")
TRAIN_SECONDS = round(time.perf_counter() - T_TRAIN, 1)
acc_df = pd.DataFrame(acc_rows)
print(acc_df.groupby("config").accuracy.agg(["mean", "std"]).round(4))


# Step 7: Latency and Memory Protocol
WHICH = {"baseline_fp32": ("late", torch.float32), "baseline_fp16": ("late", torch.float16), "linear_attn": ("linear", torch.float32),
         "early_fusion": ("early", torch.float32), "early_fusion_2x": ("early2x", torch.float32), "student_kd": ("student_kd", torch.float32),
         "student_nokd": ("student_nokd", torch.float32), "cpu_baseline_fp32": ("late", torch.float32), "cpu_baseline_int8": ("int8", torch.float32)}
BM = {k: (models[(WHICH[k][0], 0)], v[1], WHICH[k][1]) for k, v in CONFIGS.items()}  # key -> (model, where, dtype)
PRECISION = {k: ("int8" if "int8" in k else "fp16" if WHICH[k][1] == torch.float16 else "fp32") for k in CONFIGS}


def place(key):
    m, where, dt = BM[key]
    dev = DEVICE if where == "primary" else torch.device("cpu")
    mm = copy.deepcopy(m)
    if key != "cpu_baseline_int8":
        mm = mm.to(dev)
        if dt == torch.float16:
            mm = mm.half()
    return mm.eval(), dev, dt


def sample_inputs(dev, dt, n=64):
    return [(Xte[0][i:i + 1].to(dev, dt), Xte[1][i:i + 1].to(dev)) for i in range(n)]


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.inference_mode()
def time_trial(m, dev, ins, n_warm, n_iter):
    for i in range(n_warm):
        m(*ins[i % len(ins)])
    sync(dev)
    ts = []
    for i in range(n_iter):
        a = ins[i % len(ins)]
        sync(dev)
        t0 = time.perf_counter()
        m(*a)
        sync(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    return dict(p50_ms=np.percentile(ts, 50), p95_ms=np.percentile(ts, 95), mean_ms=ts.mean(), std_ms=ts.std(ddof=1), n_iter=n_iter)


# Step 8: Memory Pass (Each Configuration Alone on the Device)
proc = psutil.Process()
mem = {}
# Park everything else on the CPU first: trained models and the training tensors would otherwise
# still be resident on the GPU and be counted in every config's "peak" (reset_peak_memory_stats
# resets to the *current* allocation, not to zero).
for k in list(models):
    models[k] = models[k].cpu()
Xtr = [t.cpu() for t in Xtr]
gc.collect()
if HAS_CUDA:
    torch.cuda.empty_cache()
RESIDENT_BEFORE_MB = torch.cuda.memory_allocated() / 2 ** 20 if HAS_CUDA else float("nan")
print(f"GPU memory still allocated before the memory pass: {RESIDENT_BEFORE_MB:.2f} MB (should be ~0)")
for key in BM:
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    rss0 = proc.memory_info().rss
    m, dev, dt = place(key)
    ins = sample_inputs(dev, dt)
    with torch.inference_mode():
        for i in range(CFG["mem_iters"]):
            m(*ins[i % len(ins)])
    sync(dev)
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    src_model = BM[key][0]
    mem[key] = dict(
        peak_gpu_mb=(torch.cuda.max_memory_allocated() / 2 ** 20 if (HAS_CUDA and dev.type == "cuda") else float("nan")),
        size_mb=buf.tell() / 2 ** 20, rss_delta_mb=(proc.memory_info().rss - rss0) / 2 ** 20,
        params=n_params(models[("late", 0)]) if key == "cpu_baseline_int8" else n_params(src_model), device=dev.type)
    del m, ins
print(pd.DataFrame(mem).T.round(3))


# Step 9: Timing Pass (Interleaved Trials)
placed = {k: place(k) for k in BM}
inputs = {k: sample_inputs(placed[k][1], placed[k][2]) for k in BM}
lat_rows, order_rng = [], random.Random(0)
for trial in range(CFG["trials"]):
    order = list(BM)
    order_rng.shuffle(order)
    for key in order:
        m, dev, dt = placed[key]
        r = time_trial(m, dev, inputs[key], CFG["n_warm"], CFG["n_iter"])
        lat_rows.append(dict(tag=RUN_TAG, config=key, device=dev.type, trial=trial, **r))
lat_df = pd.DataFrame(lat_rows)
print(lat_df.groupby("config")[["p50_ms", "p95_ms"]].agg(["mean", "std"]).round(3))
del placed, inputs


# Step 10: Scaling-Law Compute Inputs (6ND, Measured FLOPs, Chinchilla Token Budget)
from torch.utils.flop_counter import FlopCounterMode

TOK_PER_SAMPLE = MAXLEN + 64


def fwd_flops(name):
    m = ARCH[name]().eval()
    x = (torch.zeros(1, 1, 64, 64), torch.ones(1, MAXLEN, dtype=torch.long))
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(*x)
    return fc.get_total_flops()


scal = []
for name, label, ep in [("late", "baseline (late fusion)", CFG["epochs"]), ("linear", "linear-attention variant", CFG["epochs"]),
                        ("early", "early fusion", CFG["epochs"]), ("student_kd", "distilled student", CFG["epochs"])]:
    scal.append(dict(model=label, params=n_params(ARCH[name]()), fwd_flops_per_sample=fwd_flops(name),
                     train_samples_seen=CFG["n_train"] * ep, unique_train_samples=CFG["n_train"], tokens_per_sample=TOK_PER_SAMPLE))
scal_df = pd.DataFrame(scal)
print(scal_df)


# Step 11: Write the Results and Hardware Manifest
def _nan_none(x):
    return None if isinstance(x, float) and math.isnan(x) else x


def build_results():
    board = {}
    for key, (label, where, ref) in CONFIGS.items():
        a = acc_df[acc_df.config == key].accuracy
        l = lat_df[lat_df.config == key]
        board[key] = dict(
            label=label, precision=PRECISION[key], device=mem[key]["device"], ref=ref, params=int(mem[key]["params"]),
            size_mb=mem[key]["size_mb"], peak_gpu_mb=mem[key]["peak_gpu_mb"], rss_delta_mb=mem[key]["rss_delta_mb"],
            acc_mean=float(a.mean()), acc_std=float(a.std(ddof=1)) if len(a) > 1 else None,
            acc_ci95=float(acc_df[acc_df.config == key].ci95.mean()),
            p50_mean=float(l.p50_ms.mean()), p50_std=float(l.p50_ms.std(ddof=1)) if len(l) > 1 else None,
            p95_mean=float(l.p95_ms.mean()), p95_std=float(l.p95_ms.std(ddof=1)) if len(l) > 1 else None)
    return dict(tag=RUN_TAG, quick=QUICK, cfg=CFG, configs=board, scaling=scal_df.to_dict("records"),
                train_seconds=TRAIN_SECONDS, classes=CLASSES, gpu_resident_before_mem_pass_mb=_nan_none(RESIDENT_BEFORE_MB))


RESULTS = build_results()


def write_outputs():
    (RESULTS_DIR / f"results_{RUN_TAG}.json").write_text(json.dumps(RESULTS, indent=1, default=str), encoding="utf-8")
    acc_df.to_csv(RESULTS_DIR / f"accuracy_runs_{RUN_TAG}.csv", index=False)
    lat_df.to_csv(RESULTS_DIR / f"latency_runs_{RUN_TAG}.csv", index=False)
    manifest = dict(
        tag=RUN_TAG, quick=QUICK, total_runtime_s=round(time.perf_counter() - T_START, 1),
        hardware=hardware_snapshot(ENV), budgets=dict(CFG), timings_s=dict(train_all_models=TRAIN_SECONDS))
    (RESULTS_DIR / f"manifest_{RUN_TAG}.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    print(f"wrote results/results_{RUN_TAG}.json, manifest_{RUN_TAG}.json, accuracy_runs_{RUN_TAG}.csv, latency_runs_{RUN_TAG}.csv "
          f"(total runtime {manifest['total_runtime_s']:.0f}s)")


write_outputs()


# Step 12: Build the Benchmark Workbook (Excel)
sys.path.insert(0, str(ROOT / "scripts"))
import build_workbook

WORKBOOK_PATHS = [build_workbook.build(ROOT, tags=[RUN_TAG], out=RESULTS_DIR / f"Week05_Benchmark_Workbook_{RUN_TAG}.xlsx")]
if not QUICK:  # the combined workbook covers every full run found in results/ (VM CPU and/or T4)
    WORKBOOK_PATHS.append(build_workbook.build(ROOT, tags=None, out=RESULTS_DIR / "Week05_Benchmark_Workbook.xlsx"))
print("workbooks:", ", ".join(p.name for p in WORKBOOK_PATHS))


# Step 13: Save the Results off Colab
if ENV["platform"] == "colab":
    import shutil
    from google.colab import files

    archive = shutil.make_archive(f"brightline_results_{RUN_TAG}", "zip", RESULTS_DIR)
    files.download(archive)
else:
    print(f"results are in {RESULTS_DIR}")


# Step 14: Write the Results into README.md and Push (asks first)
try:
    import update_readme
except ImportError:
    update_readme = None
    print("[publish] scripts/update_readme.py not found: run from a git clone to write README.md (COLAB.md).")
if update_readme and QUICK:
    print("[publish] quick smoke test: results are not written to README.md")
elif update_readme:
    update_readme.offer_publish(ROOT, RUN_TAG, interactive=IN_NOTEBOOK or sys.stdin.isatty())
