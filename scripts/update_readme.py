# =============================================================================
# region SCRIPT DETAILS
# =============================================================================
# CSP CST 627 100  Deep Learning Neural Networks
# Week 05 Assignment: Latency, Memory, Accuracy: A Measured Trade-off Board
# Objective: Write the Intel-VM vs Colab-T4 comparison into README.md and,
# optionally, push it to GitHub with a fine-grained personal access token.
#
# Usage:
#   python scripts/update_readme.py            # re-render the README results section
#   python scripts/update_readme.py --push     # also commit + push (prompts for a token)
#   python scripts/update_readme.py --push --tag t4gpu
# The last step of the assignment notebook calls offer_publish(), which asks first.
# =============================================================================
# endregion
# =============================================================================
# Step 1: Render the Results Section
# Reads results/results_<tag>.json + manifest_<tag>.json for every full-run tag present
# (cpu = the Intel VM, t4gpu = Colab T4, gpu = any other GPU) and renders side-by-side
# hardware and board tables between the README markers.
# =============================================================================
"""
https://docs.python.org/3/library/json.html
https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/organizing-information-with-tables
"""

import argparse
import base64
import getpass
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN, END = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
TAG_ORDER = ["cpu", "t4gpu", "gpu", "cpu_public", "t4gpu_public", "gpu_public"]
HW_LABEL = {"cpu": "Intel VM · CPU only", "t4gpu": "Colab · T4 GPU", "gpu": "Other GPU"}
TAG_LABEL = {**HW_LABEL, **{f"{k}_public": f"{v} (public set)" for k, v in HW_LABEL.items()}}


def hw_of(tag):
    return tag[: -len("_public")] if tag.endswith("_public") else tag


def load_runs(root):
    runs = {}
    for tag in TAG_ORDER:
        r, m = root / "results" / f"results_{tag}.json", root / "results" / f"manifest_{tag}.json"
        if r.exists() and m.exists():
            runs[tag] = (json.loads(r.read_text(encoding="utf-8")), json.loads(m.read_text(encoding="utf-8")))
    return runs


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def fmt(x, kind):
    x = _num(x)
    if x is None:
        return "n/a"
    return {"p": f"{x:.3f}", "ms": f"{x:.3f} ms", "mb": f"{x:,.2f} MB", "s": f"{x:,.1f} s", "int": f"{int(x):,}"}[kind]


def device_label(m):
    h = m["hardware"]
    if h.get("gpu_name"):
        return f"{h['gpu_name']} ({h['vram_gb']} GB, sm_{str(h['compute_capability']).replace('.', '')})"
    vm = " (virtual machine)" if h.get("virtualized") else ""
    return f"CPU only: {h['cpu_model']}, {h['cpu_count']} logical cores{vm}"


def board_lines(runs, tags):
    """Accuracy, latency and size tables for the given tags (one evaluation set, one or two machines)."""
    head = "| Config | " + " | ".join(HW_LABEL[hw_of(t)] for t in tags) + " |"
    sep = "|---|" + "---:|" * len(tags)
    keys = []
    for t in tags:
        keys += [k for k in runs[t][0]["configs"] if k not in keys]

    def acc_cell(t, k):
        c = runs[t][0]["configs"].get(k)
        if not c:
            return "–"
        sd = f" ± {c['acc_std']:.3f}" if c.get("acc_std") is not None else ""
        return f"{c['acc_mean']:.3f}{sd}"

    def lat_cell(t, k):
        c = runs[t][0]["configs"].get(k)
        return f"{c['p50_mean']:.3f} / {c['p95_mean']:.3f}" if c else "–"

    def mem_cell(t, k):
        c = runs[t][0]["configs"].get(k)
        return f"{c['size_mb']:.3f} / {fmt(c['peak_gpu_mb'], 'p')}" if c else "–"

    board = [head, sep]
    for label, fn in (("Accuracy (mean ± seed std)", acc_cell), ("p50 / p95 latency, ms (batch 1)", lat_cell),
                      ("Model size (MB) / peak GPU memory above floor (MB)", mem_cell)):
        board.append("| **" + label + "** |" + " |" * len(tags))
        for k in keys:
            board.append(f"| `{k}` | " + " | ".join(fn(t, k) for t in tags) + " |")
    return board


def render_section(root=ROOT):
    runs = load_runs(root)
    if not runs:
        return f"{BEGIN}\n_No full-run results in `results/` yet._\n{END}"
    syn = [t for t in runs if not t.endswith("_public")]
    pub = [t for t in runs if t.endswith("_public")]
    hw_tags = syn or pub
    cmp_ = len(hw_tags) > 1
    head = "| | " + " | ".join(HW_LABEL[hw_of(t)] for t in hw_tags) + " |"
    sep = "|---|" + "---:|" * len(hw_tags)
    hw = [head, sep,
          "| Device | " + " | ".join(device_label(runs[t][1]) for t in hw_tags) + " |",
          "| Host class | " + " | ".join(f"`{runs[t][1]['hardware']['host_class']}`" for t in hw_tags) + " |",
          "| Platform / Python | " + " | ".join(f"{runs[t][1]['hardware']['platform']} / {runs[t][1]['hardware']['python']}" for t in hw_tags) + " |",
          "| torch | " + " | ".join(f"`{runs[t][1]['hardware']['torch']}`" for t in hw_tags) + " |",
          "| **Total notebook runtime** (synthetic set) | " + " | ".join(fmt(runs[t][1]["total_runtime_s"], "s") for t in hw_tags) + " |"]

    body = []
    if syn:
        body += ["### Board: synthetic set (generated here)", "", *board_lines(runs, syn), ""]
    if pub:
        d = runs[pub[0]][0].get("dataset") or {}
        body += ["### Board: public set (Flickr8k photos + human captions, derived labels)", "",
                 f"Real photos with human captions from Hugging Face `{d.get('dataset', 'intro/flickr8k')}` (license declared CC0-1.0). "
                 f"{d.get('n_train', '?')} training and {d.get('n_test', '?')} held-out photos. The dataset has no class labels: the six scene classes "
                 "are derived from captions 1–4 by a keyword rule, and caption 0 is the text input. "
                 f"A keyword rule on that text alone scores {d.get('keyword_rule_on_text_acc', float('nan')):.3f} (chance {d.get('chance', float('nan')):.3f}), "
                 "so read this set as a check that the technique ranking holds on real data, not as a document-classification result. "
                 "Accuracy is not comparable with the synthetic set.", "", *board_lines(runs, pub), ""]

    note = "" if cmp_ else ("\n> Only one machine is in `results/` so far. The other column appears after you run it "
                            "(Windows/macOS installer for the VM, COLAB.md for the T4).\n")
    fig = root / "results" / "figs_workbook" / "board_tradeoff.png"
    return "\n".join([
        BEGIN,
        "<!-- Generated by scripts/update_readme.py from results/*.json; edits inside these markers are overwritten. -->",
        "## Results: Intel VM vs Colab T4",
        note,
        "Same code, same seeds, same budgets on every machine. **Accuracy should agree across hardware** (small drift comes from "
        "floating-point order); **latency and memory are what the hardware changes.** On the T4 run, GPU configs execute on the T4 and "
        "the `cpu_*` rows on Colab's 2-vCPU CPU; on the VM everything runs on the CPU. Full detail, every run, formulas and the "
        "scaling-law sheet: `results/Week05_Benchmark_Workbook.xlsx`.",
        "",
        "### Hardware", "", *hw, "",
        *body,
        *([f"![Accuracy vs latency](results/figs_workbook/{fig.name})", ""] if fig.exists() else []),
        END])


def write_readme(root=ROOT):
    path = root / "README.md"
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        raise SystemExit(f"README.md has no {BEGIN} ... {END} markers; add them where the results should go.")
    new = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: render_section(root), text, flags=re.S)
    changed = new != text
    if changed:
        path.write_text(new, encoding="utf-8")
    return changed


# =============================================================================
# Step 2: Commit and Push with a Fine-Grained Token
# The token comes from the Colab secret GITHUB_TOKEN if it exists, otherwise from a hidden
# getpass prompt (blank = use this machine's existing git credentials). It is sent per command
# as an HTTP auth header: never printed, never written to .git/config, never put in the remote
# URL. Results commit first, then a rebase pull (our results win on conflict), then the README
# is re-rendered so the comparison includes the latest results from GitHub, then push.
# =============================================================================
"""
https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token
https://git-scm.com/docs/git-config#Documentation/git-config.txt-httpextraHeader
https://git-scm.com/docs/git-pull#Documentation/git-pull.txt---rebase
https://docs.python.org/3/library/getpass.html
"""


class GitError(RuntimeError):
    pass


def _git(root, *args, token=None, check=True):
    cmd = ["git"]
    basic = None
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}"]
    out = subprocess.run(cmd + list(args), cwd=root, capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip()
    if token:
        text = text.replace(token, "***").replace(basic, "***")
    if check and out.returncode != 0:
        raise GitError(f"git {args[0]} failed:\n{text}")
    return out.returncode, text


def colab_secret_token():
    try:
        from google.colab import userdata

        return userdata.get("GITHUB_TOKEN") or None
    except Exception:  # not on Colab, secret missing, or notebook access not granted
        return None


def get_token(ask_secret=getpass.getpass):
    token = colab_secret_token()
    if token:
        print("[publish] using the Colab secret GITHUB_TOKEN")
        return token
    print("[publish] Fine-grained token: this repo only, Repository permissions > Contents: Read and write.")
    token = ask_secret("[publish] GitHub token (hidden; blank = use existing git credentials): ").strip()
    return token or None


def ensure_identity(root, ask=input):
    for key, prompt in (("user.name", "Your name for the commit: "), ("user.email", "Your email for the commit: ")):
        if not _git(root, "config", key, check=False)[1]:
            _git(root, "config", key, ask(f"[publish] {prompt}").strip())


def require_clone(root):
    code, top = _git(root, "rev-parse", "--show-toplevel", check=False)
    if code != 0 or Path(top).resolve() != Path(root).resolve():
        raise GitError(f"{root} is not the top of a git clone. On Colab: !git clone <repo> then %cd into it (COLAB.md).")


def push_results(root, tag, token=None, ask=input):
    require_clone(root)
    ensure_identity(root, ask)
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")[1]
    tags = [tag] if tag.endswith("_public") else [tag, f"{tag}_public"]  # a hardware tag also covers its public-set run
    result_paths = [p for p in [f"results/{n}_{t}.{x}" for t in tags for n, x in
                                (("results", "json"), ("manifest", "json"), ("accuracy_runs", "csv"), ("latency_runs", "csv"))]
                    + ["results/public_subset.csv", "results/Week05_Benchmark_Workbook.xlsx", "results/figs_workbook"]
                    if (root / p).exists()]
    _git(root, "checkout", "--", "README.md")  # regenerated after the pull, so drop the local render
    _git(root, "add", "--", *result_paths)
    if _git(root, "diff", "--cached", "--quiet", check=False)[0]:
        _git(root, "commit", "-m", f"Add {TAG_LABEL.get(tag, tag)} results ({tag})")
    print(f"[publish] pulling origin/{branch} ...")
    code, text = _git(root, "pull", "--rebase", "-X", "theirs", "origin", branch, token=token, check=False)
    if code != 0:
        _git(root, "rebase", "--abort", check=False)
        raise GitError(f"pull --rebase failed, nothing was pushed:\n{text}")
    write_readme(root)
    _git(root, "add", "README.md")
    if _git(root, "diff", "--cached", "--quiet", check=False)[0]:
        _git(root, "commit", "-m", "Update README results: Intel VM vs Colab T4 comparison")
    print(f"[publish] pushing to origin/{branch} ...")
    _git(root, "push", "origin", f"HEAD:{branch}", token=token)
    print(f"[publish] done: README.md and {len(result_paths)} result path(s) pushed to origin/{branch}")


# =============================================================================
# Step 3: Prompt at the End of a Run
# Called by the notebook's last step: asks before touching README.md and asks again before
# pushing. Non-interactive runs only print how to do it later.
# =============================================================================


def _yes(answer):
    return answer.strip().lower() in ("y", "yes")


def offer_publish(root, tag, interactive, ask=input, ask_secret=getpass.getpass):
    later = "python scripts/update_readme.py --push"
    if not interactive:
        print(f"[publish] non-interactive run: to write README.md and push later, run  {later}")
        return
    try:
        if not _yes(ask("Write these results into README.md (Intel VM vs Colab T4 comparison)? [y/N] ")):
            print(f"[publish] skipped. Later: {later}")
            return
        changed = write_readme(root)
        print(f"[publish] README.md {'updated' if changed else 'already up to date'} (results section only)")
        if not _yes(ask(f"Commit and push README.md + results/*_{tag}* to GitHub? [y/N] ")):
            print("[publish] not pushed. README.md is updated locally only.")
            return
        require_clone(root)  # fail before asking for a token that could not be used
        push_results(root, tag, token=get_token(ask_secret), ask=ask)
    except (EOFError, KeyboardInterrupt):
        print(f"\n[publish] cancelled. Later: {later}")
    except GitError as err:
        print(f"[publish] {err}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render the README results section; optionally commit and push.")
    ap.add_argument("--push", action="store_true", help="commit results + README and push (prompts for a token)")
    ap.add_argument("--tag", default=None, help="results tag to commit (default: t4gpu if present, else cpu)")
    args = ap.parse_args()
    if args.push:
        runs = load_runs(ROOT)
        tag = args.tag or ("t4gpu" if "t4gpu" in runs else "cpu")
        try:
            require_clone(ROOT)
            push_results(ROOT, tag, token=get_token())
        except GitError as err:
            sys.exit(f"[publish] {err}")
    else:
        print(f"README.md {'updated' if write_readme(ROOT) else 'already up to date'}")
