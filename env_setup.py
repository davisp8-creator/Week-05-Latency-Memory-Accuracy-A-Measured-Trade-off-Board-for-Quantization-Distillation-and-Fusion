"""
Environment detector + dependency installer for the Brightline "Measured Trade-off Board" project.

Supported targets:
  1. Google Colab with a T4 GPU (the graded run): detects the T4, never touches Colab's CUDA torch.
  2. Google Colab on a CPU runtime: detects it and prints how to switch to a T4.
  3. A local Intel virtual machine (e.g. Windows Server on Hyper-V): detected as `intel-vm`, CPU only.
  4. Any other Windows / macOS / Linux machine (CPU; any CUDA GPU is used if torch can see it).

Usage
-----
Notebook / script (the Colab notebook has this file inlined into its first code cell):

    from env_setup import setup
    ENV = setup()            # installs only what is missing, then reports the device
    DEVICE = ENV["device"]   # "cuda" or "cpu"
    RUN_TAG = ENV["tag"]     # "t4gpu" | "gpu" | "cpu"; results are written with this suffix
    ENV["host_class"]        # "colab-t4" | "colab-gpu" | "colab-cpu" | "intel-vm" | "intel-host" | "apple-silicon" | ...

Command line (the macOS/Windows installers call this inside the project's .venv):

    python env_setup.py              # detect, install missing packages, print a summary
    python env_setup.py --check      # detect and summarise only, install nothing
"""

import importlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# import name -> pip spec. Every third-party import in the assignment script must appear here.
PACKAGE_MAP = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "torch": "torch",
    "openpyxl": "openpyxl",
    "psutil": "psutil",
}
ASSIGNMENT_SCRIPT = Path("scripts") / "Assignment_05_Tradeoff_Board.py"
# helper modules the assignment script imports at run time; their third-party imports count too
HELPER_SCRIPTS = [Path("scripts") / "build_workbook.py", Path("scripts") / "update_readme.py"]
LOCAL_MODULES = {"env_setup", "update_readme", "build_workbook"}
COLAB_ONLY_MODULES = {"google"}   # google.colab: provided by Colab, only imported behind a platform check
MIN_PYTHON = (3, 10)
T4_RUNTIME_HINT = (
    "In Colab: Runtime > Change runtime type > Hardware accelerator: T4 GPU > Save,\n"
    "then Runtime > Run all. (The runtime restarts, so re-run from the top.)"
)
HOST_LABEL = {
    "colab-t4": "Colab (T4 GPU)", "colab-gpu": "Colab (other GPU)", "colab-cpu": "Colab (CPU runtime)",
    "intel-vm": "Intel VM (CPU only)", "intel-host": "Intel PC (CPU only)", "apple-silicon": "Apple silicon (CPU only)",
}


# ── detection ────────────────────────────────────────────────────────────────
def is_colab() -> bool:
    if importlib.util.find_spec("google") is not None:
        try:
            import google.colab  # noqa: F401
            return True
        except ImportError:
            pass
    return "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


def query_nvidia_smi() -> dict | None:
    """GPU name / VRAM / driver from nvidia-smi: the torch-independent signal that a GPU is attached."""
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False)
    if out.returncode != 0 or not out.stdout.strip():
        return None
    name, mem_mib, driver = [s.strip() for s in out.stdout.strip().splitlines()[0].split(",")]
    return {"name": name, "vram_gb": round(float(mem_mib) / 1024, 1), "driver": driver}


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _powershell(expr: str) -> str:
    return _run(["powershell", "-NoProfile", "-Command", expr])


def _read(path: str) -> str:
    try:
        return Path(path).read_text(errors="ignore").strip()
    except OSError:
        return ""


VM_WORDS = ("virtual", "vmware", "kvm", "qemu", "xen", "hyper-v", "google compute", "bochs", "parallels", "vbox")


def detect_cpu_and_vm() -> dict:
    """CPU model/vendor and whether this machine is a virtual machine, without needing torch.

    Windows: Win32_ComputerSystem / Win32_Processor.  Linux: DMI + /proc/cpuinfo.  macOS: sysctl.
    """
    system = platform.system()
    cpu, product, vendor_sys, hv_flag = "", "", "", None
    if system == "Windows":
        cpu = _powershell("(Get-CimInstance Win32_Processor | Select-Object -First 1).Name")
        csys = _powershell("$c = Get-CimInstance Win32_ComputerSystem; \"$($c.Manufacturer)|$($c.Model)|$($c.HypervisorPresent)\"")
        vendor_sys, product, hv = (csys.split("|") + ["", "", ""])[:3]
        hv_flag = hv.strip().lower() == "true"
    elif system == "Linux":
        info = _read("/proc/cpuinfo")
        m = re.search(r"model name\s*:\s*(.+)", info)
        cpu = m.group(1).strip() if m else ""
        product, vendor_sys = _read("/sys/class/dmi/id/product_name"), _read("/sys/class/dmi/id/sys_vendor")
        hv_flag = bool(re.search(r"^flags\s*:.*\bhypervisor\b", info, flags=re.M))
        if shutil.which("systemd-detect-virt"):
            sdv = _run(["systemd-detect-virt"])
            if sdv and sdv != "none":
                product = f"{product} ({sdv})".strip()
    elif system == "Darwin":
        cpu = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        hv_flag = _run(["sysctl", "-n", "kern.hv_vmm_present"]) == "1"
        product, vendor_sys = _run(["sysctl", "-n", "hw.model"]), "Apple"
    cpu = cpu or platform.processor() or platform.machine()
    sig = f"{product} {vendor_sys}".lower()
    # Model/vendor strings identify a VM reliably. HypervisorPresent alone is NOT enough on Windows:
    # a physical host running Hyper-V/VBS also reports it, so it only counts when the model agrees
    # or (macOS/Linux) the kernel flag is the sole signal available.
    virtualized = any(w in sig for w in VM_WORDS) or (system != "Windows" and bool(hv_flag))
    low = cpu.lower()
    vendor = "intel" if ("intel" in low or "genuineintel" in low) else "apple" if ("apple" in low or platform.machine() == "arm64") \
        else "amd" if "amd" in low else "other"
    return {"cpu_model": cpu, "cpu_vendor": vendor, "virtualized": virtualized, "vm_product": product or None,
            "vm_vendor": vendor_sys or None, "hypervisor_flag": hv_flag}


def ram_gib() -> float | None:
    try:
        import psutil
        return round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        return None


def classify_host(env: dict, gpu_name: str | None = None, device: str = "cpu") -> str:
    """One label that says what the numbers were measured on."""
    if env["platform"] == "colab":
        smi = env["nvidia_smi"]
        name = gpu_name or (smi["name"] if smi else None)
        if name and "T4" in name:
            return "colab-t4"
        return "colab-gpu" if (name or device == "cuda") else "colab-cpu"
    if env["cpu_vendor"] == "apple":
        return "apple-silicon"
    if env["cpu_vendor"] == "intel":
        return "intel-vm" if env["virtualized"] else "intel-host"
    return "other-vm" if env["virtualized"] else "other-host"


def detect_environment() -> dict:
    colab = is_colab()
    system = platform.system()  # "Windows" | "Darwin" | "Linux"
    env = {
        "platform": "colab" if colab else {"Windows": "windows", "Darwin": "macos"}.get(system, "linux"),
        "system": system,
        "python": platform.python_version(),
        "nvidia_smi": query_nvidia_smi(),
    }
    env.update(detect_cpu_and_vm())
    env["host_class"] = classify_host(env)
    return env


# ── installation ─────────────────────────────────────────────────────────────
def required_imports(script: Path) -> list[str]:
    """Top-level third-party imports of the assignment script (stdlib and local modules excluded)."""
    import ast
    lines = script.read_text(encoding="utf-8").splitlines()
    source = "\n".join("" if ln.lstrip()[:1] in ("!", "%") else ln for ln in lines)  # drop notebook magics
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return sorted(found - set(sys.stdlib_module_names) - LOCAL_MODULES - COLAB_ONLY_MODULES)


def missing_packages(imports: list[str]) -> list[str]:
    unmapped = [i for i in imports if i not in PACKAGE_MAP]
    if unmapped:
        raise SystemExit(f"Unmapped third-party import(s): {unmapped}. Add them to PACKAGE_MAP in env_setup.py.")
    return [PACKAGE_MAP[i] for i in imports if importlib.util.find_spec(i) is None]


def project_root() -> Path:
    # __file__ is absent when this module is inlined into the Colab notebook
    return Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def install_missing(env: dict, script: Path | None = None) -> list[str]:
    script = script or project_root() / ASSIGNMENT_SCRIPT
    if script.exists():
        found = set(required_imports(script))
        for helper in HELPER_SCRIPTS:
            if (script.parent.parent / helper).exists():
                found |= set(required_imports(script.parent.parent / helper))
        imports = sorted(found)
    else:
        imports = list(PACKAGE_MAP)
    todo = missing_packages(imports)
    if env["platform"] == "colab" and "torch" in todo:
        # Colab ships a CUDA build of torch; this only triggers if someone uninstalled it.
        print("[setup] torch is missing on Colab; installing the default (CUDA-enabled on Linux) build.")
    if todo:
        print(f"[setup] installing missing packages: {' '.join(todo)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *todo])
        importlib.invalidate_caches()
    else:
        print(f"[setup] all required packages already present: {' '.join(PACKAGE_MAP[i] for i in imports)}")
    return todo


# ── device resolution ────────────────────────────────────────────────────────
def resolve_device(env: dict) -> dict:
    torch = importlib.import_module("torch")
    info = {"torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
            "device": "cpu", "gpu_name": None, "vram_gb": None, "compute_capability": None,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(device="cuda", gpu_name=p.name, vram_gb=round(p.total_memory / 2**30, 1),
                    compute_capability=f"{p.major}.{p.minor}")
    info["is_t4"] = bool(info["gpu_name"] and "T4" in info["gpu_name"])
    info["tag"] = "t4gpu" if info["is_t4"] else ("gpu" if info["device"] == "cuda" else "cpu")
    info["host_class"] = classify_host(env, info["gpu_name"], info["device"])
    return info


def _warn(msg: str) -> None:
    bar = "!" * 72
    print(f"\n{bar}\n{msg}\n{bar}\n")


def diagnose(env: dict) -> None:
    """Explain anything that would make the run slower or the results mislabelled."""
    smi = env["nvidia_smi"]
    if env["platform"] == "colab" and smi is None:
        _warn("Colab is on a CPU-only runtime. Everything still runs, but GPU rows fall back to the CPU,\n"
              "so timings and peak VRAM will not be T4 numbers.\n" + T4_RUNTIME_HINT)
    elif smi and env["device"] == "cpu":
        _warn(f"nvidia-smi sees a GPU ({smi['name']}) but torch cannot use it "
              f"(torch {env['torch']}, CUDA build: {env['torch_cuda_build']}).\n"
              "Usually a CPU-only torch was pip-installed over the CUDA build (for example, from\n"
              "requirements.txt). On Colab: Runtime > Disconnect and delete runtime, then run again\n"
              "without installing torch. Locally: install the CUDA build from pytorch.org.")
    elif env["device"] == "cuda" and not env["is_t4"]:
        print(f"[setup] note: GPU is {env['gpu_name']}, not a T4. Results are tagged '{env['tag']}' "
              "so they are not confused with the graded T4 run.")
    if env["platform"] != "colab" and env["device"] == "cpu" and env["virtualized"]:
        print(f"[setup] note: running inside a virtual machine ({env['vm_product']}). Latency here includes "
              "hypervisor scheduling jitter; the benchmark reports p95 and trial-to-trial spread for that reason.")
    if env["mps_available"] and env["device"] == "cpu":
        print("[setup] note: Apple MPS is available but deliberately not used: the benchmark compares CPU INT8 "
              "against a single reference device, and dynamic INT8 kernels exist only on CPU.")


def hardware_snapshot(env: dict) -> dict:
    """Everything a reader needs to interpret wall-clock and VRAM numbers."""
    snap = {k: env.get(k) for k in ("platform", "system", "python", "torch", "torch_cuda_build",
                                    "device", "gpu_name", "vram_gb", "compute_capability", "tag", "host_class",
                                    "cpu_model", "cpu_vendor", "virtualized", "vm_product")}
    snap["host_label"] = HOST_LABEL.get(env.get("host_class"), env.get("host_class"))
    snap["cpu_count"] = os.cpu_count()
    snap["ram_gib"] = ram_gib()
    snap["driver"] = (env.get("nvidia_smi") or {}).get("driver")
    for pkg in ("numpy", "pandas", "matplotlib"):
        try:
            snap[pkg] = importlib.import_module(pkg).__version__
        except ImportError:
            snap[pkg] = None
    return snap


def setup(install: bool = True, require_t4: bool = False) -> dict:
    """Detect the environment, install missing packages, resolve the device, and print a summary."""
    if sys.version_info < MIN_PYTHON:
        raise SystemExit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, found {platform.python_version()}")
    env = detect_environment()
    if install:
        install_missing(env)
    env.update(resolve_device(env))
    diagnose(env)
    if require_t4 and not env["is_t4"]:
        raise RuntimeError("require_t4=True but no Tesla T4 is visible.\n" + T4_RUNTIME_HINT)
    gpu = f"{env['gpu_name']} ({env['vram_gb']} GB, sm_{(env['compute_capability'] or '').replace('.', '')})" \
        if env["device"] == "cuda" else "none"
    print(f"[setup] host={env['host_class']} ({HOST_LABEL.get(env['host_class'], env['host_class'])})  "
          f"platform={env['platform']}  python={env['python']}  torch={env['torch']}  device={env['device']}  gpu={gpu}  "
          f"results tag='{env['tag']}'")
    print(f"[setup] cpu={env['cpu_model']}  virtualized={env['virtualized']}"
          + (f" ({env['vm_product']})" if env["vm_product"] and env["virtualized"] else ""))
    return env


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="detect and summarise only; install nothing")
    args = ap.parse_args()
    result = setup(install=not args.check)
    print(json.dumps(hardware_snapshot(result), indent=1))
