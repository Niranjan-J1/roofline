"""Hardware and Slurm environment detection.

Detection never raises. Anything that cannot be determined is left as None and
explained in a warning. Only NVIDIA GPUs are assessed.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from dataclasses import dataclass, replace
from typing import Mapping

import psutil

GB = 1e9
MIB = 1024 * 1024

SCOPE_NOTE = (
    "Only NVIDIA GPUs are assessed (not Intel, AMD or Apple graphics). GPUs are "
    "checked on this machine only. Container memory limits may not be detected."
)

_SLURM_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOBID",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NNODES",
    "SLURM_NTASKS",
    "SLURM_GPUS_ON_NODE",
    "SLURM_GPUS_PER_NODE",
    "SLURM_GPUS",
    "SLURM_MEM_PER_NODE",
    "SLURM_MEM_PER_CPU",
    "SLURM_CPUS_ON_NODE",
)


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str | None
    name: str
    total_bytes: int | None
    free_bytes: int | None
    mig: bool = False


@dataclass(frozen=True)
class Probe:
    """Result of asking the system for its GPUs."""

    # "ok": the query worked (possibly zero GPUs)
    # "unavailable": no NVIDIA tooling present, so CPU-only is assumed
    # "failed": tooling is present but the check did not work
    status: str
    detail: str  # method used, or why it did not work
    gpus: tuple[GPU, ...] = ()


@dataclass(frozen=True)
class Slurm:
    job_id: str
    nodes: int | None
    tasks: int | None
    gpus_on_node: int | None
    job_gpus: int | None
    mem_per_node_bytes: int | None
    variables: tuple[tuple[str, str], ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class Hardware:
    gpu_count: int | None  # total across all nodes; None means unknown
    gpu_count_source: str  # "detected" or "override"
    vram_bytes: int | None  # per GPU (smallest if mixed)
    vram_source: str | None
    ram_total_bytes: int | None  # per node
    ram_available_bytes: int | None
    ram_source: str | None
    probe: Probe
    visible: tuple[GPU, ...]
    cuda_visible_devices: str | None
    slurm: Slurm | None
    warnings: tuple[str, ...]


# --- GPU queries -------------------------------------------------------------


def _text(value: object) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def _query_nvml() -> Probe:
    try:
        import pynvml
    except ImportError:
        return Probe("unavailable", "NVIDIA Python library not installed")
    try:
        pynvml.nvmlInit()
        try:
            gpus = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    mig = bool(pynvml.nvmlDeviceGetMigMode(handle)[0])
                except Exception:  # not supported on this GPU or driver
                    mig = False
                gpus.append(
                    GPU(
                        i,
                        _text(pynvml.nvmlDeviceGetUUID(handle)),
                        _text(pynvml.nvmlDeviceGetName(handle)),
                        int(memory.total),
                        int(memory.free),
                        mig,
                    )
                )
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        if type(e).__name__ == "NVMLError_LibraryNotFound":
            return Probe("unavailable", "NVIDIA driver library not found")
        return Probe("failed", f"NVIDIA Python library: {e}")
    return Probe("ok", "NVIDIA Python library", tuple(gpus))


def _mib(value: str) -> int | None:
    try:
        return int(float(value)) * MIB
    except (ValueError, OverflowError):  # e.g. "[N/A]"
        return None


def _query_smi() -> Probe:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return Probe("unavailable", "nvidia-smi not found")
    fields = "index,uuid,name,memory.total,memory.free"
    error = "no output"
    # Older drivers reject the MIG field, so retry without it.
    for query in (fields + ",mig.mode.current", fields):
        try:
            run = subprocess.run(
                [exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as e:
            return Probe("failed", f"nvidia-smi: {e}")
        output = run.stdout.strip()
        if "no devices were found" in (output + run.stderr).lower():
            return Probe("ok", "nvidia-smi")
        if run.returncode != 0:
            lines = (output or run.stderr.strip()).splitlines()
            error = lines[0] if lines else f"exit code {run.returncode}"
            continue
        gpus = []
        for row in csv.reader(output.splitlines(), skipinitialspace=True):
            if len(row) < 5 or not row[0].strip().isdigit():
                return Probe("failed", "nvidia-smi: unrecognized output")
            gpus.append(
                GPU(
                    int(row[0]),
                    row[1].strip(),
                    row[2].strip(),
                    _mib(row[3]),
                    _mib(row[4]),
                    len(row) > 5 and row[5].strip().lower() == "enabled",
                )
            )
        return Probe("ok", "nvidia-smi", tuple(gpus))
    return Probe("failed", f"nvidia-smi: {error}")


def probe_gpus() -> Probe:
    """Try the NVIDIA Python library, then nvidia-smi."""
    first = _query_nvml()
    if first.status == "ok":
        return first
    second = _query_smi()
    if second.status != "unavailable":
        return second
    # Neither worked: a real failure outranks missing tooling.
    return first if first.status == "failed" else second


def visible_gpus(
    gpus: tuple[GPU, ...], env: Mapping[str, str]
) -> tuple[tuple[GPU, ...] | None, tuple[str, ...]]:
    """Apply CUDA_VISIBLE_DEVICES. Returns (None, why) if it cannot be matched."""
    setting = env.get("CUDA_VISIBLE_DEVICES")
    if setting is None or not gpus:
        return gpus, ()
    chosen: list[GPU] = []
    warnings: list[str] = []
    used_index = False
    for entry in (e.strip() for e in setting.split(",")):
        if entry in ("", "-1"):  # CUDA hides this entry and all that follow
            break
        if entry.isdigit():
            used_index = True
            matches = [g for g in gpus if g.index == int(entry)]
            if not matches:
                warnings.append(
                    f"CUDA_VISIBLE_DEVICES lists GPU {entry}, which was not detected; "
                    "CUDA ignores it and any entries after it."
                )
                break
        elif entry.startswith("GPU-"):
            matches = [g for g in gpus if g.uuid and g.uuid.startswith(entry)]
        else:
            matches = []
        if len(matches) != 1 or matches[0] in chosen:
            return None, (
                "CUDA_VISIBLE_DEVICES has entries that cannot be matched reliably "
                "(such as MIG devices), so the usable GPU count is uncertain.",
            )
        chosen.append(matches[0])
    # CUDA's default numbering can differ from the numbering used to detect GPUs.
    if (
        used_index
        and 0 < len(chosen) < len(gpus)
        and len({(g.name, g.total_bytes) for g in gpus}) > 1
        and env.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID"
    ):
        warnings.append(
            "GPUs differ and CUDA_VISIBLE_DEVICES selects by number. CUDA may number "
            "them differently, so which GPUs are usable is uncertain."
        )
    return tuple(chosen), tuple(warnings)


# --- Slurm and RAM -----------------------------------------------------------


def _int(value: str | None) -> int | None:
    value = (value or "").strip()
    return int(value) if value.isascii() and value.isdigit() else None


def _gpu_spec(value: str) -> int | None:
    """Parse '4', 'a100:4', 'gpu:a100:4' or 'a100:2,v100:2'."""
    counts = [_int(part.split(":")[-1]) for part in value.split(",")]
    return None if None in counts else sum(counts)


def detect_slurm(env: Mapping[str, str]) -> Slurm | None:
    job_id = env.get("SLURM_JOB_ID") or env.get("SLURM_JOBID")
    if not job_id:
        return None
    warnings: list[str] = []

    def read(keys: tuple[str, ...], parse=_int) -> int | None:
        for key in keys:
            if env.get(key):
                value = parse(env[key])
                if value is None:
                    warnings.append(f"Could not read {key}.")
                return value
        return None

    nodes = read(("SLURM_JOB_NUM_NODES", "SLURM_NNODES"))
    if not nodes:
        nodes = None
        warnings.append("Slurm node count is unknown.")
    memory = None
    per_node = read(("SLURM_MEM_PER_NODE",))
    if per_node:  # 0 means all of the node's memory
        memory = per_node * MIB
    elif per_node is None:
        per_cpu = read(("SLURM_MEM_PER_CPU",))
        cpus = read(("SLURM_CPUS_ON_NODE",))
        if per_cpu and cpus:
            memory = per_cpu * cpus * MIB
        elif per_cpu:
            warnings.append("Slurm sets memory per CPU, but the CPU count is unknown.")
    return Slurm(
        job_id=job_id,
        nodes=nodes,
        tasks=read(("SLURM_NTASKS",)),
        gpus_on_node=read(("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE"), _gpu_spec),
        job_gpus=read(("SLURM_GPUS",), _gpu_spec),
        mem_per_node_bytes=memory,
        variables=tuple((k, env[k]) for k in _SLURM_KEYS if k in env),
        warnings=tuple(warnings),
    )


def _detect_ram() -> tuple[int | None, int | None]:
    try:
        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    except Exception:
        return None, None


# --- Combined ----------------------------------------------------------------


def detect_hardware(
    gpus: int | None = None,
    vram: float | None = None,
    ram: float | None = None,
    env: Mapping[str, str] | None = None,
) -> Hardware:
    """Detect hardware, then apply overrides (vram and ram are in GB)."""
    env = os.environ if env is None else env
    probe = probe_gpus()
    slurm = detect_slurm(env)
    warnings: list[str] = list(slurm.warnings) if slurm else []

    count: int | None = None
    smallest: int | None = None
    visible: tuple[GPU, ...] = ()
    if probe.status == "unavailable":
        count = 0
    elif probe.status == "failed":
        warnings.append(f"Could not check for GPUs ({probe.detail}).")
    else:
        matched, notes = visible_gpus(probe.gpus, env)
        warnings += notes
        if matched is not None:
            visible = matched
            count = len(visible)

    if any(g.mig for g in visible or probe.gpus):
        warnings.append(
            "MIG (split GPU) mode is on, so the usable GPU count and memory per "
            "GPU are uncertain."
        )
        count = None
    elif visible:
        sizes = [g.total_bytes for g in visible]
        if None in sizes:
            warnings.append("Memory size could not be read for every GPU.")
        else:
            smallest = min(sizes)
        if len({(g.name, g.total_bytes) for g in visible}) > 1:
            warnings.append(
                "The GPUs are not all the same; the smallest memory size is shown."
            )

    if slurm and slurm.gpus_on_node is not None and count is not None:
        if count == 0 and slurm.gpus_on_node > 0:
            warnings.append(
                f"Slurm lists {slurm.gpus_on_node} GPUs on this node, but none were "
                "detected here. You may be outside the compute node."
            )
            count = None
        elif count != slurm.gpus_on_node:
            warnings.append(
                f"Slurm lists {slurm.gpus_on_node} GPUs on this node, but {count} "
                "are visible here."
            )

    # Direct detection sees one node; extend to the whole job.
    if slurm and gpus is None:
        if slurm.nodes is None:
            count = None
        elif slurm.nodes > 1 and count:
            warnings.append(
                f"Only this node was checked. The total assumes all {slurm.nodes} "
                "nodes have the same GPUs."
            )
            count *= slurm.nodes
        if count is not None and slurm.job_gpus not in (None, count):
            warnings.append(
                f"Slurm lists {slurm.job_gpus} GPUs for the job, which differs from "
                f"the {count} worked out here."
            )

    count_source = "detected"
    if gpus is not None:
        count, count_source = gpus, "override"

    vram_source = "detected" if smallest is not None else None
    if vram is not None:
        if count == 0:
            warnings.append("--vram has no effect with no GPUs; also pass --gpus.")
            smallest, vram_source = None, None
        else:
            smallest, vram_source = round(vram * GB), "override"
    if count == 0:
        smallest, vram_source = None, None
    elif count and smallest is None:
        warnings.append("GPU memory size is unknown; pass --vram.")

    if ram is not None:
        ram_total, ram_available, ram_source = round(ram * GB), None, "override"
    else:
        ram_total, ram_available = _detect_ram()
        ram_source = "detected" if ram_total is not None else None
        if ram_total is None:
            warnings.append("System RAM could not be detected; pass --ram.")

    hardware = Hardware(
        gpu_count=count,
        gpu_count_source=count_source,
        vram_bytes=smallest,
        vram_source=vram_source,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        ram_source=ram_source,
        probe=probe,
        visible=visible,
        cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
        slurm=slurm,
        warnings=(),
    )
    return replace(hardware, warnings=tuple(warnings))
