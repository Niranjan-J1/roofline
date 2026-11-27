import subprocess
import sys
from types import SimpleNamespace

import pytest

from roofline import env
from roofline.env import GPU, MIB, Probe, detect_hardware, detect_slurm, visible_gpus
from roofline.report import render_hardware

GIB = 1024 * MIB


def gpu(i, gib=24, name="NVIDIA A", mig=False):
    return GPU(i, f"GPU-{i}{i}{i}{i}", name, gib * GIB, (gib - 1) * GIB, mig)


@pytest.fixture(autouse=True)
def fake_system(monkeypatch):
    """No test depends on the real machine. Tests may replace these."""
    monkeypatch.setattr(env, "_detect_ram", lambda: (32 * GIB, 10 * GIB))
    monkeypatch.setattr(env, "probe_gpus", lambda: Probe("ok", "test", ()))


def use(monkeypatch, *gpus, status="ok", detail="test"):
    monkeypatch.setattr(env, "probe_gpus", lambda: Probe(status, detail, gpus))


def warned(h, text):
    return any(text in w for w in h.warnings)


# --- the two detection methods ----------------------------------------------

REAL_PROBE = env.probe_gpus


def fake_nvml(**overrides):
    functions = dict(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetMemoryInfo=lambda h: SimpleNamespace(total=100, free=60),
        nvmlDeviceGetMigMode=lambda h: [0, 0],
        nvmlDeviceGetUUID=lambda h: f"GPU-{h}".encode(),
        nvmlDeviceGetName=lambda h: b"NVIDIA T",
    )
    return SimpleNamespace(**{**functions, **overrides})


def raiser(error):
    def fail(*args, **kwargs):
        raise error

    return fail


def fake_smi(monkeypatch, *results):
    queue = list(results)
    monkeypatch.setattr(env.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        env.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            **dict(zip(("returncode", "stdout", "stderr"), queue.pop(0)))
        ),
    )


def test_nvml_is_used_first(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", fake_nvml())
    probe = REAL_PROBE()
    assert probe.status == "ok" and probe.detail == "NVIDIA Python library"
    assert probe.gpus[1] == GPU(1, "GPU-1", "NVIDIA T", 100, 60)


def test_falls_back_to_smi(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)  # makes the import fail
    row = "0, GPU-abc, NVIDIA B, 24564, 24000, Disabled\n"
    fake_smi(monkeypatch, (0, row, ""))
    probe = REAL_PROBE()
    assert probe.detail == "nvidia-smi"
    assert probe.gpus == (GPU(0, "GPU-abc", "NVIDIA B", 24564 * MIB, 24000 * MIB),)


def test_smi_retries_without_mig_field_and_handles_na(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    fake_smi(
        monkeypatch,
        (2, 'Field "mig.mode.current" is not a valid field to query.', ""),
        (0, "0, GPU-abc, NVIDIA B, [N/A], [N/A]\n", ""),
    )
    only = REAL_PROBE().gpus[0]
    assert only.total_bytes is None and only.free_bytes is None and not only.mig


def test_missing_tools_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(env.shutil, "which", lambda name: None)
    assert REAL_PROBE() == Probe("unavailable", "nvidia-smi not found")


@pytest.mark.parametrize(
    "result, status",
    [
        ((9, "driver is not responding", ""), "failed"),
        ((6, "No devices were found", ""), "ok"),
        ((0, "garbage", ""), "failed"),
    ],
)
def test_smi_failure_differs_from_no_devices(monkeypatch, result, status):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    fake_smi(monkeypatch, result, result)
    probe = REAL_PROBE()
    assert probe.status == status and probe.gpus == ()


def test_smi_timeout_does_not_crash(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(env.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        env.subprocess, "run", raiser(subprocess.TimeoutExpired("nvidia-smi", 15))
    )
    assert REAL_PROBE().status == "failed"


def test_nvml_error_is_reported_when_smi_is_missing(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "pynvml", fake_nvml(nvmlInit=raiser(RuntimeError("no driver")))
    )
    monkeypatch.setattr(env.shutil, "which", lambda name: None)
    probe = REAL_PROBE()
    assert probe.status == "failed" and "no driver" in probe.detail


# --- combining results -------------------------------------------------------


def test_no_gpu_versus_failed_check(monkeypatch):
    use(monkeypatch, status="unavailable", detail="nvidia-smi not found")
    h = detect_hardware(env={})
    assert h.gpu_count == 0 and "assuming CPU-only" in render_hardware(h, False)
    use(monkeypatch, status="failed", detail="broken")
    h = detect_hardware(env={})
    assert h.gpu_count is None and warned(h, "Could not check for GPUs (broken)")
    assert "count unknown" in render_hardware(h, False)


def test_mixed_gpus_use_the_smallest(monkeypatch):
    use(monkeypatch, gpu(0, 24), gpu(1, 16, "NVIDIA B"))
    h = detect_hardware(env={})
    assert (h.gpu_count, h.vram_bytes) == (2, 16 * GIB)
    assert warned(h, "not all the same")


@pytest.mark.parametrize(
    "setting, indices",
    [
        ("1", [1]),
        ("2,0", [2, 0]),
        ("", []),
        ("-1", []),
        ("0,-1,1", [0]),
        ("GPU-1111", [1]),
        ("0,7,1", [0]),
    ],
)
def test_visible_devices(setting, indices):
    chosen, _ = visible_gpus(
        (gpu(0), gpu(1), gpu(2)), {"CUDA_VISIBLE_DEVICES": setting}
    )
    assert [g.index for g in chosen] == indices


@pytest.mark.parametrize("setting", ["MIG-abc", "GPU-", "0,0", "banana"])
def test_unmatched_visible_devices_is_uncertain(monkeypatch, setting):
    use(monkeypatch, gpu(0), gpu(1))
    h = detect_hardware(env={"CUDA_VISIBLE_DEVICES": setting})
    assert h.gpu_count is None and warned(h, "uncertain")


def test_numbered_selection_of_mixed_gpus_warns_unless_pci_order():
    gpus = (gpu(0, 24), gpu(1, 16, "NVIDIA B"))
    settings = {"CUDA_VISIBLE_DEVICES": "0"}
    assert "uncertain" in visible_gpus(gpus, settings)[1][0]
    assert visible_gpus(gpus, {**settings, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})[1] == ()


def test_all_hidden(monkeypatch):
    use(monkeypatch, gpu(0))
    h = detect_hardware(env={"CUDA_VISIBLE_DEVICES": ""})
    assert h.gpu_count == 0 and "hidden by" in render_hardware(h, False)


def test_mig_is_uncertain(monkeypatch):
    use(monkeypatch, gpu(0, mig=True))
    h = detect_hardware(env={})
    assert h.gpu_count is None and h.vram_bytes is None and warned(h, "MIG")


# --- Slurm -------------------------------------------------------------------


def job(**values):
    base = {"SLURM_JOB_ID": "42", "SLURM_JOB_NUM_NODES": "1"}
    return {**base, **{f"SLURM_{k}": v for k, v in values.items()}}


def test_no_job_without_job_id():
    assert detect_slurm({"SLURM_NTASKS": "4"}) is None


@pytest.mark.parametrize("spec", ["4", "a100:4", "gpu:a100:4", "a100:2,v100:2"])
def test_slurm_gpu_formats(spec):
    assert detect_slurm(job(GPUS_PER_NODE=spec)).gpus_on_node == 4


def test_slurm_fields_and_memory():
    s = detect_slurm(job(NTASKS="8", GPUS_ON_NODE="4", MEM_PER_NODE="1000"))
    assert (s.nodes, s.tasks, s.gpus_on_node) == (1, 8, 4)
    assert s.mem_per_node_bytes == 1000 * MIB
    per_cpu = detect_slurm(job(MEM_PER_CPU="100", CPUS_ON_NODE="8"))
    assert per_cpu.mem_per_node_bytes == 800 * MIB
    assert detect_slurm(job(MEM_PER_NODE="0")).mem_per_node_bytes is None


def test_unreadable_slurm_values_warn():
    s = detect_slurm({"SLURM_JOB_ID": "1", "SLURM_NTASKS": "many"})
    assert s.tasks is None and s.nodes is None
    assert any("SLURM_NTASKS" in w for w in s.warnings)
    assert any("node count" in w for w in s.warnings)


def test_multi_node_total_is_labelled_an_assumption(monkeypatch):
    use(monkeypatch, gpu(0), gpu(1))
    h = detect_hardware(env=job(JOB_NUM_NODES="3", GPUS_ON_NODE="2"))
    assert h.gpu_count == 6 and warned(h, "assumes all 3 nodes")
    overridden = detect_hardware(gpus=8, env=job(JOB_NUM_NODES="3"))
    assert overridden.gpu_count == 8 and not warned(overridden, "assumes")


def test_slurm_gpus_without_local_gpus_are_not_invented(monkeypatch):
    use(monkeypatch, status="unavailable", detail="nvidia-smi not found")
    h = detect_hardware(env=job(GPUS_ON_NODE="4"))
    assert h.gpu_count is None and warned(h, "outside the compute node")


def test_slurm_count_mismatches_warn(monkeypatch):
    use(monkeypatch, gpu(0))
    h = detect_hardware(env=job(GPUS_ON_NODE="2", GPUS="4"))
    assert warned(h, "1 are visible") and warned(h, "for the job")


# --- RAM and overrides -------------------------------------------------------


def test_ram_detection_and_override(monkeypatch):
    h = detect_hardware(env={})
    assert (h.ram_total_bytes, h.ram_source) == (32 * GIB, "detected")
    h = detect_hardware(ram=64, env={})
    assert (h.ram_total_bytes, h.ram_available_bytes) == (64_000_000_000, None)
    assert "(override)" in render_hardware(h, False)
    monkeypatch.setattr(env, "_detect_ram", lambda: (None, None))
    assert warned(detect_hardware(env={}), "pass --ram")


def test_ram_errors_do_not_crash(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(env.psutil, "virtual_memory", raiser(OSError("nope")))
    assert env._detect_ram() == (None, None)


def test_simulated_hardware():
    h = detect_hardware(gpus=8, vram=80, env={})
    assert (h.gpu_count, h.vram_bytes) == (8, 80_000_000_000)
    assert (h.gpu_count_source, h.vram_source) == ("override", "override")
    assert "8 total (override), 80.00 GB each (override)" in " ".join(
        render_hardware(h, False).split()
    )


def test_gpu_count_without_memory_asks_for_vram():
    h = detect_hardware(gpus=2, env={})
    assert h.vram_bytes is None and warned(h, "pass --vram")


def test_vram_alone_does_not_invent_a_gpu():
    h = detect_hardware(vram=80, env={})
    assert h.gpu_count == 0 and h.vram_bytes is None and warned(h, "no effect")


def test_full_view_lists_gpus_and_sanitizes_text(monkeypatch):
    use(monkeypatch, gpu(0, name="bad\x1b[31m"), gpu(1))
    h = detect_hardware(env={"CUDA_VISIBLE_DEVICES": "1", **job()})
    short, full = render_hardware(h, False), render_hardware(h, True)
    assert "GPU 0" not in short and "GPU 0" in full
    assert "not confirmed visible" in full and "SLURM_JOB_ID=42" in full
    assert "\x1b" not in short + full
    assert "Fit verdicts are not available yet." in short
