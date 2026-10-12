"""Embed audited V2 modules in Kaggle script kernels, which upload one code_file."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "src" / "m00_model.py"
DATA = ROOT / "src" / "m00_data.py"
TESTS = ROOT / "tests" / "test_m00.py"
M01_MODEL = ROOT / "src" / "m01_model.py"
P00_MODEL = ROOT / "src" / "p00_model.py"
P01A_MODEL = ROOT / "src" / "p01a_model.py"
M01_IMPORT = "from m00_model import TARGETS, masked_bce, predict_proba  # noqa: F401\n"


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_gpu() -> None:
    folder = ROOT / "notebooks" / "m00_gpu_pilot"
    template = source(folder / "pilot_template.py")
    assert template.count("# __M00_MODEL_SOURCE__") == 1
    code = template.replace(
        "# __M00_MODEL_SOURCE__",
        f"# Embedded M00 source SHA256: {digest(MODEL)}\n" + source(MODEL),
    )
    (folder / "pilot.py").write_text(code, encoding="utf-8")


def generate_cpu() -> None:
    folder = ROOT / "notebooks" / "m00_cpu_tests"
    template = source(folder / "check_template.py")
    assert template.count("# __EMBEDDED_TEST_FILES__") == 1
    lines = [
        'root = Path("/kaggle/working/m00_synthetic")',
        '(root / "src").mkdir(parents=True, exist_ok=True)',
        '(root / "tests").mkdir(parents=True, exist_ok=True)',
    ]
    for destination, source_path in (
        ("src/m00_model.py", MODEL),
        ("src/m00_data.py", DATA),
        ("tests/test_m00.py", TESTS),
    ):
        lines.append(f"# {destination} SHA256: {digest(source_path)}")
        lines.append(f"(root / {destination!r}).write_text({source(source_path)!r}, encoding='utf-8')")
    code = template.replace("# __EMBEDDED_TEST_FILES__", "\n".join(lines))
    (folder / "check.py").write_text(code, encoding="utf-8")


def generate_preprocess() -> None:
    folder = ROOT / "notebooks" / "m00_preprocess_pilot"
    template = source(folder / "pilot_template.py")
    assert template.count("# __M00_DATA_SOURCE__") == 1
    code = template.replace(
        "# __M00_DATA_SOURCE__",
        f"# Embedded M00 data source SHA256: {digest(DATA)}\n" + source(DATA),
    )
    (folder / "pilot.py").write_text(code, encoding="utf-8")


def generate_loader() -> None:
    folder = ROOT / "notebooks" / "m00_loader_throughput"
    template = source(folder / "throughput_template.py")
    assert template.count("# __M00_DATA_SOURCE__") == 1
    code = template.replace(
        "# __M00_DATA_SOURCE__",
        f"# Embedded M00 data source SHA256: {digest(DATA)}\n" + source(DATA),
    )
    (folder / "throughput.py").write_text(code, encoding="utf-8")


def generate_cache_pilot() -> None:
    folder = ROOT / "notebooks" / "m00_cache_pilot"
    template = source(folder / "cache_pilot_template.py")
    assert template.count("# __M00_DATA_SOURCE__") == 1
    code = template.replace(
        "# __M00_DATA_SOURCE__",
        f"# Embedded M00 data source SHA256: {digest(DATA)}\n" + source(DATA),
    )
    (folder / "cache_pilot.py").write_text(code, encoding="utf-8")


def generate_cache_alternatives() -> None:
    folder = ROOT / "notebooks" / "m00_cache_alternatives"
    template = source(folder / "cache_alternatives_template.py")
    assert template.count("# __M00_DATA_SOURCE__") == 1
    code = template.replace(
        "# __M00_DATA_SOURCE__",
        f"# Embedded M00 data source SHA256: {digest(DATA)}\n" + source(DATA),
    )
    (folder / "cache_alternatives.py").write_text(code, encoding="utf-8")


def generate_integrated() -> None:
    folder = ROOT / "notebooks" / "m00_integrated_gpu_pilot"
    code = source(folder / "pilot_template.py")
    for marker, path in (("# __M00_DATA_SOURCE__", DATA), ("# __M00_MODEL_SOURCE__", MODEL)):
        assert code.count(marker) == 1
        code = code.replace(marker, f"# Embedded {path.name} SHA256: {digest(path)}\n" + source(path))
    (folder / "pilot.py").write_text(code, encoding="utf-8")


def generate_full_train() -> None:
    folder = ROOT / "notebooks" / "m00_full_train"
    template = folder / "train_template.py"
    code = source(template)
    hashes = {path.name: digest(path) for path in (DATA, MODEL, template)}
    assert code.count("# __SOURCE_SHA256__") == 1
    code = code.replace("# __SOURCE_SHA256__", f"SOURCE_SHA256 = {hashes!r}")
    for marker, path in (("# __M00_DATA_SOURCE__", DATA), ("# __M00_MODEL_SOURCE__", MODEL)):
        assert code.count(marker) == 1
        code = code.replace(marker, f"# Embedded {path.name} SHA256: {digest(path)}\n" + source(path))
    (folder / "train.py").write_text(code, encoding="utf-8")


def generate_m01_full_train() -> None:
    """Embeds m00_model.py unchanged (TARGETS, masked_bce, predict_proba) followed by m01_model.py
    minus its single `from m00_model import` line, which the embedded M00 source satisfies."""
    folder = ROOT / "notebooks" / "m01_full_train"
    template = folder / "train_template.py"
    code = source(template)
    hashes = {path.name: digest(path) for path in (DATA, MODEL, M01_MODEL, template)}
    assert code.count("# __SOURCE_SHA256__") == 1
    code = code.replace("# __SOURCE_SHA256__", f"SOURCE_SHA256 = {hashes!r}")
    m01 = source(M01_MODEL)
    assert m01.count(M01_IMPORT) == 1
    for marker, path, text in (("# __M00_DATA_SOURCE__", DATA, source(DATA)),
                               ("# __M00_MODEL_SOURCE__", MODEL, source(MODEL)),
                               ("# __M01_MODEL_SOURCE__", M01_MODEL, m01.replace(M01_IMPORT, ""))):
        assert code.count(marker) == 1
        code = code.replace(marker, f"# Embedded {path.name} SHA256: {digest(path)}\n" + text)
    (folder / "train.py").write_text(code, encoding="utf-8")


def generate_p00_full_train() -> None:
    """Embed unchanged M00 data/model and only replace encoder initialization."""
    folder = ROOT / "notebooks" / "p00_full_train"
    template = folder / "train_template.py"
    code = source(template)
    hashes = {path.name: digest(path) for path in (DATA, MODEL, P00_MODEL, template)}
    assert code.count("# __SOURCE_SHA256__") == 1
    code = code.replace("# __SOURCE_SHA256__", f"SOURCE_SHA256 = {hashes!r}")
    p00 = source(P00_MODEL)
    import_line = "from m00_model import M00\n"
    assert p00.count(import_line) == 1
    for marker, path, body in (("# __M00_DATA_SOURCE__", DATA, source(DATA)),
                               ("# __M00_MODEL_SOURCE__", MODEL, source(MODEL)),
                               ("# __P00_MODEL_SOURCE__", P00_MODEL, p00.replace(import_line, ""))):
        assert code.count(marker) == 1
        code = code.replace(marker, f"# Embedded {path.name} SHA256: {digest(path)}\n" + body)
    (folder / "train.py").write_text(code, encoding="utf-8")


def generate_e01_six_epoch() -> None:
    """Build the isolated six-epoch P00 duration experiment with embedded source hashes."""
    folder = ROOT / "notebooks" / "e01_p00_six_epoch"
    template = folder / "train_template.py"
    code = source(template)
    hashes = {path.name: digest(path) for path in (DATA, MODEL, P00_MODEL, template)}
    assert code.count("# __SOURCE_SHA256__") == 1
    code = code.replace("# __SOURCE_SHA256__", f"SOURCE_SHA256 = {hashes!r}")
    p00 = source(P00_MODEL)
    import_line = "from m00_model import M00\n"
    assert p00.count(import_line) == 1
    for marker, path, body in (("# __M00_DATA_SOURCE__", DATA, source(DATA)),
                               ("# __M00_MODEL_SOURCE__", MODEL, source(MODEL)),
                               ("# __P00_MODEL_SOURCE__", P00_MODEL, p00.replace(import_line, ""))):
        assert code.count(marker) == 1
        code = code.replace(marker, f"# Embedded {path.name} SHA256: {digest(path)}\n" + body)
    (folder / "train.py").write_text(code, encoding="utf-8")


def generate_p00_cpu_check() -> None:
    """Build an offline CPU gate from the exact generated P00 train source."""
    folder = ROOT / "notebooks" / "p00_cpu_check"
    folder.mkdir(parents=True, exist_ok=True)
    train = source(ROOT / "notebooks" / "p00_full_train" / "train.py")
    marker = 'if __name__ == "__main__":\n'
    assert train.count(marker) == 1
    code = train.split(marker)[0] + '''\n\nif __name__ == "__main__":
    assert self_check()
    prepare()
    model = P00()
    model.eval()
    x = torch.rand((1, 3, 1, 3, 32, 32), dtype=torch.float32)
    mask = torch.ones((1, 3, 1), dtype=torch.bool)
    logits = model(x, mask)
    assert logits.shape == (1, len(TARGETS)) and torch.isfinite(logits).all()
    targets = torch.zeros_like(logits)
    observed = torch.ones_like(logits, dtype=torch.bool)
    loss = masked_bce(logits, targets, observed)
    loss.backward()
    assert torch.isfinite(loss) and model.encoder.conv1.weight.grad is not None
    print(json.dumps({"status": "passed", "source_sha256": SOURCE_SHA256,
                      "weight_sha256": WEIGHT_SHA256, "weight_file": WEIGHT_FILE,
                      "inputs": summary["inputs"], "logits_shape": list(logits.shape),
                      "encoder_grad_finite": bool(torch.isfinite(model.encoder.conv1.weight.grad).all())}), flush=True)
'''
    (folder / "check.py").write_text(code, encoding="utf-8")


def generate_p00_gpu_pilot() -> None:
    """Matched real-data P00 throughput pilot, with a one-hour watchdog."""
    folder = ROOT / "notebooks" / "p00_gpu_pilot"
    folder.mkdir(parents=True, exist_ok=True)
    train = source(ROOT / "notebooks" / "p00_full_train" / "train.py")
    marker = 'if __name__ == "__main__":\n'
    assert train.count(marker) == 1
    code = train.split(marker)[0] + '''\n\n# Pilot stops no later than one hour; finish work by 55 minutes.
HARD_CAP_S, STOP_AT_S = 3600.0, 3300.0
PILOT_TRAIN, PILOT_VAL = 80, 40

def pilot():
    signal.signal(signal.SIGTERM, deadline_handler)
    watchdog = start_watchdog()
    assert self_check() and torch.cuda.is_available()
    seed_all(SEED)
    data, by_study, train_uids, val_uids, tr_t, tr_o, va_t, va_o = prepare()
    # Frozen eligible UID lists only; no expert studies or labels are read.
    train_idx = sorted(range(len(train_uids)), key=lambda i: h64(train_uids[i]))[:PILOT_TRAIN]
    val_idx = sorted(range(len(val_uids)), key=lambda i: h64(val_uids[i]))[:PILOT_VAL]
    tr_u, va_u = [train_uids[i] for i in train_idx], [val_uids[i] for i in val_idx]
    tr_t, tr_o = tr_t[train_idx], tr_o[train_idx]
    dev = torch.device("cuda")
    model = P00().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS * math.ceil(len(train_uids) / TRAIN_BS))
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    train_ds = StudyDS(data, tr_u, by_study, True, 1)
    train_start = time.perf_counter()
    tr, stopped = train_epoch(model, opt, sched, scaler, train_ds, tr_t, tr_o, 1, dev)
    torch.cuda.synchronize()
    train_s = time.perf_counter() - train_start
    if stopped or tr["studies_failed"] or tr["optimizer_steps_succeeded"] == 0:
        raise RuntimeError("pilot training gate failed")
    val_ds = StudyDS(data, va_u, by_study, False)
    val_start = time.perf_counter()
    model.eval()
    val_seen = val_failed = 0
    with torch.no_grad():
        for b in make_loader(val_ds, VAL_BS, False, SEED):
            val_seen += b["n"]
            val_failed += sum(b["errs"].values())
            if b["idx"]:
                check_batch(b, INFER_WINDOWS)
                with torch.autocast("cuda", dtype=torch.float16):
                    z = model(b["x"].to(dev, non_blocking=True), b["mask"].to(dev, non_blocking=True)).float()
                if z.shape != (len(b["idx"]), len(TARGETS)) or not torch.isfinite(z).all():
                    raise FloatingPointError("pilot validation logits invalid")
            if elapsed() > STOP_AT_S:
                raise TimeoutError("pilot time cap during validation")
    torch.cuda.synchronize()
    val_s = time.perf_counter() - val_start
    if val_seen != PILOT_VAL or val_failed:
        raise RuntimeError("pilot validation gate failed")
    projection = 1.2 * MAX_EPOCHS * (train_s / PILOT_TRAIN * len(train_uids) + val_s / PILOT_VAL * len(val_uids))
    result = {"status": "passed" if projection < 8*3600 - 30*60 and elapsed() < HARD_CAP_S else "failed",
              "source_sha256": SOURCE_SHA256, "weight_sha256": WEIGHT_SHA256,
              "train_studies": PILOT_TRAIN, "train_seconds": train_s, "train_failed": tr["studies_failed"],
              "optimizer_steps_succeeded": tr["optimizer_steps_succeeded"],
              "validation_studies": val_seen, "validation_seconds": val_s, "validation_failed": val_failed,
              "projected_three_epoch_seconds_with_20pct_headroom": projection,
              "full_run_stop_at_seconds": 8*3600 - 30*60,
              "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
              "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
              "gpu": torch.cuda.get_device_name(0), "script_seconds": elapsed(),
              "inputs": summary["inputs"]}
    (OUT / "p00_gpu_pilot_summary.json").write_text(json.dumps(result, indent=2) + "\\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise RuntimeError("P00 projected full runtime gate failed")

if __name__ == "__main__":
    pilot()
'''
    (folder / "pilot.py").write_text(code, encoding="utf-8")


def generate_p01a_full_train() -> None:
    """Embed unchanged M00 data/model and p01a_model.py minus its `from m00_model import` line."""
    folder = ROOT / "notebooks" / "p01a_full_train"
    template = folder / "train_template.py"
    code = source(template)
    hashes = {path.name: digest(path) for path in (DATA, MODEL, P01A_MODEL, template)}
    assert code.count("# __SOURCE_SHA256__") == 1
    code = code.replace("# __SOURCE_SHA256__", f"SOURCE_SHA256 = {hashes!r}")
    p01a = source(P01A_MODEL)
    import_line = "from m00_model import M00, TARGETS\n"
    assert p01a.count(import_line) == 1
    for marker, path, body in (("# __M00_DATA_SOURCE__", DATA, source(DATA)),
                               ("# __M00_MODEL_SOURCE__", MODEL, source(MODEL)),
                               ("# __P01A_MODEL_SOURCE__", P01A_MODEL, p01a.replace(import_line, ""))):
        assert code.count(marker) == 1
        code = code.replace(marker, f"# Embedded {path.name} SHA256: {digest(path)}\n" + body)
    assert "from m00_model" not in code and "import m00_" not in code
    (folder / "train.py").write_text(code, encoding="utf-8")


if __name__ == "__main__":
    generate_p01a_full_train()
    generate_p00_full_train()
    generate_p00_cpu_check()
    generate_p00_gpu_pilot()
    generate_m01_full_train()
    generate_full_train()
    generate_integrated()
    generate_gpu()
    generate_cpu()
    generate_preprocess()
    generate_loader()
    generate_cache_pilot()
    generate_cache_alternatives()
