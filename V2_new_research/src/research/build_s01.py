"""Build the S01 multi-series coverage experiment without touching P00/D01 sources.

Writes only notebooks/s01_full_train, notebooks/s01_cpu_check and notebooks/s01_gpu_pilot.
"""

from __future__ import annotations

import json

from build_d01 import ROOT, digest, once, read


def main() -> None:
    src = ROOT / "src"
    m00_data, s01_data, model, p00 = (src / n for n in ("m00_data.py", "s01_data.py", "m00_model.py", "p00_model.py"))

    folder = ROOT / "notebooks" / "s01_full_train"
    folder.mkdir(parents=True, exist_ok=True)
    base = read(ROOT / "notebooks" / "p00_full_train" / "train_template.py")
    base = once(base, "P00 full training:", "S01 multi-series coverage:")
    for old in ("p00_full_train_summary", "p00_best.pt", "p00_val_predictions_epoch"):
        assert old in base, old
        base = base.replace(old, "s01" + old[3:])
    base = once(base,
                '"model": "P00 ImageNet-pretrained ResNet18 2.5D (3 adjacent slices), shared max over valid windows",',
                '"model": "S01 P00 ImageNet-pretrained ResNet18 2.5D, up to two series per plane sharing the '
                'P00 window budget; shared max",\n    "series_per_role": SERIES_PER_ROLE,')
    base = once(base, """            for role, sid in select_series(self.by_study.get(uid, [])).items():
                if sid is None:
                    continue
                paths = sorted((self.data / "train_series" / uid / sid).glob("*.dcm"))
                roles[role] = load_series([pydicom.dcmread(p) for p in paths])
""", """            # Every selected series must decode; any failure fails the study (no substitution).
            for role, sids in select_series_multi(self.by_study.get(uid, [])).items():
                roles[role] = [load_series([pydicom.dcmread(p) for p in
                                            sorted((self.data / "train_series" / uid / sid).glob("*.dcm"))])
                               for sid in sids]
""")
    base = once(base, "x, mask = build_study(roles, train=self.train, rng=rng)",
                "x, mask = build_study_multi(roles, train=self.train, rng=rng)")
    # Kaggle may signal the process during notebook cleanup after all epochs have
    # been saved. Keep the completed summary intact when that happens.
    base = once(base,
                '    summary["status"], summary["completed"] = "incomplete", False\n'
                '    summary["stop_reason"] = "external_watchdog_deadline"',
                '    if summary.get("completed") and summary.get("epochs_completed") == MAX_EPOCHS:\n'
                '        save()\n'
                '        raise SystemExit(0)\n'
                '    summary["status"], summary["completed"] = "incomplete", False\n'
                '    summary["stop_reason"] = "external_watchdog_deadline"')
    template = folder / "train_template.py"
    template.write_text(base, encoding="utf-8")

    hashes = {path.name: digest(path) for path in (m00_data, s01_data, model, p00, template)}
    code = once(base, "# __SOURCE_SHA256__", f"SOURCE_SHA256 = {hashes!r}")
    s01_body = once(read(s01_data),
                    "from m00_data import INFER_WINDOWS, ROLES, SIZE, TRAIN_WINDOWS, StudyError, series_windows\n", "")
    code = once(code, "# __M00_DATA_SOURCE__",
                f"# Embedded m00_data.py SHA256: {digest(m00_data)}\n{read(m00_data)}\n"
                f"# Embedded s01_data.py SHA256: {digest(s01_data)}\n{s01_body}")
    code = once(code, "# __M00_MODEL_SOURCE__", f"# Embedded m00_model.py SHA256: {digest(model)}\n{read(model)}")
    code = once(code, "# __P00_MODEL_SOURCE__",
                f"# Embedded p00_model.py SHA256: {digest(p00)}\n{once(read(p00), 'from m00_model import M00' + chr(10), '')}")
    (folder / "train.py").write_text(code, encoding="utf-8")
    base_meta = json.loads(read(ROOT / "notebooks" / "p00_full_train" / "kernel-metadata.json"))
    base_meta.update(id="aymenslimani/rsna-knee-v2-s01-full-train", title="RSNA knee V2 S01 full train")
    (folder / "kernel-metadata.json").write_text(json.dumps(base_meta, indent=2) + "\n", encoding="utf-8")

    marker = 'if __name__ == "__main__":\n'
    assert code.count(marker) == 1
    head = code.split(marker, 1)[0]
    cpu_suffix = '''    assert self_check()
    data, by_study, train_uids, val_uids, *_ = prepare()
    model = P00()
    model.eval()
    x = torch.rand((1, 3, 1, 3, 32, 32), dtype=torch.float32)
    mask = torch.ones((1, 3, 1), dtype=torch.bool)
    logits = model(x, mask)
    assert logits.shape == (1, len(TARGETS)) and torch.isfinite(logits).all()
    loss = masked_bce(logits, torch.zeros_like(logits), torch.ones_like(logits, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(model.encoder.conv1.weight.grad).all()
    sampled = {"train": 0, "validation": 0, "selected_series": 0, "valid_windows": 0,
               "failed_studies": 0}
    for phase, uids, training in (("train", train_uids, True), ("validation", val_uids, False)):
        rank = lambda uid: (-sum(len(select_series_multi(by_study[uid])[r]) == 2 for r in ROLES),
                            h64("s01-cpu:" + uid))
        ordered = sorted(range(len(uids)), key=lambda i: rank(uids[i]))
        # Include two low-coverage studies as a one-series-path check.
        chosen = ordered[:8] + ordered[-2:]
        ds = StudyDS(data, uids, by_study, training, 1)
        for i in chosen:
            item = ds[i]
            sampled[phase] += 1
            if not item["ok"]:
                sampled["failed_studies"] += 1
                continue
            expected = TRAIN_WINDOWS if training else INFER_WINDOWS
            assert tuple(item["x"].shape) == (3, expected, 3, SIZE, SIZE)
            assert tuple(item["mask"].shape) == (3, expected)
            sampled["selected_series"] += sum(len(v) for v in select_series_multi(by_study[uids[i]]).values())
            sampled["valid_windows"] += int(item["mask"].sum())
    assert sampled["train"] == sampled["validation"] == 10 and sampled["failed_studies"] == 0
    print(json.dumps({"status": "passed", "source_sha256": SOURCE_SHA256,
                      "weight_sha256": WEIGHT_SHA256, "inputs": summary["inputs"],
                      "real_dicom_sample": sampled, "encoder_grad_finite": True}), flush=True)
'''
    cpu_folder = ROOT / "notebooks" / "s01_cpu_check"
    cpu_folder.mkdir(parents=True, exist_ok=True)
    (cpu_folder / "check.py").write_text(head + marker + cpu_suffix, encoding="utf-8")
    cpu_meta = dict(base_meta, id="aymenslimani/rsna-knee-v2-s01-cpu-preflight",
                    title="RSNA knee V2 S01 CPU preflight", code_file="check.py", enable_gpu="false")
    (cpu_folder / "kernel-metadata.json").write_text(json.dumps(cpu_meta, indent=2) + "\n", encoding="utf-8")

    pilot_marker = "# Pilot stops no later than one hour; finish work by 55 minutes."
    pilot_source = read(ROOT / "notebooks" / "p00_gpu_pilot" / "pilot.py")
    assert pilot_source.count(pilot_marker) == 1
    pilot_suffix = once(pilot_marker + pilot_source.split(pilot_marker, 1)[1],
                        "p00_gpu_pilot_summary.json", "s01_gpu_pilot_summary.json")
    pilot_suffix = once(pilot_suffix, "def pilot():", '''def pilot_indices(uids, by_study, n, rich_n):
    # Include series-rich studies, then fill by the frozen hash order.
    richness = lambda uid: sum(len(v) for v in select_series_multi(by_study[uid]).values())
    rich = sorted(range(len(uids)), key=lambda i: (-richness(uids[i]), h64(uids[i])))[:rich_n]
    rest = sorted((i for i in range(len(uids)) if i not in set(rich)),
                  key=lambda i: h64(uids[i]))[:n-rich_n]
    return rich + rest


def pilot():''')
    pilot_suffix = once(pilot_suffix,
                        'train_idx = sorted(range(len(train_uids)), key=lambda i: h64(train_uids[i]))[:PILOT_TRAIN]',
                        'train_idx = pilot_indices(train_uids, by_study, PILOT_TRAIN, 20)')
    pilot_suffix = once(pilot_suffix,
                        'val_idx = sorted(range(len(val_uids)), key=lambda i: h64(val_uids[i]))[:PILOT_VAL]',
                        'val_idx = pilot_indices(val_uids, by_study, PILOT_VAL, 10)')
    pilot_suffix = once(pilot_suffix,
                        'if stopped or tr["studies_failed"] or tr["optimizer_steps_succeeded"] == 0:',
                        'if stopped or tr["studies_failed"] or tr["optimizer_steps_succeeded"] == 0 '
                        'or tr.get("amp_overflow_gate_failed"):')
    pilot_suffix = once(pilot_suffix, '"train_studies": PILOT_TRAIN, "train_seconds": train_s,',
                        '"train_studies": PILOT_TRAIN, "selected_train_series": '
                        'sum(sum(len(v) for v in select_series_multi(by_study[u]).values()) for u in tr_u), '
                        '"train_seconds": train_s, "optimizer_calls": tr["optimizer_calls"], '
                        '"amp_skipped_steps": tr["amp_skipped_steps"],')
    pilot_suffix = once(pilot_suffix, '"validation_studies": val_seen, "validation_seconds": val_s,',
                        '"validation_studies": val_seen, "selected_validation_series": '
                        'sum(sum(len(v) for v in select_series_multi(by_study[u]).values()) for u in va_u), '
                        '"validation_seconds": val_s,')
    pilot_folder = ROOT / "notebooks" / "s01_gpu_pilot"
    pilot_folder.mkdir(parents=True, exist_ok=True)
    (pilot_folder / "pilot.py").write_text(head + pilot_suffix, encoding="utf-8")
    pilot_meta = dict(base_meta, id="aymenslimani/rsna-knee-v2-s01-gpu-pilot",
                      title="RSNA knee V2 S01 GPU pilot", code_file="pilot.py")
    (pilot_folder / "kernel-metadata.json").write_text(json.dumps(pilot_meta, indent=2) + "\n", encoding="utf-8")
    print(folder / "train.py")
    print(hashes)


if __name__ == "__main__":
    main()
