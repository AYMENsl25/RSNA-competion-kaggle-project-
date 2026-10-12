"""Generate a private Kaggle CPU preflight without invoking full inference."""
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "infer.py").read_text(encoding="utf-8")
head = source.split('if __name__ == "__main__":')[0]
body = '''if __name__ == "__main__":
    for filename, digest in SOURCE_SHA256.items():
        assert hashlib.sha256(EMBEDDED_SOURCES[filename].encode("utf-8")).hexdigest() == digest
    checkpoint = exactly_one("p00_best.pt", CHECKPOINT_DIRS)
    assert sha256(checkpoint) == CHECKPOINT_SHA256
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert ckpt["source_sha256"]["m00_data.py"] == SOURCE_SHA256["m00_data.py"]
    assert ckpt["source_sha256"]["m00_model.py"] == SOURCE_SHA256["m00_model.py"]
    assert ckpt["epoch"] == 3 and ckpt["config"]["targets"] == TARGETS
    model = M00().eval()
    model.load_state_dict(ckpt["model"], strict=True)
    data = next((p for p in COMPETITION_DIRS if (p / "sample_submission.csv").is_file()), None)
    assert data is not None
    columns, uids, by_study = read_schema(data / "sample_submission.csv", data / "test.csv", data / "test_series.csv")
    synthetic = self_test()
    result = {"status": "passed", "checkpoint_sha256": CHECKPOINT_SHA256,
              "sample_rows": len(uids), "sample_columns": len(columns),
              "test_studies_with_series": len(by_study), "synthetic": synthetic}
    (OUT / "p00_cpu_preflight_summary.json").write_text(json.dumps(result, indent=2) + "\\n")
    print(json.dumps(result), flush=True)
'''
(root / "preflight.py").write_text(head + body, encoding="utf-8")
