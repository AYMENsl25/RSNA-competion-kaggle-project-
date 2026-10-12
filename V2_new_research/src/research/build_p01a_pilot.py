"""Matched one-hour P01A real-data GPU pilot from the P00 pilot protocol."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P01_TRAIN = ROOT / "notebooks/p01a_full_train/train.py"
P00_PILOT = ROOT / "notebooks/p00_gpu_pilot/pilot.py"
OUT = ROOT / "notebooks/p01a_gpu_pilot/pilot.py"


def main():
    train = P01_TRAIN.read_text(encoding="utf-8")
    ref = P00_PILOT.read_text(encoding="utf-8")
    marker = "# Pilot stops no later than one hour; finish work by 55 minutes."
    assert ref.count(marker) == 1
    suffix = ref[ref.index(marker):].replace("P00", "P01A").replace("p00_", "p01a_")
    # M00 make_loader returns a list of error names; the earlier P00 pilot used dict syntax.
    suffix = suffix.replace('sum(b["errs"].values())', 'len(b["errs"])')
    assert 'len(b["errs"])' in suffix and 'sum(b["errs"].values())' not in suffix
    main_marker = 'if __name__ == "__main__":\n'
    assert train.count(main_marker) == 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(train.split(main_marker)[0] + suffix, encoding="utf-8")


if __name__ == "__main__":
    main()
