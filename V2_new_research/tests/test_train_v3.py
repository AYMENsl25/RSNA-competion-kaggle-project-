"""Plain-assert checks for src/train_v3.py on the pilot cache (outputs/cache_v3/pilot).
Run: .venv/Scripts/python tests/test_train_v3.py  (needs CUDA)."""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_v3 as T  # noqa: E402

PILOT = [str(ROOT / "outputs/cache_v3/pilot")]


def main():
    split = T.read_split()
    rows = T.read_index(PILOT)
    studies = T.build_studies(rows, split)
    assert len(studies) == 20 and all(1 <= len(r) <= 3 for r in studies.values())

    # excluded / non-dev studies never load, whatever the index says
    a, b = sorted(studies)[:2]
    sealed = next(u for u, (_, p) in split.items() if p == "excluded")
    rows2 = [dict(r, exclude="1") if r["study"] == a else r for r in rows]
    split2 = dict(split, **{b: (split[b][0], "excluded")})
    rows2 += [dict(rows[0], study=sealed, partition="excluded", exclude="0")]
    s2 = T.build_studies(rows2, split2)
    assert a not in s2 and b not in s2 and sealed not in s2 and len(s2) == 18
    assert all(split[u][1] in T.PARTS for u in studies)

    # mirror twice = identity; sagittal reverses slices, others flip columns
    vol = np.load(PILOT[0] + "/slices.npy", mmap_mode="r")[:5]
    for plane in ("Sagittal", "Coronal", "Axial"):
        assert np.array_equal(T.mirror(T.mirror(vol, plane), plane), vol)
    assert np.array_equal(T.mirror(vol, "Sagittal")[0], vol[-1])
    assert np.array_equal(T.mirror(vol, "Axial")[:, :, 0], vol[:, :, -1])

    # dataset shapes, determinism, mirror flag only changes L studies
    items = [(u, studies[u]) for u in sorted(studies)]
    ds = T.CacheDS(PILOT, items, 4, True)
    i, x, m = ds[(0, 1)]
    assert x.shape == (3, 4, 3, 256, 256) and x.dtype == torch.uint8 and m.shape == (3, 4) and m.any()
    assert torch.equal(ds[(0, 1)][1], x) and not torch.equal(ds[(0, 2)][1], x)
    ev = T.CacheDS(PILOT, items, 16, False)
    evm = T.CacheDS(PILOT, items, 16, False, mirror_left=True)
    for j, (u, roles) in enumerate(items):
        same = torch.equal(ev[j][1], evm[j][1])
        assert same == all(r[3] != "L" for r in roles.values()), u

    # --slab: deterministic evenly spaced windows, identical across epochs and equal to eval windows
    sl = T.CacheDS(PILOT, items, 12, True, slab=True)
    e12 = T.CacheDS(PILOT, items, 12, False)
    xs, ms = sl[(0, 1)][1:]
    assert xs.shape == (3, 12, 3, 256, 256) and torch.equal(xs, sl[(0, 5)][1]) and torch.equal(xs, e12[0][1])
    shard, start, length, _, _ = next(iter(items[0][1].values()))
    assert int(ms[0].sum()) == min(12, length - 2)

    # attn head: shapes, mask invariance, plane bias, max head == P00 state_dict
    cpu = torch.device("cpu")
    att = T.V3("attn").eval()
    xa = T.prep(torch.stack([sl[(k, 1)][1] for k in range(2)]), 64)
    ma = torch.stack([sl[(k, 1)][2] for k in range(2)])
    with torch.no_grad():
        la = att(xa, ma)
        assert la.shape == (2, 12) and torch.isfinite(la).all()
        xa2 = xa.clone()
        xa2[~ma] = 7.0  # padded windows must not matter
        assert torch.allclose(att(xa2, ma), la, atol=1e-5)
        att.plane_bias[1, 3] = 5.0
        assert not torch.allclose(att(xa, ma)[:, 3], la[:, 3]) and torch.allclose(att(xa, ma)[:, 0], la[:, 0])
    assert set(T.V3("max").state_dict()) == set(T.make_model(cpu).state_dict())
    assert {"att_v.weight", "plane_bias"} <= set(T.make_model(cpu, "attn").state_dict())

    # one train step on GPU, P00 model, normalised 224 input
    dev = torch.device("cuda")
    model = T.make_model(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda")
    xb = torch.stack([ds[(k, 1)][1] for k in range(2)]).to(dev)
    mb = torch.stack([ds[(k, 1)][2] for k in range(2)]).to(dev)
    xp = T.prep(xb, 224)
    assert xp.shape == (2, 3, 4, 3, 224, 224) and abs(float(xp.mean())) < 3
    before = model.head.weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(xp, mb)
    assert logits.shape == (2, 12)
    t, o = T.targets(np.array([[0.2, 0.5, 0.8] * 4] * 2), "soft")
    assert o.sum() == 16 and t[0, 2] == np.float32(0.8)
    loss = F.binary_cross_entropy_with_logits(logits.float(), torch.from_numpy(t).to(dev),
                                              reduction="none")[torch.from_numpy(o).to(dev)].mean()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    assert torch.isfinite(loss) and not torch.equal(before, model.head.weight)
    # convnext_tiny backbone: pretrained encoder loaded, one fp16 step, round-trips through V3(backbone=)
    cm = T.make_model(dev, "max", "convnext_tiny")
    copt = torch.optim.AdamW(cm.parameters(), lr=1e-4)
    with torch.autocast("cuda", dtype=torch.float16):
        cl = cm(xp, mb)
    assert cl.shape == (2, 12) and cm.head.in_features == 768
    cb = cm.head.weight.detach().clone()
    closs = F.binary_cross_entropy_with_logits(cl.float(), torch.from_numpy(t).to(dev))
    closs.backward()  # no scaler: a first-step overflow skip would make the check flaky
    copt.step()
    assert torch.isfinite(closs) and not torch.equal(cb, cm.head.weight)
    T.V3("max", backbone="convnext_tiny").load_state_dict(cm.state_dict())  # strict: inference path loads it
    assert T.roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.4, 0.8]) == 0.875
    print("test_train_v3: all passed")


if __name__ == "__main__":
    main()


def test_augment():
    import torch
    from train_v3 import augment
    x = torch.randn(2, 3, 4, 3, 32, 32)
    assert torch.allclose(augment(x, 0, 0, 0, 0), x, atol=1e-5)  # zero strength = identity
    y = augment(x.clone().fill_(1.0))
    assert y.shape == x.shape and torch.isfinite(y).all()
    # one transform per study-plane: identical windows stay identical after aug
    z = x[:, :, :1].expand(-1, -1, 4, -1, -1, -1).contiguous()
    a = augment(z)
    assert torch.allclose(a[:, :, 0], a[:, :, 3], atol=1e-5)


if __name__ == "__main__":
    test_augment()
    print("test_augment OK")


def test_slots():
    """CPU only: --slots 3 unchanged, --slots 6 = plane x fat-sat with missing slots masked."""
    split, rows = T.read_split(), T.read_index(PILOT)
    s3, s6 = T.build_studies(rows, split), T.build_studies(rows, split, 6)
    assert T.build_studies(rows, split, 3) == s3 and set(s3) == set(s6)
    ok = {(r["study"], r["series"]): r for r in rows if r["ok"] == "1" and int(r["length"]) >= 3}
    for u in s6:
        for (plane, fs), v in s6[u].items():
            r = next(r for (st, _), r in ok.items() if st == u and int(r["start"]) == v[1] and r["shard"] == v[0])
            assert r["plane"] == plane and int(r["fatsat"]) == fs
        assert all(any(k[0] == p and s6[u][k] == s3[u][p] for k in s6[u]) for p in s3[u])  # 3-slot pick is one of the 6
    items = [(u, s6[u]) for u in sorted(s6)]
    ds = T.CacheDS(PILOT, items, 4, True, slots=6)
    for j, (u, sl) in enumerate(items):
        x, m = ds[(j, 1)][1:]
        assert x.shape == (6, 4, 3, 256, 256)
        for k, slot in enumerate(T.slot_names(6)):
            assert m[k].any() == (slot in sl) and (slot in sl or not x[k].any())
    assert any(len(v) < 6 for v in s6.values())  # pilot has missing slots
    x, m = ds[(0, 1)][1:]
    with torch.no_grad():
        assert T.V3("max")(T.prep(x[None], 64), m[None]).shape == (1, 12)  # CPU forward, 6 slots
