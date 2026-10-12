"""Emit a standalone Kaggle CPU smoke test for M01 attention."""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
base = (root / "src/m00_model.py").read_text(encoding="utf-8")
model = (root / "src/m01_model.py").read_text(encoding="utf-8")
model = model.replace("from m00_model import TARGETS, masked_bce, predict_proba  # noqa: F401", "")
assert "from m00_model import" not in model
checks = '''
import json
torch.set_num_threads(2)
torch.manual_seed(0)
a = M00().state_dict()
torch.manual_seed(0)
net = M01().eval()
b = net.state_dict()
assert set(b) - set(a) == {"queries"}
assert all(torch.equal(a[k], b[k]) for k in a)
with torch.no_grad(): net.queries.normal_()
g = torch.Generator().manual_seed(1)
x = torch.rand(2, 3, 4, 3, 32, 32, generator=g)
mask = torch.zeros(2, 3, 4, dtype=torch.bool)
mask[0, 0, :3] = True
mask[1, 1, :2] = True
mask[1, 2, :4] = True
ref = predict_proba(net, x, mask)
x_pad = x.clone(); x_pad[~mask] = float("nan")
assert torch.equal(ref, predict_proba(net, x_pad, mask))
with torch.no_grad():
    for i in range(2):
        f = net.encoder(x[i][mask[i]])
        weight = (net.queries @ f.T / 512 ** 0.5).softmax(-1)
        expected = torch.sigmoid(((weight @ f) * net.head.weight).sum(-1) + net.head.bias)
        assert torch.allclose(expected, ref[i], atol=1e-5)
solo = torch.cat([predict_proba(net, x[i:i+1], mask[i:i+1]) for i in range(2)])
assert torch.allclose(ref, solo, atol=1e-5)
x_other = x.clone(); x_other[1] = torch.rand_like(x_other[1])
assert torch.allclose(ref[0], predict_proba(net, x_other, mask)[0], atol=1e-5)
bad_mask = mask.clone(); bad_mask[1] = False
try: net(x, bad_mask); raise AssertionError("empty study accepted")
except ValueError: pass
torch.manual_seed(0)
train_model = M01().train()
logits = train_model(x, mask)
assert logits.shape == (2, 12)
loss = masked_bce(logits, torch.ones(2, 12), torch.ones(2, 12, dtype=torch.bool))
loss.backward()
assert torch.isfinite(loss) and train_model.queries.grad.abs().sum() > 0
assert all(torch.isfinite(p.grad).all() for p in train_model.parameters() if p.grad is not None)
print(json.dumps({"status": "passed", "attention_mask": True, "batch_isolation": True,
                  "manual_attention": True, "finite_gradients": True, "init_match": True}))
'''
output = root / "notebooks/m01_cpu_check/check.py"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(base + "\n" + model + "\n" + checks, encoding="utf-8")
print(output)
