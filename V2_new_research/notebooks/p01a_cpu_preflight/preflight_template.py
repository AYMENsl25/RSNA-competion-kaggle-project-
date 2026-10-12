"""P01A offline CPU preflight; synthetic pixels only, no patient images or labels."""
import json
import time
from pathlib import Path

import torch

# __M00_MODEL_SOURCE__
# __P01A_MODEL_SOURCE__

def main():
    t0 = time.perf_counter()
    import timm
    import safetensors
    torch.set_num_threads(2)
    model = P01A()
    x = torch.rand(1, 1, 2, 3, 224, 224)
    mask = torch.tensor([[[True, False]]])
    x[:, :, 1] = float('nan')
    model.eval()
    with torch.no_grad():
        y1 = model(x, mask)
        x[:, :, 1] = 1.0
        y2 = model(x, mask)
    assert y1.shape == (1, 12) and torch.isfinite(y1).all()
    assert torch.equal(y1, y2), 'padded window changed logits'
    model.train()
    y = model(x, mask)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y, torch.zeros_like(y))
    loss.backward()
    assert model.head.weight.grad is not None and torch.isfinite(model.head.weight.grad).all()
    result = {
        'status': 'passed', 'checkpoint_sha256': WEIGHT_SHA256,
        'model': TIMM_NAME, 'torch': torch.__version__, 'timm': timm.__version__,
        'safetensors': safetensors.__version__, 'elapsed_s': time.perf_counter() - t0,
        'logits_shape': list(y.shape), 'parameter_count': sum(p.numel() for p in model.parameters()),
    }
    Path('/kaggle/working/p01a_cpu_preflight.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))

if __name__ == '__main__':
    main()
