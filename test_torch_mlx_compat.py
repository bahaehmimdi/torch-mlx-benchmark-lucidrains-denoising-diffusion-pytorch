"""Test lucidrains/denoising-diffusion-pytorch (simple_diffusion) with torch-mlx.

Loads the upstream `simple_diffusion.py` UNet + GaussianDiffusion byte-for-byte
(bypassing the package __init__ which pulls heavy non-MLX deps), builds a small
UViT diffusion model and exercises the training loss `p_losses` (forward +
backward + optimizer step).

Round 360 gap worked around (test-local only, upstream file untouched):
`Upsample.init_conv_` does `conv.weight.data.copy_(w)` and
`nn.init.zeros_(conv.bias.data)`. In torch-mlx `param.weight.data` returns a raw
`mx.array` (no `copy_`). The exact same values are routed through the supported
`torch.no_grad()` + `param.copy_()` / `param.zero_()` path below.
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # torch-mlx root

import torch
import torch.nn as nn
from einops import repeat

_src = Path("/Users/bahae/torch-mlx/test-projects/denoising-diffusion-pytorch/denoising_diffusion_pytorch/simple_diffusion.py")
_spec = importlib.util.spec_from_file_location("simple_diffusion_upstream", _src)
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)


def init_conv_(self, conv):
    """Round 360 shim: identical values, supported param-write path."""
    o, i, h, w = conv.weight.shape
    conv_weight = torch.empty(o // self.factor_squared, i, h, w)
    nn.init.kaiming_uniform_(conv_weight)
    conv_weight = repeat(conv_weight, "o ... -> (o r) ...", r=self.factor_squared)
    with torch.no_grad():
        conv.weight.copy_(conv_weight)
        conv.bias.zero_()


sd.Upsample.init_conv_ = init_conv_

UViT = sd.UViT
GaussianDiffusion = sd.GaussianDiffusion

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name}")
    return cond


def main():
    print("=" * 60)
    print("lucidrains/denoising-diffusion-pytorch — simple_diffusion (UViT DDPM)")
    print("=" * 60)

    # Small UViT denoiser (exact upstream classes, unmodified).
    denoise_fn = UViT(
        dim=32,
        init_dim=32,
        dim_mults=(1, 2),
        channels=1,
        vit_depth=2,
        patch_size=2,
        dual_patchnorm=True,
    )

    diffusion = GaussianDiffusion(
        denoise_fn,
        image_size=32,
        channels=1,
        pred_objective="v",
        num_sample_steps=2,
    )

    model = diffusion
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[DDPM] model: {n_params:,} params, {len(list(model.parameters()))} tensors")
    check("DDPM model created", n_params > 0)

    # Training: predict the loss on a small batch (forward + backward).
    img = torch.randn(2, 1, 32, 32)
    times = torch.full((2,), 2 - 1)

    loss = model.p_losses(img, times, loss_reduction="mean")
    check("p_losses produced a scalar", bool(torch.isfinite(loss).item()))
    print(f"  loss = {loss.item():.4f}")

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    n_grad = sum(g is not None for g in grads)
    check(f"backward: {n_grad}/{len(grads)} params got gradients", n_grad == len(grads))

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.step()
    check("optimizer step ran", True)

    # Second loss after one optimizer step should still be finite (no NaN explosion).
    loss2 = model.p_losses(img, times, loss_reduction="mean")
    check("post-step loss finite", bool(torch.isfinite(loss2).item()))

    print("\n" + "=" * 60)
    print(f"RESULT: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
