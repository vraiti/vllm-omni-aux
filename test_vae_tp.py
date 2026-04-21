"""Standalone VAE encode test to check TP invariance.

Run on a single GPU:
    python test_vae_tp.py --image sample-input.png --output /tmp/vae_test_gpu0.pt --device cuda:0

Then compare:
    python test_vae_tp.py --image sample-input.png --output /tmp/vae_test_gpu1.pt --device cuda:1

    python test_vae_tp.py --compare /tmp/vae_test_gpu0.pt /tmp/vae_test_gpu1.pt
"""

import argparse
import sys

import torch
from diffusers import AutoencoderKLFlux2
from diffusers.image_processor import VaeImageProcessor
from PIL import Image


def encode_image(args):
    image_processor = VaeImageProcessor(
        do_resize=True,
        vae_scale_factor=16,
        vae_latent_channels=32,
        do_normalize=True,
        do_convert_rgb=True,
    )

    vae = AutoencoderKLFlux2.from_pretrained(
        args.model, subfolder="vae", local_files_only=True,
    ).to(args.device)

    img = Image.open(args.image)
    w, h = img.size
    multiple_of = 16
    w = (w // multiple_of) * multiple_of
    h = (h // multiple_of) * multiple_of
    if w * h > 1024 * 1024:
        scale = (1024 * 1024 / (w * h)) ** 0.5
        w = int(w * scale) // multiple_of * multiple_of
        h = int(h * scale) // multiple_of * multiple_of
    img_tensor = image_processor.preprocess(img, height=h, width=w, resize_mode="crop")
    img_tensor = img_tensor.to(device=args.device, dtype=vae.dtype)

    print(f"Input tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
    print(f"Input tensor range: [{img_tensor.min():.4f}, {img_tensor.max():.4f}]")

    checkpoints = {}
    checkpoints["input_tensor"] = img_tensor.cpu()

    with torch.no_grad():
        # Stage 1: encoder
        enc = vae.encoder(img_tensor)
        checkpoints["after_encoder"] = enc.cpu()

        # Stage 2: quant_conv
        if vae.quant_conv is not None:
            enc = vae.quant_conv(enc)
        checkpoints["after_quant_conv"] = enc.cpu()

        # Stage 3: DiagonalGaussianDistribution.mode()
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
        posterior = DiagonalGaussianDistribution(enc)
        latents = posterior.mode()
        checkpoints["after_mode"] = latents.cpu()

    # Stage 4: patchify
    b, c, fh, fw = latents.shape
    latents = latents.view(b, c, fh // 2, 2, fw // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(b, c * 4, fh // 2, fw // 2)
    checkpoints["after_patchify"] = latents.cpu()

    # Stage 5: batch norm
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    checkpoints["bn_mean"] = bn_mean.cpu()
    checkpoints["bn_std"] = bn_std.cpu()
    latents = (latents - bn_mean) / bn_std
    checkpoints["after_bn"] = latents.cpu()

    for k, v in checkpoints.items():
        print(f"  {k}: shape={v.shape}, range=[{v.min():.4f}, {v.max():.4f}]")

    torch.save(checkpoints, args.output)
    print(f"Saved to {args.output}")


def compare(args):
    a = torch.load(args.files[0], map_location="cpu", weights_only=True)
    b = torch.load(args.files[1], map_location="cpu", weights_only=True)

    stages = [
        "input_tensor", "after_encoder", "after_quant_conv",
        "after_mode", "after_patchify", "bn_mean", "bn_std", "after_bn",
    ]
    for key in stages:
        if key not in a or key not in b:
            print(f"{key}: MISSING")
            continue
        ta = a[key].float()
        tb = b[key].float()
        if ta.shape != tb.shape:
            print(f"{key}: SHAPE MISMATCH {ta.shape} vs {tb.shape}")
            continue
        diff = (ta - tb).abs()
        cos = torch.nn.functional.cosine_similarity(
            ta.flatten().unsqueeze(0), tb.flatten().unsqueeze(0),
        ).item()
        print(f"{key}:")
        print(f"  shape: {ta.shape}")
        print(f"  max abs diff:  {diff.max():.6e}")
        print(f"  mean abs diff: {diff.mean():.6e}")
        print(f"  cosine sim:    {cos:.8f}")
        print()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    enc = sub.add_parser("encode")
    enc.add_argument("--image", required=True)
    enc.add_argument("--output", required=True)
    enc.add_argument("--device", default="cuda:0")
    enc.add_argument("--model", default="black-forest-labs/FLUX.2-dev")

    cmp = sub.add_parser("compare")
    cmp.add_argument("files", nargs=2)

    args = parser.parse_args()
    if args.cmd == "encode":
        encode_image(args)
    elif args.cmd == "compare":
        compare(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
