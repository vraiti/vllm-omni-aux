"""
Demo of FLUX.2-dev prompt upsampling for both T2I and I2I.
Generates two images per run: one without upsampling and one with.

Usage:
    # T2I (no input image)
    python demo_upsample.py --prompt "a cat sitting on a windowsill"

    # I2I (with input image)
    python demo_upsample.py --prompt "replace the bunny with a dog" --image sample-input.png
"""

import argparse
import os

import torch
from PIL import Image

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.2-dev prompt upsampling demo")
    parser.add_argument("--model", default="black-forest-labs/FLUX.2-dev")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", type=str, default=None, help="Input image for I2I upsampling")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    tp_size = torch.cuda.device_count()
    parallel_config = DiffusionParallelConfig(tensor_parallel_size=tp_size)

    omni = Omni(
        model=args.model,
        parallel_config=parallel_config,
    )

    input_image = None
    if args.image:
        if not os.path.exists(args.image):
            raise FileNotFoundError(f"Input image not found: {args.image}")
        input_image = Image.open(args.image).convert("RGB")

    mode = "i2i" if input_image else "t2i"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nMode: {mode.upper()}")
    print(f"Prompt: {args.prompt}")

    prompt_data = {
        "prompt": args.prompt,
        "multi_modal_data": {"image": input_image},
    }

    for label, temperature in [("baseline", None), ("upsampled", args.temperature)]:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

        extra_args = {}
        if temperature is not None:
            extra_args["caption_upsample_temperature"] = temperature

        sampling_params = OmniDiffusionSamplingParams(
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            extra_args=extra_args,
        )

        print(f"\nGenerating {label} (temperature={temperature})...")
        outputs = omni.generate(prompt_data, sampling_params)
        image = outputs[0].request_output.images[0]

        path = os.path.join(args.output_dir, f"flux2-{mode}-{label}.png")
        image.save(path)
        print(f"Saved to {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
