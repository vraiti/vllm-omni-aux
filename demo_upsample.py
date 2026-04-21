"""
Demo of FLUX.2-dev prompt upsampling for both T2I and I2I.

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
    parser.add_argument("--output", type=str, default=None)
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

    mode = "I2I" if input_image else "T2I"
    print(f"\nMode: {mode}")
    print(f"Original prompt: {args.prompt}")
    print(f"Temperature: {args.temperature}")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    prompt_data = {
        "prompt": args.prompt,
        "multi_modal_data": {"image": input_image},
    }

    sampling_params = OmniDiffusionSamplingParams(
        generator=generator,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        extra_args={"caption_upsample_temperature": args.temperature},
    )

    output_path = args.output
    if output_path is None:
        output_path = f"outputs/flux2-upsample-{mode.lower()}.png"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    outputs = omni.generate(prompt_data, sampling_params)

    first_output = outputs[0]
    images = first_output.request_output.images
    images[0].save(output_path)
    print(f"Saved to {os.path.abspath(output_path)}")


if __name__ == "__main__":
    main()
