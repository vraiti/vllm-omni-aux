"""FLUX.2-dev prompt upsampling demo: generates baseline + upsampled for both T2I and I2I."""
import argparse, os, torch
from PIL import Image
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

p = argparse.ArgumentParser()
p.add_argument("--model", default="black-forest-labs/FLUX.2-dev")
p.add_argument("--t2i-prompt", required=True)
p.add_argument("--i2i-prompt", required=True)
p.add_argument("--image", required=True)
p.add_argument("--temperature", type=float, default=0.15)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--steps", type=int, default=50)
p.add_argument("--guidance", type=float, default=4.0)
p.add_argument("--tensor-parallel-size", type=int, default=1)
p.add_argument("--output-dir", default="outputs")
args = p.parse_args()

img = Image.open(args.image).convert("RGB")
omni = Omni(model=args.model, parallel_config=DiffusionParallelConfig(tensor_parallel_size=args.tensor_parallel_size))
os.makedirs(args.output_dir, exist_ok=True)

for mode, prompt, image in [("t2i", args.t2i_prompt, None), ("i2i", args.i2i_prompt, img)]:
    for label, temp in [("baseline", None), ("upsampled", args.temperature)]:
        extra = {"caption_upsample_temperature": temp} if temp else {}
        out = omni.generate(
            {"prompt": prompt, "multi_modal_data": {"image": image}},
            OmniDiffusionSamplingParams(
                generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(args.seed),
                guidance_scale=args.guidance, num_inference_steps=args.steps, extra_args=extra,
            ),
        )
        path = os.path.join(args.output_dir, f"flux2-{mode}-{label}.png")
        out[0].request_output.images[0].save(path)
        print(f"[{mode}/{label}] {os.path.abspath(path)}")
