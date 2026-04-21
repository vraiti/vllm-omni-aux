"""FLUX.2-dev prompt upsampling demo: generates baseline + upsampled for both T2I and I2I."""
import os, torch
from PIL import Image
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

if __name__ == "__main__":
    img = Image.open(os.path.join(os.path.dirname(__file__), "sample-input.png")).convert("RGB")
    omni = Omni(model="black-forest-labs/FLUX.2-dev", parallel_config=DiffusionParallelConfig(tensor_parallel_size=4))
    os.makedirs("outputs/upsample", exist_ok=True)

    for mode, prompt, image in [("t2i", "a cat sitting on a windowsill", None), ("i2i", "replace the bunny with a dog", [img])]:
        for label, temp in [("baseline", None), ("upsampled", 0.15)]:
            out = omni.generate(
                {"prompt": prompt, "multi_modal_data": {"image": image}},
                OmniDiffusionSamplingParams(
                    generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
                    guidance_scale=4.0, num_inference_steps=50,
                    extra_args={"caption_upsample_temperature": temp} if temp else {},
                ),
            )
            path = os.path.join("outputs/upsample", f"flux2-{mode}-{label}.png")
            out[0].request_output.images[0].save(path)
            print(f"[{mode}/{label}] {os.path.abspath(path)}")
