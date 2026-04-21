"""FLUX.2-dev prompt upsampling demo: generates baseline + upsampled for both T2I and I2I."""
import os, torch
from PIL import Image
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

MODEL = "black-forest-labs/FLUX.2-dev"
T2I_PROMPT = "a cat sitting on a windowsill"
I2I_PROMPT = "replace the bunny with a dog"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "sample-input.png")
TEMPERATURE = 0.15
SEED = 42
STEPS = 50
GUIDANCE = 4.0
TP_SIZE = 4
OUTPUT_DIR = "outputs/upsample"

if __name__ == "__main__":
    img = Image.open(IMAGE_PATH).convert("RGB")
    omni = Omni(model=MODEL, parallel_config=DiffusionParallelConfig(tensor_parallel_size=TP_SIZE))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for mode, prompt, image in [("t2i", T2I_PROMPT, None), ("i2i", I2I_PROMPT, img)]:
        for label, temp in [("baseline", None), ("upsampled", TEMPERATURE)]:
            extra = {"caption_upsample_temperature": temp} if temp else {}
            out = omni.generate(
                {"prompt": prompt, "multi_modal_data": {"image": image}},
                OmniDiffusionSamplingParams(
                    generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(SEED),
                    guidance_scale=GUIDANCE, num_inference_steps=STEPS, extra_args=extra,
                ),
            )
            path = os.path.join(OUTPUT_DIR, f"flux2-{mode}-{label}.png")
            out[0].request_output.images[0].save(path)
            print(f"[{mode}/{label}] {os.path.abspath(path)}")
