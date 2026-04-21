"""Debug script: run only the MistralEncoderModel prompt upsampling loop."""
import os, torch
from PIL import Image
from transformers import AutoConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.models.mistral_encoder.mistral_encoder import MistralEncoderModel
from diffusers.pipelines.flux2.system_messages import (
    SYSTEM_MESSAGE_UPSAMPLING_I2I,
    SYSTEM_MESSAGE_UPSAMPLING_T2I,
)

if __name__ == "__main__":
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    init_distributed_environment(world_size=1, rank=0)
    initialize_model_parallel(tensor_parallel_size=1)

    model_name = "black-forest-labs/FLUX.2-dev"
    config = AutoConfig.from_pretrained(model_name, subfolder="text_encoder")
    encoder = MistralEncoderModel(config).to("cuda").to(torch.bfloat16)

    from safetensors.torch import load_file
    from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

    model_path = download_weights_from_hf_specific(model_name, None, ["*"])
    te_dir = os.path.join(model_path, "text_encoder")
    weights = []
    for f in sorted(os.listdir(te_dir)):
        if f.endswith(".safetensors"):
            weights.extend(load_file(os.path.join(te_dir, f)).items())
    encoder.load_weights(weights)

    from transformers import PixtralProcessor
    processor = PixtralProcessor.from_pretrained(model_name, subfolder="tokenizer")
    encoder.set_processor(processor, SYSTEM_MESSAGE_UPSAMPLING_T2I, SYSTEM_MESSAGE_UPSAMPLING_I2I)

    img = Image.open(os.path.join(os.path.dirname(__file__), "sample-input.png")).convert("RGB")

    cases = [
        ("t2i", "a cat sitting on a windowsill", None),
        ("i2i", "replace the bunny with a dog", [[img]]),
    ]
    temperatures = [0.05, 0.10, 0.15, 0.30, 0.50]

    for mode, prompt, images in cases:
        print(f"\n{'='*60}")
        print(f"Mode: {mode} | Original prompt: {prompt!r}")
        print(f"{'='*60}")
        for temp in temperatures:
            result = encoder.upsample_prompt(prompt, images=images, temperature=temp)
            print(f"  temp={temp:.2f}: {result[0]!r}")
