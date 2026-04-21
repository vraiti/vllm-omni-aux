"""Debug script: run only the Mistral text encoder prompt upsampling loop.

Uses HuggingFace transformers directly (no vLLM parallel infrastructure)
to isolate the upsampling behavior.
"""
import os, torch
from PIL import Image
from transformers import Mistral3ForConditionalGeneration, PixtralProcessor
from diffusers.pipelines.flux2.system_messages import (
    SYSTEM_MESSAGE_UPSAMPLING_I2I,
    SYSTEM_MESSAGE_UPSAMPLING_T2I,
)

def _format_upsample_input(prompts, system_message, images=None):
    cleaned = [p.replace("[IMG]", "") for p in prompts]
    if images is None or len(images) == 0:
        return [[
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": p}]},
        ] for p in cleaned]
    messages = [[{"role": "system", "content": [{"type": "text", "text": system_message}]}] for _ in cleaned]
    for i, (el, batch_images) in enumerate(zip(messages, images)):
        if batch_images is not None:
            el.append({"role": "user", "content": [{"type": "image", "image": img} for img in batch_images]})
        el.append({"role": "user", "content": [{"type": "text", "text": cleaned[i]}]})
    return messages

def upsample(model, processor, prompt, images=None, temperature=0.15):
    prompts = [prompt] if isinstance(prompt, str) else prompt
    sys_msg = SYSTEM_MESSAGE_UPSAMPLING_I2I if images else SYSTEM_MESSAGE_UPSAMPLING_T2I
    messages = _format_upsample_input(prompts, sys_msg, images)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", padding="max_length",
        truncation=True, max_length=2048,
    )
    inputs["input_ids"] = inputs["input_ids"].to(model.device)
    inputs["attention_mask"] = inputs["attention_mask"].to(model.device)
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.device, model.dtype)
    ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=temperature, use_cache=True)
    return processor.tokenizer.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)

if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    model_name = "black-forest-labs/FLUX.2-dev"
    processor = PixtralProcessor.from_pretrained(model_name, subfolder="tokenizer")
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_name, subfolder="text_encoder", dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

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
            result = upsample(model, processor, prompt, images=images, temperature=temp)
            print(f"  temp={temp:.2f}: {result[0]!r}")
