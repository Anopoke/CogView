import torch

from cogview.CogView4 import CogView4

# Model path
MODEL_PATH = "/mnt/models/CogView4-6B"
# Precision
TORCH_DTYPE = torch.float32
# The device allocation scheme
DEVICE_MAP = {
    "text_encoder": "cuda:0",
    "tokenizer": "cpu",
    "transformer": "cuda:1",
    "vae": "cpu",
    "scheduler": "cpu"
}

# Test Code
if __name__ == '__main__':
    pipeline = CogView4(
        model_path=MODEL_PATH,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP
    )
    pipeline.enable_model_cpu_offload()
    pipeline.enable_attention_slicing()
    prompt = "一艘未来的宇宙飞船绕着一颗霓虹灯行星运行。"
    # prompt = "A future spaceship orbits a neon planet."
    output = pipeline(
        prompt=prompt,
        guidance_scale=3.5,
        num_images_per_prompt=1,
        num_inference_steps=50,
        width=512,
        height=512,
        # generator=torch.Generator(device="cpu")  # Appoint generator device
    )
    output.images[0].save("CogView4.png")
