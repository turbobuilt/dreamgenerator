from pathlib import Path
from modal import Image, Mount, Stub, asgi_app, gpu, method, Volume

def download_models():
    from huggingface_hub import snapshot_download

    ignore = ["*.bin", "*.onnx_data", "*/diffusion_pytorch_model.safetensors"]
    snapshot_download(
        "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
    )
    snapshot_download(
        "stabilityai/stable-diffusion-xl-refiner-1.0", ignore_patterns=ignore
    )

    import torch
    import io
    from diffusers import DiffusionPipeline
    print("running enter")
    load_options = dict(
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        device_map="auto",
    )

    # Load base model
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", **load_options
    )
    #make sure /data dir exists
    Path("/data").mkdir(parents=True, exist_ok=True)
    base.save_pretrained("/data/base")

    # Load refiner model
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        **load_options,
    )
    refiner.save_pretrained("/data/refiner")


sdxl_image = (
    Image.debian_slim()
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1", "curl" #, "software-properties-common"
    )
    # .run_commands(["add-apt-repository ppa:jonathonf/rustlang -y"])
    # .run_commands(["apt-get update"])
    # .apt_install("rustc")
    # .run_commands(["curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"])
    # .run_commands(["source $HOME/.cargo/env"])
    .pip_install(
        # "tokenizers==0.12.1",
        "diffusers~=0.19",
        "invisible_watermark~=0.1",
        "transformers~=4.31",
        "accelerate~=0.21",
        "safetensors~=0.3",
    )
    .run_function(download_models)
    .pip_install("requests~=2.26")
    # now run this command
    # transformers.utils.move_cache() to move cache
    .run_commands(["echo 'import transformers; transformers.utils.move_cache()' | python3"])
)

stub = Stub("stable-diffusion-xl", image=sdxl_image)


with stub.image.imports():
    import torch
    import io
    from diffusers import DiffusionPipeline

    load_options = dict(
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        # device_map=torch.device("cpu"),
        local_files_only=True
    )

    # Load base model
    base = DiffusionPipeline.from_pretrained("/data/base", **load_options)
    # print(base.hf_device_map)
    refiner = DiffusionPipeline.from_pretrained(
        "/data/refiner",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        **load_options,
    )
    print('ran enter')


@stub.function(checkpointing_enabled=False, gpu=gpu.A10G(), container_idle_timeout=2, image=sdxl_image)
def inference(prompt, task_id, upload_url, progress_url, n_steps=48, high_noise_frac=0.9):
    print('starting inference')
    
    # move to cuda
    image = base.to("cuda")(
        prompt=prompt,
        # negative_prompt=negative_prompt,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images
    image = refiner.to("cuda")(
        prompt=prompt,
        # negative_prompt=negative_prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images

    image = image[0]


    byte_stream = io.BytesIO()
    image.save(byte_stream, format="png")
    image_bytes = byte_stream.getvalue()

    return image_bytes


def complete(upload_url, image_bytes, progress_url, task_id):
    import urllib.request
    import json

    # Upload image
    print("uploading image")
    req = urllib.request.Request(upload_url, data=image_bytes, method='PUT', headers={ "Content-Type": "image/png" })
    with urllib.request.urlopen(req) as response:
        pass  # Do something with the response if needed

    # Ping progress URL
    print("pinging progress url")
    progress_data = {"taskId": task_id, "progress": 100, "status": "COMPLETED"}
    req = urllib.request.Request(progress_url, data=json.dumps(progress_data).encode("utf-8"), method='PUT')
    with urllib.request.urlopen(req) as response:
        pass  # Do something with the response if needed


@stub.function(
    allow_concurrent_inputs=2,
)
@asgi_app()
def app():
    import fastapi.staticfiles
    from fastapi import FastAPI, Request

    web_app = FastAPI()

    @web_app.post("/infer")
    async def infer(request: Request): #, prompt: str, task_id: str, upload_url: str, progress_url: str):
        body = await request.json()
        prompt = body["prompt"]
        task_id = body["task_id"]
        upload_url = body["upload_url"]
        progress_url = body["progress_url"]
        from fastapi.responses import Response

        if request.headers["authorization"] != "OIMWEFOIJWEOIHFEFHNMFIJFHUFUHFHFHUEFFUHF":
            return Response("Unauthorized", status_code=401)
        print('calling inference')
        image_bytes = inference.remote(prompt, task_id, upload_url, progress_url)

        # upload image_bytes to upload_url using fastapi
        complete(upload_url, image_bytes, progress_url, task_id)

        return { "success": True }
    
    # serve images in volume
    @web_app.post("/images/{task_id}")
    async def get_image(task_id: str):
        from fastapi.responses import FileResponse
        return FileResponse(f"/data/{task_id}.png")

    # web_app.mount(
    #     "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    # )

    return web_app