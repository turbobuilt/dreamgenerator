# ---
# output-directory: "/tmp/stable-diffusion-xl"
# args: ["--prompt", "An astronaut riding a green horse"]
# runtimes: ["runc", "gvisor"]
# ---
# # Stable Diffusion XL 1.0
#
# This example is similar to the [Stable Diffusion CLI](/docs/examples/stable_diffusion_cli)
# example, but it generates images from the larger SDXL 1.0 model. Specifically, it runs the
# first set of steps with the base model, followed by the refiner model.
#
# [Try out the live demo here!](https://modal-labs--stable-diffusion-xl-app.modal.run/) The first
# generation may include a cold-start, which takes around 20 seconds. The inference speed depends on the GPU
# and step count (for reference, an A100 runs 40 steps in 8 seconds).

# ## Basic setup

from pathlib import Path

from modal import Image, Mount, Stub, asgi_app, gpu, method, Volume

# ## Define a container image
#
# To take advantage of Modal's blazing fast cold-start times, we'll need to download our model weights
# inside our container image with a download function. We ignore binaries, ONNX weights and 32-bit weights.
#
# Tip: avoid using global variables in this function to ensure the download step detects model changes and
# triggers a rebuild.


def download_models():
    from huggingface_hub import snapshot_download

    ignore = ["*.bin", "*.onnx_data", "*/diffusion_pytorch_model.safetensors"]
    snapshot_download(
        "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
    )
    snapshot_download(
        "stabilityai/stable-diffusion-xl-refiner-1.0", ignore_patterns=ignore
    )


sdxl_image = (
    Image.debian_slim()
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers~=0.19",
        "invisible_watermark~=0.1",
        "transformers~=4.31",
        "accelerate~=0.21",
        "safetensors~=0.3",
    )
    .run_function(download_models)
    .pip_install("requests~=2.26")
)

stub = Stub("stable-diffusion-xl")

# ## Load model and run inference
#
# The container lifecycle [`__enter__` function](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta)
# loads the model at startup. Then, we evaluate it in the `run_inference` function.
#
# To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.

# vol = Volume.persisted("generated-images")

@stub.cls(gpu=gpu.A10G(), container_idle_timeout=2, image=sdxl_image) #, checkpointing_enabled=True)
class Model:
    def __enter__(self):
        import torch
        from diffusers import DiffusionPipeline

        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_options
        )

        # Load refiner model
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            **load_options,
        )

        # Compiling the model graph is JIT so this will increase inference time for the first run
        # but speed up subsequent runs. Uncomment to enable.
        # self.base.unet = torch.compile(self.base.unet, mode="reduce-overhead", fullgraph=True)
        # self.refiner.unet = torch.compile(self.refiner.unet, mode="reduce-overhead", fullgraph=True)

    @method()
    def inference(self, prompt, task_id, upload_url, progress_url, n_steps=48, high_noise_frac=0.9):
        # negative_prompt = "disfigured, ugly, deformed"
        image = self.base(
            prompt=prompt,
            # negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            denoising_end=.9,
            output_type="latent",
        ).images
        image = self.refiner(
            prompt=prompt,
            # negative_prompt=negative_prompt,
            num_inference_steps=2,
            denoising_start=high_noise_frac,
            image=image,
        ).images

        image = image[0]

        import io

        byte_stream = io.BytesIO()
        image.save(byte_stream, format="png")
        image_bytes = byte_stream.getvalue()

        return image_bytes


# And this is our entrypoint; where the CLI is invoked. Explore CLI options
# with: `modal run stable_diffusion_xl.py --prompt 'An astronaut riding a green horse'`


@stub.local_entrypoint()
def main(prompt: str):
    image_bytes = Model().inference.remote(prompt)

    dir = Path("/tmp/stable-diffusion-xl")
    if not dir.exists():
        dir.mkdir(exist_ok=True, parents=True)

    output_path = dir / "output.png"
    print(f"Saving it to {output_path}")
    with open(output_path, "wb") as f:
        f.write(image_bytes)


# ## A user interface
#
# Here we ship a simple web application that exposes a front-end (written in Alpine.js) for
# our backend deployment.
#
# The Model class will serve multiple users from a its own shared pool of warm GPU containers automatically.
#
# We can deploy this with `modal deploy stable_diffusion_xl.py`.

frontend_path = Path(__file__).parent / "frontend"


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
    mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
    allow_concurrent_inputs=5,
    
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
        print("prompt", prompt, "task_id", task_id, "upload_url", upload_url, "progress_url", progress_url)
        if request.headers["authorization"] != "OIMWEFOIJWEOIHFEFHNMFIJFHUFUHFHFHUEFFUHF":
            return Response("Unauthorized", status_code=401)

        image_bytes = Model().inference.remote(prompt, task_id, upload_url, progress_url)

        # upload image_bytes to upload_url using fastapi
        complete(upload_url, image_bytes, progress_url, task_id)
        # import requests
        # requests.put(upload_url, data=image_bytes)
        # # ping progress url with { taskId: task_id, progress: 100, status: "COMPLETED" }
        # requests.put(progress_url, json={"taskId": task_id, "progress": 100, "status": "COMPLETED"})

        # # save to volume {task_id}.png
        # with open(f"/data/{task_id}.png", "wb") as f:
        #     image.save(f, format="png")
        # vol.commit() 

        return { "success": True } # Response(image_bytes, media_type="image/png")
    
    # serve images in volume
    @web_app.post("/images/{task_id}")
    async def get_image(task_id: str):
        from fastapi.responses import FileResponse
        return FileResponse(f"/data/{task_id}.png")

    web_app.mount(
        "/", fastapi.staticfiles.StaticFiles(directory="/assets", html=True)
    )

    return web_app


# https://noise-destroyer--stable-diffusion-xl-app-turbobuilt-dev.modal.run/infer/a-roaring-lion