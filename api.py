import time
import torch
from einops import rearrange
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from flux.util import (
    SamplingOptions,
    load_ae,
    load_clip,
    load_flow_model,
    load_flow_model_quintized,
    load_t5,
)
from pulid.pipeline_flux import PuLIDPipeline
from pulid.utils import resize_numpy_image_long

app = FastAPI()

class ImageGenerationRequest(BaseModel):
    width: int
    height: int
    num_steps: int
    start_step: int
    guidance: float
    seed: int
    prompt: str
    id_weight: float = 1.0
    neg_prompt: str = ""
    true_cfg: float = 1.0
    timestep_to_start_cfg: int = 1
    max_sequence_length: int = 128

def get_models(name: str, device: torch.device, offload: bool, fp8: bool):
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    if fp8:
        model = load_flow_model_quintized(name, device="cpu" if offload else device)
    else:
        model = load_flow_model(name, device="cpu" if offload else device)
    model.eval()
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip

class FluxGenerator:
    def __init__(self, model_name: str, device: str, offload: bool, aggressive_offload: bool, args):
        self.device = torch.device(device)
        self.offload = offload
        self.aggressive_offload = aggressive_offload
        self.model_name = model_name
        self.model, self.ae, self.t5, self.clip = get_models(
            model_name,
            device=self.device,
            offload=self.offload,
            fp8=args["fp8"],
        )
        self.pulid_model = PuLIDPipeline(
            self.model,
            device="cpu" if offload else device,
            weight_dtype=torch.bfloat16,
            onnx_provider=args["onnx_provider"],
        )
        if offload:
            self.pulid_model.face_helper.face_det.mean_tensor = (
                self.pulid_model.face_helper.face_det.mean_tensor.to(
                    torch.device("cuda")
                )
            )
            self.pulid_model.face_helper.face_det.device = torch.device("cuda")
            self.pulid_model.face_helper.device = torch.device("cuda")
            self.pulid_model.device = torch.device("cuda")
        self.pulid_model.load_pretrain(args["pretrained_model"])

    @torch.inference_mode()
    def generate_image(
        self,
        width,
        height,
        num_steps,
        start_step,
        guidance,
        seed,
        prompt,
        id_image=None,
        id_weight=1.0,
        neg_prompt="",
        true_cfg=1.0,
        timestep_to_start_cfg=1,
        max_sequence_length=128,
    ):
        self.t5.max_length = max_sequence_length

        seed = int(seed)
        if seed == -1:
            seed = None

        opts = SamplingOptions(
            prompt=prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )

        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        print(f"Generating '{opts.prompt}' with seed {opts.seed}")
        t0 = time.perf_counter()

        use_true_cfg = abs(true_cfg - 1.0) > 1e-2

        # prepare input
        x = get_noise(
            1,
            opts.height,
            opts.width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=opts.seed,
        )
        timesteps = get_schedule(
            opts.num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=True,
        )

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=opts.prompt)
        inp_neg = (
            prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)
            if use_true_cfg
            else None
        )

        # offload TEs to CPU, load processor models and id encoder to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.pulid_model.components_to_device(torch.device("cuda"))

        if id_image is not None:
            id_image = resize_numpy_image_long(id_image, 1024)
            id_embeddings, uncond_id_embeddings = self.pulid_model.get_id_embedding(
                id_image, cal_uncond=use_true_cfg
            )
        else:
            id_embeddings = None
            uncond_id_embeddings = None

        # offload processor models and id encoder to CPU, load dit model to gpu
        if self.offload:
            self.pulid_model.components_to_device(torch.device("cpu"))
            torch.cuda.empty_cache()
            if self.aggressive_offload:
                self.model.components_to_gpu()
            else:
                self.model = self.model.to(self.device)

        # denoise initial noise
        x = denoise(
            self.model,
            **inp,
            timesteps=timesteps,
            guidance=opts.guidance,
            id=id_embeddings,
            id_weight=id_weight,
            start_step=start_step,
            uncond_id=uncond_id_embeddings,
            true_cfg=true_cfg,
            timestep_to_start_cfg=timestep_to_start_cfg,
            neg_txt=inp_neg["txt"] if use_true_cfg else None,
            neg_txt_ids=inp_neg["txt_ids"] if use_true_cfg else None,
            neg_vec=inp_neg["vec"] if use_true_cfg else None,
            aggressive_offload=self.aggressive_offload,
        )

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x.float(), opts.height, opts.width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")
        # bring into PIL format
        x = x.clamp(-1, 1)
        # x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        return img, str(opts.seed), self.pulid_model.debug_img_list
args = {
    "fp8": True,
    "onnx_provider": "gpu",
    "pretrained_model": None,
    "dev": True
}
generator = FluxGenerator("flux-dev", "cuda", False, False, args)

@app.post("/generate_image")
async def generate_image_endpoint(request: ImageGenerationRequest, id_image: Optional[UploadFile] = None):
    id_image_data = None
    if id_image:
        id_image_data = Image.open(id_image.file)
    
    output_image, seed_output, intermediate_output = generator.generate_image(
        request.width,
        request.height,
        request.num_steps,
        request.start_step,
        request.guidance,
        request.seed,
        request.prompt,
        id_image_data,
        request.id_weight,
        request.neg_prompt,
        request.true_cfg,
        request.timestep_to_start_cfg,
        request.max_sequence_length,
    )

    output_image.save("generated_image.png")
    return JSONResponse(content={"message": "Image generated successfully", "seed": seed_output})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)