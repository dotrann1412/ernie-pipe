from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict
from contextlib import asynccontextmanager
from conveyor import Pipeline, Stage
import torch
from fastapi import Depends
from diffusers.pipelines.ernie_image.pipeline_ernie_image import ErnieImagePipeline
from PIL import Image
from io import BytesIO
import base64

DEFAULT_NEGATIVE_PROMPT = '''
(worst quality, low quality, normal quality:1.5), lowres, blurry, out of focus, noise, grain, jpeg artifacts, compression artifacts, oversharpen, overexposed, underexposed, bad lighting, flat lighting, unrealistic lighting, CGI, 3d render, cartoon, anime, painting, illustration, sketch, stylized, artificial, plastic skin, waxy skin, smooth skin, fake textures, unrealistic materials,

bad anatomy, deformed, disfigured, malformed, mutated, extra limbs, missing limbs, extra fingers, missing fingers, fused fingers, long neck, bad proportions, incorrect perspective, twisted body, duplicate, cloned face,

bad face, ugly, distorted face, asymmetrical face, cross-eye, lazy eye, bad eyes, unrealistic eyes, extra eyes, poorly drawn eyes, dead eyes, bad teeth, unnatural smile,

floating objects, disconnected limbs, poorly drawn hands, bad hands, bad feet, incorrect shadows, inconsistent shadows, incorrect reflections, inconsistent reflections,

watermark, signature, logo, text, caption, username, border, frame
'''

class ImageGenerationRequest(BaseModel):
    prompt: str

    # default values but customizable
    height: int = 1264
    width: int = 848
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    use_pe: bool = True 

    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT

class InterImageGenerationResponse(ImageGenerationRequest):
    generated_image: Image.Image
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ImageGenerationResponse(ImageGenerationRequest):
    image_url: str

def make_generation_stage(device_id: int):
    pipe = ErnieImagePipeline.from_pretrained(
        "Baidu/ERNIE-Image",
        torch_dtype=torch.bfloat16,
    ).to(f"cuda:{device_id}")

    def generate(request: ImageGenerationRequest) -> InterImageGenerationResponse:
        image = pipe(
            request.prompt, 
            height=request.height, 
            width=request.width, 
            num_inference_steps=request.num_inference_steps, 
            guidance_scale=request.guidance_scale, 
            use_pe=request.use_pe,
            negative_prompt=request.negative_prompt
        ).images[0] # type: ignore

        torch.cuda.empty_cache()

        return InterImageGenerationResponse(
            **request.model_dump(), 
            generated_image=image
        )

    return generate

async def upload_image(req: InterImageGenerationResponse) -> ImageGenerationResponse:
    buffer = BytesIO()
    req.generated_image.save(buffer, format="PNG")

    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

    uri = f"data:image/png;base64,{encoded_image}"

    return ImageGenerationResponse(
        **req.model_dump(),
        image_url=uri
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    device_ids = [0, 1]

    app.state.pipeline = Pipeline(
        stages=[
            Stage(
                [make_generation_stage(i) for i in device_ids],
                queue_size_per_worker=10,
                stage_name='generation'
            ),
            Stage(
                [upload_image] * 4,
                queue_size_per_worker=10,
                stage_name='upload'
            ),
        ]   
    )

    async with app.state.pipeline:
        yield

def depends_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline

app = FastAPI(prefix="/v1", lifespan=lifespan)

@app.post("/images/generations")
async def generate_image(
    request: ImageGenerationRequest, 
    pipeline: Pipeline = Depends(depends_pipeline)
) -> ImageGenerationResponse:
    return await pipeline.submit(request)
