from sys import prefix
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from conveyor import Pipeline, Stage
import torch
from fastapi import Depends
from diffusers.pipelines.ernie_image.pipeline_ernie_image import ErnieImagePipeline

class ImageGenerationRequest(BaseModel):
    prompt: str

    # default values but customizable
    height: int = 1264
    width: int = 848
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    use_pe: bool = True 

class InterImageGenerationResponse(ImageGenerationRequest):
    data: bytes
    mime_type: str = "image/png"

class ImageGenerationResponse(ImageGenerationRequest):
    image_url: str

def make_generation_stage():
    pipe = ErnieImagePipeline.from_pretrained(
        "Baidu/ERNIE-Image",
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    def generate(request: ImageGenerationRequest) -> InterImageGenerationResponse:
        image = pipe(
            request.prompt, 
            height=request.height, 
            width=request.width, 
            num_inference_steps=request.num_inference_steps, 
            guidance_scale=request.guidance_scale, 
            use_pe=request.use_pe
        ).images[0] # type: ignore

        data = image.tobytes()
        
        return InterImageGenerationResponse(
            **request.model_dump(), 
            data=data
        )

    return generate

async def upload_image(req: InterImageGenerationResponse) -> ImageGenerationResponse:
    
    return ImageGenerationResponse(
        **req.model_dump(),
        image_url="https://example.com/image.png"
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = Pipeline(
        stages=[
            Stage(
                [make_generation_stage()],
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

def depends_pipeline(app: FastAPI) -> Pipeline:
    return app.state.pipeline

app = FastAPI(prefix="/v1")

@app.post("/images/generations")
async def generate_image(
    request: ImageGenerationRequest, 
    pipeline: Pipeline = Depends(depends_pipeline)
) -> ImageGenerationResponse:
    return await pipeline.submit(request)
