from apis import ImageGenerationRequest, InterImageGenerationResponse, make_generation_stage
import os
import asyncio
from conveyor import Pipeline, Stage
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    device_ids = [0, 1]

    pipeline = Pipeline(
        stages=[
            Stage(
                [make_generation_stage(i) for i in device_ids],
                queue_size_per_worker=10,
                stage_name='generation'
            ),
        ]
    )

    with open("prompts.json", "r") as f:
        prompts_dicts = sorted(json.load(f), key=lambda x: x.get("request_id"))

    tasks: list[asyncio.Task[tuple[str, Exception | InterImageGenerationResponse]]] = []

    async def wraps(request_id, coroutine):
        try:
            resp = await coroutine
        except Exception as e:
            logger.error(f"Error generating image for request {request_id}: {e}")
            return request_id, e

        return request_id, resp

    os.makedirs("images", exist_ok=True)

    async with pipeline:
        for prompt_dict in prompts_dicts:
            request_id, prompt = prompt_dict.get("request_id"), prompt_dict.get("prompt")
            image_path = os.path.join("images", f"{request_id}.png")

            if os.path.exists(image_path):
                continue

            request = ImageGenerationRequest(prompt=prompt, use_pe=True, num_inference_steps=8)
            tasks.append(asyncio.create_task(wraps(request_id, pipeline.submit(request))))

        for future in asyncio.as_completed(tasks):
            request_id, resp = await future
            if isinstance(resp, Exception):
                logger.error(f"Error generating image for request {request_id}: {resp}")
                continue

            image_path = os.path.join("images", f"{request_id}.png")
            resp.generated_image.save(image_path)
            logger.info(f"Saved image for request {request_id}; {image_path}")

if __name__ == "__main__":
    asyncio.run(main())