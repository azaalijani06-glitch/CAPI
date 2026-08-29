import os
import base64
import httpx
import uuid
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from httpx import HTTPStatusError

apikey = os.environ.get("apikey")

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("images", exist_ok=True)
    app.state.client = httpx.AsyncClient(timeout=30)
    yield
    await app.state.client.aclose()

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],  # Allow your own origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.mount("/images", StaticFiles(directory="images"), name="images")

class Generate(BaseModel):
    height: int
    width: int
    prompt: str

@app.get("/")
def root():
    return JSONResponse(status_code=200, content={"message": "Hello"})

@app.post("/Generate")
async def Generation(request_body: Generate, request: Request):
    invoke_url = "https://ai.api.nvidia.com/v1/genai/black-forest-labs/flux.2-klein-4b"

    headers = {
        "Authorization": f"Bearer {apikey}",
        "Accept": "application/json",
    }

    payload = {
        "prompt": request_body.prompt,
        "width": request_body.width,
        "height": request_body.height,
        "seed": 0,
        "steps": 4
    }

    client = request.app.state.client
    response = await client.post(invoke_url, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except (HTTPStatusError):
        return JSONResponse(status_code=500, content={"message": response.text})
    
    response_json = response.json()
    encoded_image = response_json["artifacts"][0]["base64"]

    def save_image_to_disk(filepath: str, data: bytes):
        with open(filepath, "wb") as f:
            f.write(data)

    image_data = await asyncio.to_thread(base64.b64decode, encoded_image)

    filename = f"{uuid.uuid4()}.jpg"
    filepath = f"images/{filename}"

    await asyncio.to_thread(save_image_to_disk, filepath, image_data)

    base_url = str(request.base_url)

    return JSONResponse(
        status_code=200,
        content={
            "message": {
                "image_url": f"{base_url}images/{filename}"
            }
        }
    )