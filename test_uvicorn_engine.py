import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
logging.basicConfig(level=logging.ERROR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    from qwen3_tts_gguf.inference.engine import TTSEngine
    import time
    e = TTSEngine(model_dir='model-base', onnx_provider='CUDA')
    print('Engine ready:', e.ready)
    yield

app = FastAPI(lifespan=lifespan)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8105)
