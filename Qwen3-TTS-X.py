"""
Qwen3-TTS-X — 智能流式语音克隆工作台 + OpenAI 兼容 API 一体化服务

整合了两大功能模块于同一进程：
  1. Studio WebUI  — 可视化音色管理与实时合成工作台 (根路径 /)
  2. OpenAI API    — 兼容 /v1/audio/speech 流式语音合成接口

单引擎启动，共享推理资源，无需分别运行两个脚本。

=== OpenAI 兼容接口文档 ===

  POST /v1/audio/speech
  {
      "model": "qwen3-tts-x",         # 模型名称（兼容但实际固定）
      "input": "你好世界",             # 待合成文本
      "voice": "test1",               # 音色名称（兼容但实际固定）
      "response_format": "wav",       # 音频格式：wav / pcm
      "speed": 1.0                    # 语速（保留字段，当前模型不直接支持变速）
  }

  调用示例：
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8100/v1", api_key="not-needed")
    response = client.audio.speech.create(
        model="qwen3-tts-x",
        voice="test1",
        input="你好，测试语音合成系统。",
    )
    response.stream_to_file("output.wav")

  完整接口文档：http://localhost:8100/docs
"""
import os
import sys
import glob
import time
import json
import struct
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import asyncio
import numpy as np
import urllib.parse
import shutil
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, Form, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from contextlib import asynccontextmanager

from qwen3_tts_gguf.inference import TTSEngine, TTSConfig


# =====================================================================
# 配置常量
# =====================================================================
MODEL_DIR = "model-base"
REF_JSON = "output/elaborate/test1-voice.json"    # API 默认音色 JSON

PORT = 8100                # 统一服务端口

SAMPLE_RATE = 24000        # 采样率 24kHz
CHANNELS = 1               # 单声道
BITS_PER_SAMPLE = 16       # 16-bit PCM（行业标准）

# OpenAI 兼容模型信息
MODEL_ID = "qwen3-tts-x"
AVAILABLE_VOICES = ["test1"]


# =====================================================================
# 全局状态管理
# =====================================================================
engine = None
stream = None
stream_lock = asyncio.Lock()


# =====================================================================
# WAV 头构造（API 流式输出用）
# =====================================================================
def make_wav_header(data_size: int = 0x7FFFFFFF) -> bytes:
    """
    构造标准 WAV 文件头（PCM int16, 24kHz, mono）。
    流式输出时 data_size 用最大值占位，客户端可直接流式播放。
    """
    byte_rate = SAMPLE_RATE * CHANNELS * BITS_PER_SAMPLE // 8
    block_align = CHANNELS * BITS_PER_SAMPLE // 8

    header = struct.pack('<4sI4s', b'RIFF', 36 + data_size, b'WAVE')
    header += struct.pack('<4sIHHIIHH', b'fmt ', 16,
                          1,                    # PCM 格式
                          CHANNELS,
                          SAMPLE_RATE,
                          byte_rate,
                          block_align,
                          BITS_PER_SAMPLE)
    header += struct.pack('<4sI', b'data', data_size)
    return header  # 44 字节


# =====================================================================
# 生命周期管理（引擎初始化与释放）
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, stream
    print("🚀 正在初始化 Qwen3-TTS-X 一体化服务...")

    # 确保存放目录存在
    os.makedirs("example_audio", exist_ok=True)
    os.makedirs("output/studio_history", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)

    try:
        engine = TTSEngine(model_dir=MODEL_DIR, onnx_provider="CUDA", chunk_size=8)
        stream = engine.create_stream()

        # 尝试加载 API 默认音色（如果存在）
        if os.path.exists(REF_JSON):
            try:
                stream.set_voice(REF_JSON)
                print(f"🎤 [API] 默认音色已加载: {REF_JSON}")
            except Exception as e:
                print(f"⚠️ [API] 默认音色加载失败（不影响 Studio 使用）: {e}")
        else:
            print(f"⚠️ [API] 默认音色文件不存在: {REF_JSON}（API 调用前需先通过 Studio 设置音色）")

        print(f"✅ Qwen3-TTS-X 引擎就绪")
        print(f"🌐 服务地址: http://0.0.0.0:{PORT}")
        print(f"🖥️ Studio 工作台: http://localhost:{PORT}/")
        print(f"📖 接口文档: http://localhost:{PORT}/docs")
        print(f"🔗 OpenAI 兼容端点: POST http://localhost:{PORT}/v1/audio/speech")
        yield
    except Exception as e:
        # 🌟 新增这一段：抓取真实报错并高亮打印 🌟
        import traceback
        print("\n" + "="*50)
        print("❌ 引擎初始化失败！真实报错信息如下：")
        traceback.print_exc()
        print("="*50 + "\n")
    finally:
        print("🛑 收到退出信号，正在释放引擎资产...")
        if engine:
            try:
                engine.shutdown()
                print("✅ 引擎资产归还完毕。")
            except:
                pass
        # 终极奥义：彻底杀灭后台死锁，完美回退终端句柄！
        os._exit(0)


app = FastAPI(
    title="Qwen3-TTS-X",
    description="智能流式语音克隆工作台 + OpenAI 兼容 TTS API 一体化服务",
    version="1.0.0",
    lifespan=lifespan,
)

# 确保所需静态目录存在，防止 app.mount 初始化时因目录不存在报错
os.makedirs("example_audio", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("static", exist_ok=True)

# 挂载静态服务目录
app.mount("/example_audio", StaticFiles(directory="example_audio"), name="example_audio")
app.mount("/output", StaticFiles(directory="output"), name="output")
app.mount("/static", StaticFiles(directory="static"), name="static")


# =====================================================================
#  第一部分：Studio WebUI 路由
# =====================================================================

@app.get("/", tags=["Studio WebUI"])
async def index():
    """Studio 工作台首页"""
    try:
        with open("studio.html", "r", encoding="utf-8") as f:
            html_content = f.read()
    except UnicodeDecodeError:
        with open("studio.html", "r", encoding="gbk") as f:
            html_content = f.read()
    return HTMLResponse(content=html_content)


@app.get("/api/examples", tags=["Studio WebUI"])
async def list_examples():
    """扫描 example_audio 下的子文件夹，一层结构"""
    os.makedirs("example_audio", exist_ok=True)
    results = []

    subdirs = [os.path.join("example_audio", d) for d in os.listdir("example_audio") if os.path.isdir(os.path.join("example_audio", d))]
    subdirs.sort()

    for d in subdirs:
        voice_name = os.path.basename(d)

        # 寻找音频文件
        audio_path = None
        for ext in [".wav", ".mp3", ".flac"]:
            if os.path.exists(os.path.join(d, f"{voice_name}{ext}")):
                audio_path = f"/example_audio/{voice_name}/{voice_name}{ext}"
                break

        # 寻找有损克隆保存后的 json
        json_path = None
        if os.path.exists(os.path.join(d, f"{voice_name}.json")):
            json_path = f"/example_audio/{voice_name}/{voice_name}.json"

        # 寻找图片
        image_path = None
        for img_ext in [".jpg", ".png", ".gif"]:
            if os.path.exists(os.path.join(d, f"{voice_name}{img_ext}")):
                image_path = f"/example_audio/{voice_name}/{voice_name}{img_ext}"
                break

        # 寻找参考文本
        text_content = ""
        if os.path.exists(os.path.join(d, f"{voice_name}.txt")):
            with open(os.path.join(d, f"{voice_name}.txt"), "r", encoding="utf-8") as tf:
                text_content = tf.read().strip()

        results.append({
            "name": voice_name,
            "folder_url": f"/example_audio/{voice_name}",
            "audio_url": audio_path,
            "json_url": json_path,
            "image": image_path,
            "text": text_content
        })

    return JSONResponse({"data": results})


class SaveTextReq(BaseModel):
    voice_name: str
    text: str

@app.post("/api/save_text", tags=["Studio WebUI"])
async def save_text(req: SaveTextReq):
    """保存或更新右侧填写的参考文本到指定子目录的txt文件"""
    # 限制路径防穿透
    safe_name = os.path.basename(req.voice_name)
    target_dir = os.path.join("example_audio", safe_name)
    if not os.path.exists(target_dir):
        return JSONResponse({"success": False, "msg": "目录不存在"})

    target_txt = os.path.join(target_dir, f"{safe_name}.txt")
    with open(target_txt, "w", encoding="utf-8") as f:
        f.write(req.text)
    return JSONResponse({"success": True})


class ExtractCloneReq(BaseModel):
    voice_name: str
    ref_text: str

@app.post("/api/extract_clone", tags=["Studio WebUI"])
async def extract_clone(req: ExtractCloneReq):
    """
    通过流式无损写入触发本地保存 .json 功能。
    用户在左侧点击克隆时的动作
    """
    if not engine or not engine.ready:
        raise HTTPException(status_code=503, detail="引擎未就绪")

    safe_name = os.path.basename(req.voice_name)
    target_dir = os.path.join("example_audio", safe_name)

    # 寻找真实音频文件
    audio_path = None
    for ext in [".wav", ".mp3", ".flac"]:
        if os.path.exists(os.path.join(target_dir, f"{safe_name}{ext}")):
            audio_path = os.path.join(target_dir, f"{safe_name}{ext}")
            break

    if not audio_path:
        raise HTTPException(status_code=400, detail="未找到原始 wav/mp3 文件去抽取配置")

    async with stream_lock:
        try:
            original_put = stream.decoder.play_q.put
            stream.decoder.play_q.put = lambda *args, **kwargs: None

            stream.set_voice(audio_path, req.ref_text)

            # 使用相同文本触发底层的 clone 后向落地保存
            config = TTSConfig(max_steps=400, temperature=0.6, streaming=False)
            result = await asyncio.to_thread(stream.clone, req.ref_text, 'Chinese', config)

            target_json = os.path.join(target_dir, f"{safe_name}.json")
            result.save(target_json)

            return JSONResponse({"success": True, "json_url": f"/example_audio/{safe_name}/{safe_name}.json"})

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            stream.decoder.play_q.put = original_put


@app.get("/api/history", tags=["Studio WebUI"])
async def load_history():
    """遍历 output/studio_history 并根据时间戳或文件修改时间回溯加载记录"""
    history_dir = "output/studio_history"
    os.makedirs(history_dir, exist_ok=True)

    results = []
    # 寻找所有的 .json 生成记录
    for json_file in glob.glob(os.path.join(history_dir, "*.json")):
        base = os.path.splitext(os.path.basename(json_file))[0]
        # 对应配套必须有 wav
        if os.path.exists(os.path.join(history_dir, f"{base}.wav")):
            # 从 json 文件中提取完整的生成文本记录
            text = base
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    j_data = json.load(f)
                    if 'text' in j_data and j_data['text']:
                        text = j_data['text']
            except Exception:
                pass

            # 兼容时间错乱，解析修改时间
            stat = os.stat(json_file)
            ts = int(stat.st_mtime * 1000)
            results.append({
                "timestamp": ts,
                "text": text,
                "audio_url": f"/output/studio_history/{urllib.parse.quote(base)}.wav",
                "json_url": f"/output/studio_history/{urllib.parse.quote(base)}.json",
                "id": f"history-{urllib.parse.quote(base)}"
            })

    # 按时间降序
    results.sort(key=lambda x: x["timestamp"], reverse=True)
    return JSONResponse({"data": results})


@app.delete("/api/history/delete/{name}", tags=["Studio WebUI"])
async def delete_history(name: str):
    """删除指定历史记录（wav 和 json）"""
    safe_name = os.path.basename(urllib.parse.unquote(name))
    history_dir = "output/studio_history"

    # 删除 .wav 和 .json
    for ext in [".wav", ".json"]:
        target_file = os.path.join(history_dir, f"{safe_name}{ext}")
        if os.path.exists(target_file):
            os.remove(target_file)

    return JSONResponse({"success": True, "deleted": safe_name})


# =================== 音色管理接口 ===================

@app.post("/api/voice/add", tags=["Studio WebUI"])
async def add_voice(
    name: str = Form(...),
    image: Optional[UploadFile] = File(None),
    audio: UploadFile = File(...),
    text: Optional[str] = Form(None)
):
    """添加新音色：创建子目录并保存音频/图片/文本"""
    # 过滤非法字符，防路径穿越
    safe_name = "".join(c for c in name if c not in r'<>:"/\|?*').strip()
    if not safe_name:
        raise HTTPException(status_code=400, detail="音色名称包含非法字符或为空")

    target_dir = os.path.join("example_audio", safe_name)
    if os.path.exists(target_dir):
        raise HTTPException(status_code=409, detail=f"音色「{safe_name}」已存在，请换一个名称")

    os.makedirs(target_dir, exist_ok=True)

    try:
        # 保存音频
        audio_ext = os.path.splitext(audio.filename)[1].lower() if audio.filename else ".wav"
        if audio_ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            audio_ext = ".wav"
        audio_path = os.path.join(target_dir, f"{safe_name}{audio_ext}")
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

        # 保存图片（可选）
        if image and image.filename:
            img_ext = os.path.splitext(image.filename)[1].lower() if image.filename else ".jpg"
            if img_ext not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                img_ext = ".jpg"
            img_path = os.path.join(target_dir, f"{safe_name}{img_ext}")
            with open(img_path, "wb") as f:
                f.write(await image.read())

        # 保存字幕文本（可选）
        if text and text.strip():
            txt_path = os.path.join(target_dir, f"{safe_name}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text.strip())

        return JSONResponse({"success": True, "name": safe_name})

    except Exception as e:
        # 创建失败时清理目录
        shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/edit", tags=["Studio WebUI"])
async def edit_voice(
    old_name: str = Form(...),
    new_name: str = Form(...),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    """编辑音色：重命名文件夹内部文件，并可选替换图像或音频"""
    safe_old_name = "".join(c for c in old_name if c not in r'<>:"/\|?*').strip()
    safe_new_name = "".join(c for c in new_name if c not in r'<>:"/\|?*').strip()

    if not safe_old_name or not safe_new_name:
        raise HTTPException(status_code=400, detail="音色名称包含非法字符或为空")

    old_dir = os.path.join("example_audio", safe_old_name)
    if not os.path.exists(old_dir):
        raise HTTPException(status_code=404, detail="要编辑的音色不存在")

    # 重命名处理
    if safe_old_name != safe_new_name:
        new_dir = os.path.join("example_audio", safe_new_name)
        if os.path.exists(new_dir):
            raise HTTPException(status_code=409, detail=f"音色目录「{safe_new_name}」已存在，请换一个名称")

        # 内部文件重命名
        for filename in os.listdir(old_dir):
            if filename.startswith(safe_old_name):
                ext = filename[len(safe_old_name):]
                old_filepath = os.path.join(old_dir, filename)
                new_filepath = os.path.join(old_dir, f"{safe_new_name}{ext}")
                os.rename(old_filepath, new_filepath)

        # 文件夹重命名
        os.rename(old_dir, new_dir)
        target_dir = new_dir
        target_name = safe_new_name
    else:
        target_dir = old_dir
        target_name = safe_old_name

    try:
        # 替换音频
        if audio and audio.filename:
            for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
                p = os.path.join(target_dir, f"{target_name}{ext}")
                if os.path.exists(p): os.remove(p)

            audio_ext = os.path.splitext(audio.filename)[1].lower()
            if audio_ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]: audio_ext = ".wav"
            audio_path = os.path.join(target_dir, f"{target_name}{audio_ext}")
            with open(audio_path, "wb") as f:
                f.write(await audio.read())

        # 替换图片
        if image and image.filename:
            for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
                p = os.path.join(target_dir, f"{target_name}{ext}")
                if os.path.exists(p): os.remove(p)

            img_ext = os.path.splitext(image.filename)[1].lower()
            if img_ext not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]: img_ext = ".jpg"
            img_path = os.path.join(target_dir, f"{target_name}{img_ext}")
            with open(img_path, "wb") as f:
                f.write(await image.read())

        # 更新字幕
        if text is not None:
            txt_path = os.path.join(target_dir, f"{target_name}.txt")
            if text.strip():
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text.strip())
            else:
                if os.path.exists(txt_path): os.remove(txt_path)

        return JSONResponse({"success": True, "name": target_name})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/voice/delete/{name}", tags=["Studio WebUI"])
async def delete_voice(name: str):
    """删除指定音色及其目录下所有文件"""
    safe_name = os.path.basename(urllib.parse.unquote(name))
    target_dir = os.path.join("example_audio", safe_name)
    if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
        raise HTTPException(status_code=404, detail=f"音色「{safe_name}」不存在")
    shutil.rmtree(target_dir)
    return JSONResponse({"success": True, "deleted": safe_name})


# ================= WebSocket 流式真打字机合成核心 =================
_ws_original_play_q_put = None

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    global _ws_original_play_q_put
    await websocket.accept()

    try:
        data = await websocket.receive_json()
        target_text = data.get("text", "").strip()
        ref_audio_url = data.get("ref_audio", "")
        ref_text = data.get("ref_text", "")
        config_data = data.get("config", {})

        if not target_text:
            await websocket.send_json({"event": "error", "message": "文本为空"})
            return

        async with stream_lock:
            await websocket.send_json({"event": "start"})

            audio_queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            original_put = stream.decoder.play_q.put
            _ws_original_play_q_put = original_put

            def ws_mock_speaker_put(item, *args, **kwargs):
                if getattr(item, "msg_type", "") == "AUDIO" and item.audio is not None:
                    loop.call_soon_threadsafe(audio_queue.put_nowait, item.audio)

            try:
                local_ref_path = urllib.parse.unquote(ref_audio_url.lstrip("/"))
                cache_key = f"{local_ref_path}|{ref_text}"

                if getattr(websocket.app, "last_voice_cache_key", None) != cache_key:
                    print(f"🔄 [WS] 切换音色，重新提取特征...")
                    if local_ref_path.endswith(".json"):
                        stream.set_voice(local_ref_path)
                    else:
                        stream.set_voice(local_ref_path, ref_text)
                    websocket.app.last_voice_cache_key = cache_key
                else:
                    print(f"⚡ [WS] 命中全局音色缓存，跳过提取...")

                stream.decoder.play_q.put = ws_mock_speaker_put

                # 构建推理配置（接收前端参数）
                config = TTSConfig(
                    max_steps=int(config_data.get("max_steps", 1000)),
                    temperature=float(config_data.get("temperature", 0.6)),
                    sub_temperature=float(config_data.get("sub_temperature", 0.6)),
                    top_p=float(config_data.get("top_p", 1.0)),
                    top_k=int(config_data.get("top_k", 50)),
                    min_p=float(config_data.get("min_p", 0.0)),
                    repeat_penalty=float(config_data.get("repeat_penalty", 1.05)),
                    seed=int(config_data["seed"]) if config_data.get("seed") is not None else None,
                    sub_seed=int(config_data["sub_seed"]) if config_data.get("sub_seed") is not None else None,
                    streaming=True,
                )
                print(f"🎛️ [WS] 推理参数: temp={config.temperature}, sub_temp={config.sub_temperature}, seed={config.seed}, sub_seed={config.sub_seed}, max_steps={config.max_steps}")

                async def run_inference():
                    try:
                        res = await asyncio.to_thread(stream.clone, target_text, 'Chinese', config)
                        return res
                    except Exception as e:
                        print(f"❌ [WS] 推理异常: {e}")
                        return None
                    finally:
                        loop.call_soon_threadsafe(audio_queue.put_nowait, None)

                inference_task = asyncio.create_task(run_inference())

                all_chunks = []
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        break

                    all_chunks.append(chunk)
                    audio_bytes = chunk.astype(np.float32).tobytes()
                    await websocket.send_bytes(audio_bytes)

                result = await inference_task
                if result:
                    secure_text = "".join(c for c in target_text if c not in '<>:"/\\|?*\n\r\t')[:30].strip()
                    if not secure_text: secure_text = str(int(time.time()))

                    os.makedirs("output/studio_history", exist_ok=True)

                    save_base = os.path.join("output", "studio_history", secure_text)
                    counter = 1
                    while os.path.exists(f"{save_base}.json"):
                        save_base = os.path.join("output", "studio_history", f"{secure_text}_{counter}")
                        counter += 1

                    result.save(f"{save_base}.wav")
                    result.save(f"{save_base}.json")

                    final_base_name = os.path.basename(save_base)

                    await websocket.send_json({
                        "event": "end",
                        "timestamp": int(time.time() * 1000),
                        "text": target_text,
                        "id": f"history-{int(time.time() * 1000)}",
                        "audio_url": f"/output/studio_history/{urllib.parse.quote(final_base_name)}.wav",
                        "json_url": f"/output/studio_history/{urllib.parse.quote(final_base_name)}.json"
                    })

            except Exception as e:
                await websocket.send_json({"event": "error", "message": str(e)})
            finally:
                stream.decoder.play_q.put = original_put
                _ws_original_play_q_put = None

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")
    finally:
        if _ws_original_play_q_put is not None:
            try:
                stream.decoder.play_q.put = _ws_original_play_q_put
                _ws_original_play_q_put = None
            except:
                pass


# =====================================================================
#  第二部分：OpenAI 兼容 API 路由
# =====================================================================

# ---- OpenAI 兼容请求模型 ----
class SpeechRequest(BaseModel):
    """
    兼容 OpenAI POST /v1/audio/speech 的请求体。

    参考：https://platform.openai.com/docs/api-reference/audio/createSpeech
    """
    model: str = Field(
        default="qwen3-tts-x",
        description="模型 ID（兼容字段，当前固定使用 qwen3-tts-x）"
    )
    input: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="待合成的文本内容"
    )
    voice: str = Field(
        default="test1",
        description="音色名称（兼容字段，当前固定使用 test1）"
    )
    response_format: str = Field(
        default="wav",
        description="音频输出格式：wav（带标准文件头）或 pcm（纯 int16le 数据流）"
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="语速控制（保留字段，当前模型不直接支持变速，传入后忽略）"
    )
    # 以下为扩展字段（OpenAI 标准中不存在，但便于高级用户微调）
    language: str = Field(
        default="Chinese",
        description="[扩展] 语言：Chinese / English / Japanese / Korean 等"
    )
    temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="[扩展] 采样温度"
    )
    seed: int = Field(
        default=-1,
        description="[扩展] 随机种子（-1 为随机）"
    )


# ---- OpenAI 兼容：模型列表 ----
@app.get("/v1/models", summary="列出可用模型", tags=["OpenAI 兼容"])
async def list_models():
    """
    兼容 OpenAI GET /v1/models 接口。
    返回可用的 TTS 模型列表。
    """
    return JSONResponse({
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ]
    })


# ---- OpenAI 兼容：获取单个模型信息 ----
@app.get("/v1/models/{model_id}", summary="获取模型信息", tags=["OpenAI 兼容"])
async def retrieve_model(model_id: str):
    """
    兼容 OpenAI GET /v1/models/{model} 接口。
    """
    if model_id != MODEL_ID:
        raise HTTPException(status_code=404, detail=f"模型 '{model_id}' 不存在")

    return JSONResponse({
        "id": MODEL_ID,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "local",
    })


# ---- 健康检查 ----
@app.get("/v1/health", summary="健康检查", tags=["系统"])
async def health_check():
    """返回服务状态和模型信息"""
    return JSONResponse({
        "status": "ready" if (engine and engine.ready) else "not_ready",
        "model": MODEL_ID,
        "voice": AVAILABLE_VOICES[0],
        "sample_rate": SAMPLE_RATE,
        "format": f"int{BITS_PER_SAMPLE}le, {CHANNELS}ch, {SAMPLE_RATE}Hz",
    })


# ---- OpenAI 兼容核心接口：语音合成 ----
@app.post("/v1/audio/speech",
          summary="创建语音（OpenAI 兼容）",
          tags=["OpenAI 兼容"],
          response_class=StreamingResponse,
          responses={
              200: {"description": "流式音频数据",
                    "content": {"audio/wav": {}, "audio/pcm": {}}},
              503: {"description": "引擎未就绪"},
          })
async def create_speech(req: SpeechRequest):
    """
    兼容 OpenAI POST /v1/audio/speech 接口。

    流式返回合成的语音音频数据。

    **支持的格式：**
    - `wav`：标准 WAV 文件（44 字节头 + int16le PCM），可直接保存为 .wav
    - `pcm`：纯 int16le PCM 数据流，采样率 24000Hz，单声道

    **与 OpenAI SDK 配合使用：**
    ```python
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8100/v1", api_key="not-needed")
    response = client.audio.speech.create(
        model="qwen3-tts-x",
        voice="test1",
        input="你好世界",
    )
    response.stream_to_file("output.wav")
    ```
    """
    if not engine or not engine.ready:
        raise HTTPException(status_code=503, detail="引擎未就绪，请稍后重试")

    # 解析音频格式
    output_format = req.response_format.lower()
    # OpenAI 标准格式映射：不支持 mp3/opus/aac/flac，退回为 wav
    if output_format in ("mp3", "opus", "aac", "flac"):
        print(f"⚠️ [API] 不支持 {output_format} 格式，自动退回为 wav")
        output_format = "wav"
    if output_format not in ("wav", "pcm"):
        raise HTTPException(status_code=400, detail=f"不支持的格式: {req.response_format}，请使用 wav 或 pcm")

    content_type = "audio/wav" if output_format == "wav" else "audio/pcm"

    # 响应头
    headers = {
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-Channels": str(CHANNELS),
        "X-Bits-Per-Sample": str(BITS_PER_SAMPLE),
        "Transfer-Encoding": "chunked",
    }

    async def audio_generator():
        """异步生成器：边推理边 yield 音频数据块"""
        t_start = time.time()
        chunk_count = 0
        total_samples = 0

        # WAV 格式：先发送标准文件头
        if output_format == "wav":
            yield make_wav_header()

        # 获取推理锁（串行排队）
        async with stream_lock:
            audio_queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            # 保存原始 play_q.put
            original_put = stream.decoder.play_q.put

            def api_mock_put(item, *args, **kwargs):
                """截获解码器输出的音频块，转发到异步队列"""
                if getattr(item, "msg_type", "") == "AUDIO" and item.audio is not None:
                    loop.call_soon_threadsafe(audio_queue.put_nowait, item.audio)

            try:
                stream.decoder.play_q.put = api_mock_put

                # 构建推理配置
                config = TTSConfig(
                    max_steps=1000,
                    temperature=req.temperature,
                    seed=req.seed if req.seed >= 0 else None,
                    streaming=True,
                )

                # 后台推理任务
                async def run_inference():
                    try:
                        await asyncio.to_thread(
                            stream.clone, req.input, req.language, config
                        )
                    except Exception as e:
                        print(f"❌ [API] 推理异常: {e}")
                    finally:
                        loop.call_soon_threadsafe(audio_queue.put_nowait, None)

                inference_task = asyncio.create_task(run_inference())

                # 从队列中取出音频块，float32 → int16 后 yield
                while True:
                    chunk = await audio_queue.get()
                    if chunk is None:
                        break

                    chunk_count += 1
                    pcm_f32 = np.clip(chunk, -1.0, 1.0)
                    pcm_i16 = (pcm_f32 * 32767).astype(np.int16)
                    total_samples += len(pcm_i16)
                    yield pcm_i16.tobytes()

                await inference_task

            except Exception as e:
                print(f"❌ [API] 生成器异常: {e}")
                raise

            finally:
                stream.decoder.play_q.put = original_put

        elapsed = time.time() - t_start
        duration = total_samples / SAMPLE_RATE
        rtf = elapsed / duration if duration > 0 else 0
        print(f"✅ [API] 合成完毕 | {chunk_count} 块 | {duration:.2f}s 音频 | {elapsed:.2f}s 耗时 | RTF: {rtf:.2f}")

    return StreamingResponse(
        audio_generator(),
        media_type=content_type,
        headers=headers,
    )


# =====================================================================
# 启动入口
# =====================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
