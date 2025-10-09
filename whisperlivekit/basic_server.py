from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from whisperlivekit import TranscriptionEngine, AudioProcessor, get_web_interface_html, parse_args
import asyncio
import logging
import uuid
from pydub import AudioSegment
import httpx
import os
import subprocess
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

args = parse_args()
transcription_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcription_engine
    transcription_engine = TranscriptionEngine(
        **vars(args),
    )
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get():
    return HTMLResponse(get_web_interface_html())


async def handle_websocket_results(websocket, results_generator, audio_file, audio_filename):
    """Consumes results from the audio processor and sends them via WebSocket."""
    lastest_response = None

    try:
        async for response in results_generator:
            lastest_response = response
            await websocket.send_json(response)

        audio_file.close()
        logger.info("Audio file closed.")

        mp3_filename = audio_filename.replace(".webm", ".mp3")
        await convert_webm_to_mp3(audio_filename, mp3_filename)
        logger.info(f"Converted to MP3: {mp3_filename}")

        # endpoint_url = "https://ais.skiesoft.com/v1/audio/diarization" 
        # processing_result = await send_audio_and_get_response(mp3_filename, endpoint_url)
        # logger.info("Received processing result from external endpoint")
        
        try:
            # await websocket.send_json({"type": "diarization_result", "data": processing_result})
            await websocket.send_json({"type": "test"})
        except Exception as e:
            logger.warning(f"Failed to send processing result via WebSocket: {e}")
        
        # when the results_generator finishes it means all audio has been processed
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client likely closed connection).")
    except Exception as e:
        logger.warning(f"Error in WebSocket results handler: {e}")

    return lastest_response

async def convert_webm_to_mp3(webm_path: str, mp3_path: str) -> None:
    """
    將 webm 格式音檔轉換為 mp3。
    :param webm_path: webm 檔案路徑
    :param mp3_path: mp3 輸出路徑
    """
    audio = AudioSegment.from_file(webm_path, format="webm")
    audio.export(mp3_path, format="mp3")

async def send_audio_and_get_response(mp3_path: str, endpoint_url: str) -> dict:
    """
    將 mp3 音檔傳送到指定 endpoint，並等待回傳結果。
    :param mp3_path: mp3 檔案路徑
    :param endpoint_url: API endpoint URL
    :return: 回傳的 JSON 結果
    """
    async with httpx.AsyncClient() as client:
        with open(mp3_path, "rb") as audio_file:
            files = {"data": (mp3_path.split("/")[-1], audio_file, "audio/mp3")}
            response = await client.post(endpoint_url, files=files)
            response.raise_for_status()
            return response.json()

async def verify_token(token: str) -> bool:
    """
    驗證 token 是否有效
    :param token: 要驗證的 token
    :return: 驗證是否成功
    """
    try:
        # 驗證端點 URL - 你可以根據需要修改這個 URL
        auth_endpoint = "https://developer.skiesoft.com/api/verify"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(auth_endpoint, headers=headers)
            logger.info(f"Token verification response status: {response.status_code}")
            logger.info(f"Token verification response text: {response.text}")
            return response.status_code == 200
    except httpx.TimeoutException:
        logger.warning(f"Token verification timeout for token: {token[:10]}...")
        return False
    except httpx.RequestError as e:
        logger.warning(f"Token verification request error: {e}")
        return False
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return False

async def log_usage(token: str, file_path: str, model: str):
    """
    記錄使用者的使用情況
    :param token: 使用者的 token
    :param file_path: 上傳的音訊檔案路徑
    :param model: 使用的模型名稱
    """
    try:
        usage_endpoint = "https://developer.skiesoft.com/api/usage/log"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        result = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        duration = float(json.loads(result.stdout)["format"]["duration"])
        logger.info(f"Audio duration: {duration} seconds")
        
        data = {
            "duration_s": duration,
            "model": "thiannu-v1",
            "usage_type": "realtime"
        }
        
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(usage_endpoint, json=data, headers=headers)
            response.raise_for_status()
            logger.info(f"Usage logged successfully: {response.status_code}")
    except Exception as e:
        logger.warning(f"Failed to log usage: {e}")

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine
    logger.info("New WebSocket connection attempt.")
    # 在接受連接前先驗證 Authorization header
    try:
        # 從 headers 中獲取 Authorization token
        authorization = websocket.headers.get("Authorization")
        
        if not authorization:
            await websocket.close(code=4001, reason="Missing Authorization header")
            logger.warning("WebSocket connection rejected: Missing Authorization header")
            return
        
        # 檢查 Bearer token 格式
        if not authorization.startswith("Bearer "):
            await websocket.close(code=4001, reason="Invalid Authorization header format")
            logger.warning("WebSocket connection rejected: Invalid Authorization header format")
            return
        
        # 提取 token
        token = authorization[7:]  # 移除 "Bearer " 前綴
        logger.info(f"Extracted token: {token}")
        
        # 驗證 token
        is_valid = await verify_token(token)
        
        if not is_valid:
            await websocket.close(code=4001, reason="Invalid token")
            logger.info("WebSocket connection rejected: Invalid token")
            return
        
        logger.info("WebSocket authentication successful.")
        
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        await websocket.close(code=4000, reason="Authentication error")
        return
    
    # 驗證成功後接受連接
    await websocket.accept()
    logger.info("WebSocket connection accepted after successful authentication.")
    
    # 發送認證成功訊息
    await websocket.send_json({
        "type": "auth_success"
    })
    
    # 認證成功後，繼續原本的流程
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
    )
    logger.info("Starting audio processing...")
            
    results_generator = await audio_processor.create_tasks()

    audio_folder = "./received_audio"
    os.makedirs(audio_folder, exist_ok=True)
    audio_filename = f"./received_audio/audio_session_{uuid.uuid4().hex}.webm"
    audio_file = open(audio_filename, "ab")
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator, audio_file, audio_filename))

    final_response = None

    try:
        while True:
            message = await websocket.receive_bytes()
            audio_file.write(message)
            audio_file.flush()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if 'bytes' in str(e):
            logger.warning(f"Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up WebSocket endpoint...")
        if not websocket_task.done():
            websocket_task.cancel()
        try:
            final_response = await websocket_task
        except asyncio.CancelledError:
            logger.info("WebSocket results handler task was cancelled.")
        except Exception as e:
            logger.warning(f"Exception while awaiting websocket_task completion: {e}")
            
        if not audio_file.closed:
            audio_file.close()

        # Add usage log
        await log_usage(token, audio_filename, "thiannu-v1")

        # TODO: Clear received audio files if needed

        await audio_processor.cleanup()
    
        logger.info("WebSocket endpoint cleaned up successfully.")

def main():
    """Entry point for the CLI command."""
    import uvicorn
    
    uvicorn_kwargs = {
        "app": "whisperlivekit.basic_server:app",
        "host":args.host, 
        "port":args.port, 
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }
    
    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}

    uvicorn.run(**uvicorn_kwargs)

if __name__ == "__main__":
    main()
