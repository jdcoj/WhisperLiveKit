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

async def convert_to_rttm(audio_filename, final_response):
    """
    Convert the transcription response to RTTM format and save to a file.
    """

    if not final_response or "lines" not in final_response:
        logger.warning("No transcription lines found for RTTM conversion.")
        return

    rttm_lines = []
    for idx, line in enumerate(final_response["lines"]):
        speaker = line.get("speaker", -1)
        beg = line.get("beg", "0:00:00")
        end = line.get("end", "0:00:00")
        # Convert beg and end to seconds
        def time_to_seconds(t):
            parts = t.split(":")
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        start_time = time_to_seconds(beg)
        end_time = time_to_seconds(end)
        duration = end_time - start_time
        # RTTM format: SPEAKER <file-id> 1 <start-time> <duration> <ortho> <stype> <name> <conf> <slat>
        # We'll use audio_filename (without extension) as file-id
        file_id = audio_filename.split("/")[-1].split(".")[0]
        speaker_id = 1 if speaker == -1 else speaker
        rttm_line = f"SPEAKER {file_id} 1 {start_time:.2f} {duration:.2f} <NA> <NA> speaker{speaker_id} <NA> <NA>"
        rttm_lines.append(rttm_line)

    rttm_folder = "./rttm"
    os.makedirs(rttm_folder, exist_ok=True)
    rttm_filename = f"./rttm/{file_id}.rttm"
    with open(rttm_filename, "w", encoding="utf-8") as f:
        for line in rttm_lines:
            f.write(line + "\n")
    logger.info(f"RTTM file written to {rttm_filename}")


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

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    global transcription_engine
    audio_processor = AudioProcessor(
        transcription_engine=transcription_engine,
    )
    await websocket.accept()
    logger.info("WebSocket connection opened.")
            
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

        await audio_processor.cleanup()

        await convert_to_rttm(audio_filename, final_response)
    
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
