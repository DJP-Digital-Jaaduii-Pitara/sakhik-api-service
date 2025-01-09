import time
import json
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Message
from fastapi import Request
from starlette.responses import StreamingResponse
from io import BytesIO

from logger import logger
from utils import get_from_env_or_config
from telemetry_logger import TelemetryLogger


# https://github.com/tiangolo/fastapi/issues/394 
# Stream response does not work => https://github.com/tiangolo/fastapi/issues/394#issuecomment-994665859
async def set_body(request: Request, body: bytes):
    async def receive() -> Message:
        return {"type": "http.request", "body": body}

    request._receive = receive


async def get_body(request: Request) -> bytes:
    body = await request.body()
    await set_body(request, body)
    return body

telemetryLogger =  TelemetryLogger()
telemetry_log_enabled = get_from_env_or_config('telemetry', 'telemetry_log_enabled', None).lower() == "true"

class TelemetryMiddleware(BaseHTTPMiddleware):
    def __init__(
            self,
            app
    ):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        await set_body(request, await request.body())
        body = await get_body(request)
        if body.decode("utf-8"):
            body = json.loads(body)
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        if "v1" in str(request.url):
            event: dict = {
                "status_code": response.status_code,
                "duration": round(process_time * 1000),
                "body": body,
                "method": request.method,
                "url": request.url
            }

            request_id = f"{int(time.time())}-{uuid.uuid4().hex}"
            event["x-request-id"] = request_id

            event.update(request.headers)
            
            if isinstance(response, StreamingResponse):
                response_body = b""
                async for chunk in response.body_iterator:
                    response_body += chunk
                response.body_iterator = self.iterate_in_chunks(response_body)

                event["response"] = response_body.decode('utf-8').strip()
                logger.warning({"label": "api_call_response", "response": response_body.decode('utf-8').strip()})

            logger.warning({"label": "api_call", "event": event})

            if telemetry_log_enabled:
                if response.status_code == 200:
                    event = telemetryLogger.prepare_log_event(eventInput=event, message="success")
                else:
                    event = telemetryLogger.prepare_log_event(eventInput=event, elevel="ERROR", message="failed")
                telemetryLogger.add_event(event)
        return response

    async def iterate_in_chunks(self, data: bytes, chunk_size: int = 4096):
        stream = BytesIO(data)
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            yield chunk