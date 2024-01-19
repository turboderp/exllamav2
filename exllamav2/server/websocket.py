
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

import exllamav2.server.websocket_actions as actions

import websockets, asyncio
import json
import threading, asyncio

class ExLlamaV2WebSocketServer:

    ip: str
    port: int

    model: ExLlamaV2
    tokenizer: ExLlamaV2Tokenizer
    cache: ExLlamaV2Cache
    generator = ExLlamaV2StreamingGenerator

    stop_signal = threading.Event()
    model_lock = asyncio.Lock()
    active_requests: list


    def __init__(self, ip: str, port: int, model: ExLlamaV2, tokenizer: ExLlamaV2Tokenizer, cache: ExLlamaV2Cache):

        self.ip = ip
        self.port = port
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache

        self.generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

        self.stop_signal.clear()
        self.active_requests = []


    def serve(self):

        print(f" -- Starting WebSocket server on {self.ip} port {self.port}")

        start_server = websockets.serve(self.main, self.ip, self.port)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()


    async def main(self, websocket, path):

        async for message in websocket:

            request = json.loads(message)
            r = asyncio.create_task(actions.dispatch(request, websocket, self))
            self.active_requests.append(r)
            self.active_requests = [r for r in self.active_requests if not r.done()]
