
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    model_init,
)

from exllamav2.server import (
    ExLlamaV2WebSocketServer
)

import argparse

# Configure and init model

parser = argparse.ArgumentParser(description = "WebSocket server example")
parser.add_argument("-host", "--host", type = str, default = "0.0.0.0:7862", help = "IP:PORT eg, 0.0.0.0:7862")

model_init.add_args(parser)
args = parser.parse_args()
model_init.check_args(args)
model_init.print_options(args)
model, tokenizer = model_init.init(args)

# Create cache

cache = ExLlamaV2Cache(model)

# Create server

ip, port = args.host.split(":")
port = int(port)

server = ExLlamaV2WebSocketServer(ip, port, model, tokenizer, cache)
server.serve()
