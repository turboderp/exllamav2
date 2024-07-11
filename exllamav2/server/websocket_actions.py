
# from exllamav2 import (
#     ExLlamaV2,
#     ExLlamaV2Config,
#     ExLlamaV2Cache,
#     ExLlamaV2Tokenizer
# )

import json
import asyncio

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

async def dispatch(request, ws, server):

    action_ = request["action"]

    response = { "action": action_ }
    if "request_id" in request: response["request_id"] = request["request_id"]
    if "response_id" in request: response["response_id"] = request["response_id"]

    if action_ == "echo": echo(request, ws, server, response)
    elif action_ == "estimate_token": estimate_token(request, ws, server, response)
    elif action_ == "lefttrim_token": lefttrim_token(request, ws, server, response)
    elif action_ == "infer": await infer(request, ws, server, response)
    elif action_ == "stop": stop(request, ws, server, response)

    else:
        print(f" ## Unknown request from client: {request}")
        return

    await ws.send(json.dumps(response))


def echo(request, ws, server, response):

    """
    request:  { action: str = "echo",
                request_id: str,                    # (optional) request ID to echo in response packet
                response_id: str }                  # (optional) response ID to echo in response packet

    response: { action: str = "echo",
                request_id: str,                    # (optional)
                response_id: str }                  # (optional)
    """

    pass


def estimate_token(request, we, server, response):

    """
    request:  { action: str = "estimate_token",
                request_id: str,                    # (optional) request ID to echo in response packet
                response_id: str,                   # (optional) response ID to echo in response packet
                text: str }                         # text to measure

    response: { action: str = "estimate_token",
                request_id: str,                    # (optional)
                response_id: str,                   # (optional)
                num_tokens: int }                   # length of input text, in tokens
    """

    text = request["text"]
    ids = server.tokenizer.cached_encode_str(text)
    response["num_tokens"] = ids.shape[-1]


def lefttrim_token(request, ws, server, response):

    """
    request:  { action: str = "lefttrim_token",
                request_id: str,                    # (optional) request ID to echo in response packet
                response_id: str,                   # (optional) response ID to echo in response packet
                text: str,                          # text to trim
                trimmed_length: int }               # num tokens to keep, from right

    response: { action: str = "lefttrim_token",
                request_id: str,                    # (optional)
                response_id: str,                   # (optional)
                trimmed_text: str }                 # input, trimmed
    """

    text = request["text"]
    length = int(request["trimmed_length"])

    ids = server.tokenizer.encode(text, encode_special_tokens = True)
    if ids.shape[-1] <= length:
        response["trimmed_text"] = text
    else:
        response["trimmed_text"] = server.tokenizer.decode(ids[:, -length:], decode_special_tokens = True)[0]


async def infer(request, ws, server, response):

    """
    request:  { action: str = "infer",
                request_id: str,                    # (optional) request ID to echo in response packet
                response_id: str,                   # (optional) response ID to echo in response packet
                text: str,                          # input prompt
                max_new_tokens: int,                # max num new tokens
                stream: bool,                       # stream response
                stream_full: bool,                  # return full response-so-far with each streamed chunk
                top_p: float,                       # (optional) top-P threshold (0 to disable)
                top_k: int,                         # (optional) top-K count (0 to disable)
                top_a: float,                       # (optional) top-A threshold (0 to disable)
                min_p: float,                       # (optional) min-P threshold (0 to disable)
                typical: float,                     # (optional) typical threshold (0 to disable)
                temperature: float,                 # (optional) sampling temperature (1.0 = no temp adjust)
                rep_pen: float,                     # (optional) repetition penalty (1.0 = no penalty)
                freq_pen: float,                    # (optional) frequency penalty (0.0 = no penalty)
                pres_pen: float,                    # (optional) presence penalty (0.0 = no penalty)
                skew: float,                        # (optional) skew factor (0.0 = disabled)
                customBos: str,                     # (optional) custom BOS token
                stop_conditions: [str|int],         # (optional) list of stop conditions
                token_healing: bool,                # (optionsl) enable token healing
                tag: str }                          # (optional) tag to echo in response packet

    streams:  { action: str = "infer",
                request_id: str,                    # (optional)
                response_id: str,                   # (optional)
                response_type: str = "chunk",
                chunk: str,                         # next chunk of response
                tag: str }                          # (optional)

    response: { action: str = "infer",
                request_id: str,                    # (optional)
                response_id: str,                   # (optional)
                response_type: str = "full",
                util_text: str,                     # input context (pruned if max_seq_len exceeded)
                response: str,                      # full response excluding input prompt
                tag: str,                           # (optional)
                stop_reason: str }                  # "eos", "num_tokens" or "interrupted"
    """

    async with server.model_lock:

        server.stop_signal.clear()

        # Mode

        stream = request["stream"]
        if "tag" in request:
            response["tag"] = request["tag"]

        # Stop conditions

        sc = [server.tokenizer.eos_token_id]
        if "stop_conditions" in request:
            ss = request["stop_conditions"]
            if not isinstance(ss, list): ss = [ss]
            sc += ss

        if "bann_bann" in request:
            bb = request["bann_bann"]
            if not isinstance(bb, list): bb = [bb]
        else:
            bb = None

        # Full response

        full_response = request['stream_full'] if 'stream_full' in request else False
        # Tokenize and trim prompt

        full_ctx = request["text"]
        num_tokens = request["max_new_tokens"]

        cb=''
        if 'customBos' in request:
            cb = request['customBos']
        ids = server.tokenizer.cached_encode_str(cb+full_ctx)
        overflow = ids.shape[-1] + num_tokens - server.model.config.max_seq_len


        if overflow < 0:
            util_ctx = cb+full_ctx

        elif 'customBos' in request:
            ids = ids[:,:1]+ids[:, overflow+1:]
            util_ctx = server.tokenizer.decode(ids)

        else:
            ids = ids[:, overflow:]
            util_ctx = server.tokenizer.decode(ids)
            

        # Sampler

        gs = ExLlamaV2Sampler.Settings()
        gs.top_k = int(request["top_k"]) if "top_k" in request else 100
        gs.top_p = float(request["top_p"]) if "top_p" in request else 0.8
        gs.top_a = float(request["top_a"]) if "top_a" in request else 0
        gs.min_p = float(request["min_p"]) if "min_p" in request else 0
        gs.typical = float(request["typical"]) if "typical" in request else 0
        gs.temperature = float(request["temperature"]) if "temperature" in request else 0.9
        gs.skew = float(request["skew"]) if "skew" in request else 0.0
        gs.token_repetition_penalty = float(request["rep_pen"]) if "rep_pen" in request else 1.05
        gs.token_frequency_penalty = float(request["freq_pen"]) if "freq_pen" in request else 0.0
        gs.token_presence_penalty = float(request["pres_pen"]) if "pres_pen" in request else 0.0
        if bb is not None:
            gs.disallow_tokens(server.tokenizer, bb)

        # Generate

        server.generator.set_stop_conditions(sc)
        server.generator.begin_stream(ids, gs, token_healing = request["token_healing"] if "token_healing" in request else False)

        completion = ""
        gen_tokens = 0
        response["util_text"] = util_ctx
        while True:
            chunk, eos, _ = server.generator.stream()
            completion += chunk
            gen_tokens += 1

            if stream and chunk != "":
                response["response_type"] = "chunk"
                response["chunk"] = chunk
                if full_response: response["response"] = completion
                await ws.send(json.dumps(response))
            response["chunk"] = ''

            if eos:
                response["stop_reason"] = "eos"
                break

            if gen_tokens >= num_tokens:
                response["stop_reason"] = "num_tokens"
                break

            await asyncio.sleep(0)

            if server.stop_signal.is_set():
                server.stop_signal.clear()
                response["stop_reason"] = "interrupted"
                break

        #if stream: del response["chunk"]
        response["response_type"] = "full"

        response["response"] = completion


def stop(request, ws, server, response):

    """
    request:  { action: str = "stop",
                request_id: str,                    # (optional) request ID to echo in response packet
                response_id: str }                  # (optional) response ID to echo in response packet

    response: { action: str = "stop",
                request_id: str,                    # (optional)
                response_id: str }                  # (optional)
    """

    server.stop_signal.set()
