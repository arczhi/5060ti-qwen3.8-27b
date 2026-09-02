#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Small Anthropic Messages -> llama.cpp OpenAI Chat Completions adapter."""

import json
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


LISTEN_HOST = os.environ.get("LISTEN_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8025"))
UPSTREAM_URL = os.environ.get(
    "UPSTREAM_URL", "http://127.0.0.1:8024/v1/chat/completions"
)
UPSTREAM_MODEL = os.environ.get(
    "UPSTREAM_MODEL", "qwen3.8-27b-ud-iq4-xs-mtp1-5060ti"
)
# Keep authentication enabled unless the operator explicitly changes this
# placeholder in a private environment file.
PROXY_TOKEN = os.environ.get("PROXY_TOKEN", "replace-with-private-token")
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "1800"))


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def text_from_content(content):
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if not isinstance(content, list):
        return str(content)
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type in ("text", "thinking"):
            parts.append(str(block.get("text", block.get("thinking", ""))))
        elif block_type == "image" or block_type == "image_url":
            raise ValueError("The configured text-only model does not accept images")
    return "\n".join(part for part in parts if part)


def tool_result_messages(content):
    if not isinstance(content, list):
        return []
    results = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_result":
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": str(block.get("tool_use_id", "")),
                    "content": text_from_content(block.get("content", "")),
                }
            )
    return results


def normalize_messages(body):
    """Move every system/developer instruction into one leading system message."""
    system_parts = []
    if body.get("system"):
        system_parts.append(text_from_content(body["system"]))

    messages = body.get("messages") or []
    for message in messages:
        if message.get("role") in ("system", "developer"):
            content = text_from_content(message.get("content", ""))
            if content:
                system_parts.append(content)

    normalized = []
    if system_parts:
        normalized.append({"role": "system", "content": "\n\n".join(system_parts)})

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role in ("system", "developer"):
            continue
        if role == "user":
            tool_results = tool_result_messages(content)
            text = text_from_content(content)
            if text:
                normalized.append({"role": "user", "content": text})
            normalized.extend(tool_results)
            if not text and not tool_results:
                normalized.append({"role": "user", "content": ""})
            continue
        if role == "assistant":
            assistant = {"role": "assistant"}
            if isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(str(block.get("text", "")))
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": str(block.get("id", "toolu_" + uuid.uuid4().hex[:12])),
                                "type": "function",
                                "function": {
                                    "name": str(block.get("name", "")),
                                    "arguments": json.dumps(
                                        block.get("input", {}),
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    ),
                                },
                            }
                        )
                assistant["content"] = "\n".join(text_parts) or None
                if tool_calls:
                    assistant["tool_calls"] = tool_calls
            else:
                assistant["content"] = text_from_content(content)
            normalized.append(assistant)
            continue
        if role == "tool":
            normalized.append(
                {
                    "role": "tool",
                    "tool_call_id": str(message.get("tool_call_id", "")),
                    "content": text_from_content(content),
                }
            )
            continue
        raise ValueError(f"Unsupported message role: {role}")
    return normalized


def normalize_tools(tools):
    result = []
    for tool in tools or []:
        if tool.get("type") == "function" and "function" in tool:
            result.append(tool)
            continue
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            }
        )
    return result


def openai_body(body):
    result = {
        "model": UPSTREAM_MODEL,
        "messages": normalize_messages(body),
        "stream": bool(body.get("stream", False)),
        "max_tokens": int(body.get("max_tokens", 4096)),
    }
    for key in ("temperature", "top_p", "stop"):
        if key in body:
            result[key] = body[key]
    tools = normalize_tools(body.get("tools"))
    if tools:
        result["tools"] = tools
    if result["stream"]:
        result["stream_options"] = {"include_usage": True}
    return result


def anthropic_message(response, requested_model):
    choices = response.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments", "{}"))
        except json.JSONDecodeError:
            arguments = {}
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") or "toolu_" + uuid.uuid4().hex[:12],
                "name": function.get("name", ""),
                "input": arguments,
            }
        )
    finish = choice.get("finish_reason")
    stop_reason = "tool_use" if finish == "tool_calls" else "max_tokens" if finish == "length" else "end_turn"
    usage = response.get("usage") or {}
    return {
        "id": "msg_" + uuid.uuid4().hex,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": response.get("model") or requested_model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        },
    }


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n".encode("utf-8")


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {fmt % args}", flush=True)

    def send_json(self, status, value):
        payload = json_bytes(value)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_HEAD(self):
        path = urlsplit(self.path).path
        if path in ("/", "/health", "/v1/models", "/models"):
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/health"):
            self.send_json(200, {"status": "ok", "upstream": UPSTREAM_URL})
            return
        if path in ("/v1/models", "/models"):
            self.send_json(
                200,
                {
                    "data": [
                        {
                            "id": UPSTREAM_MODEL,
                            "object": "model",
                            "owned_by": "local",
                        }
                    ],
                    "object": "list",
                },
            )
            return
        self.send_json(404, {"type": "error", "error": {"type": "not_found", "message": "Not found"}})

    def do_POST(self):
        path = urlsplit(self.path).path
        if path not in ("/v1/messages", "/messages"):
            self.send_json(404, {"type": "error", "error": {"type": "not_found", "message": "Use /v1/messages"}})
            return
        if PROXY_TOKEN:
            supplied = self.headers.get("x-api-key", "") or self.headers.get("authorization", "")
            if PROXY_TOKEN not in supplied:
                self.send_json(401, {"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key"}})
                return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            upstream_payload = json_bytes(openai_body(body))
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": str(exc)}})
            return
        started = time.monotonic()
        request = Request(
            UPSTREAM_URL,
            data=upstream_payload,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + PROXY_TOKEN},
            method="POST",
        )
        try:
            upstream = urlopen(request, timeout=REQUEST_TIMEOUT)
            if body.get("stream", False):
                self.stream_response(upstream, UPSTREAM_MODEL)
            else:
                response = json.loads(upstream.read())
                self.send_json(200, anthropic_message(response, body.get("model") or UPSTREAM_MODEL))
            self.log_message("POST %s -> 200 in %.2fs", self.path, time.monotonic() - started)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.send_json(502, {"type": "error", "error": {"type": "api_error", "message": detail[:4000]}})
            self.log_message("POST %s -> upstream %s in %.2fs", self.path, exc.code, time.monotonic() - started)
        except (URLError, TimeoutError, ValueError) as exc:
            self.send_json(502, {"type": "error", "error": {"type": "api_error", "message": str(exc)}})
            self.log_message("POST %s -> proxy error in %.2fs: %s", self.path, time.monotonic() - started, exc)

    def stream_response(self, upstream, requested_model):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()
        message_id = "msg_" + uuid.uuid4().hex
        self.wfile.write(
            sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": requested_model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        )
        self.wfile.flush()
        block_index = -1
        block_type = None
        tool_indexes = {}
        output_tokens = 0
        input_tokens = 0
        stop_reason = "end_turn"

        def close_block():
            nonlocal block_index, block_type
            if block_index >= 0:
                self.wfile.write(sse("content_block_stop", {"type": "content_block_stop", "index": block_index}))
                self.wfile.flush()
                block_index = -1
                block_type = None

        def start_block(kind, block):
            nonlocal block_index, block_type
            close_block()
            block_index += 1
            block_type = kind
            self.wfile.write(sse("content_block_start", {"type": "content_block_start", "index": block_index, "content_block": block}))
            self.wfile.flush()

        for raw_line in upstream:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage") or {}
            input_tokens = int(usage.get("prompt_tokens", input_tokens))
            output_tokens = int(usage.get("completion_tokens", output_tokens))
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            finish = choice.get("finish_reason")
            if finish:
                stop_reason = "tool_use" if finish == "tool_calls" else "max_tokens" if finish == "length" else "end_turn"
            text = delta.get("content")
            if text:
                if block_type != "text":
                    start_block("text", {"type": "text", "text": ""})
                self.wfile.write(sse("content_block_delta", {"type": "content_block_delta", "index": block_index, "delta": {"type": "text_delta", "text": text}}))
                self.wfile.flush()
            for call in delta.get("tool_calls") or []:
                call_index = call.get("index", 0)
                if call_index not in tool_indexes:
                    start_block(
                        "tool_use",
                        {
                            "type": "tool_use",
                            "id": call.get("id") or "toolu_" + uuid.uuid4().hex[:12],
                            "name": (call.get("function") or {}).get("name", ""),
                            "input": {},
                        },
                    )
                    tool_indexes[call_index] = block_index
                current_index = tool_indexes[call_index]
                arguments = (call.get("function") or {}).get("arguments", "")
                if arguments:
                    self.wfile.write(sse("content_block_delta", {"type": "content_block_delta", "index": current_index, "delta": {"type": "input_json_delta", "partial_json": arguments}}))
                    self.wfile.flush()
        close_block()
        self.wfile.write(sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": output_tokens}}))
        self.wfile.write(sse("message_stop", {"type": "message_stop"}))
        self.wfile.flush()


def main():
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), ProxyHandler)
    print(f"Anthropic adapter listening on http://{LISTEN_HOST}:{LISTEN_PORT}", flush=True)
    print(f"Forwarding to {UPSTREAM_URL} as {UPSTREAM_MODEL}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
