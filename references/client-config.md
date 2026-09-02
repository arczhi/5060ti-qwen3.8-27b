# SPDX-License-Identifier: Apache-2.0

# Client Configuration

Replace every value in angle brackets. Keep the model alias identical to the value returned by `GET /v1/models`.

## Pi and Codex

Use the llama.cpp endpoint directly:

```text
Base URL: http://<model-host>:8024/v1
Model: qwen3.8-27b-ud-iq4-xs-mtp1
Model page: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
Context: 81920 on the server by default; choose a smaller client window if compaction is needed
Temperature: 0.2-0.7 for coding, depending on the agent
```

The `/v1` suffix is required for OpenAI-compatible clients. Do not use the Claude adapter for Pi or Codex unless the client only speaks Anthropic Messages.

## Claude Code

Run the optional adapter on a separate port and point Claude Code at the adapter root, not at `/v1`:

```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "<private-token>",
    "ANTHROPIC_BASE_URL": "http://<model-host>:8025",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3.8-27b-ud-iq4-xs-mtp1",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3.8-27b-ud-iq4-xs-mtp1",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.8-27b-ud-iq4-xs-mtp1"
  }
}
```

The adapter accepts `/v1/messages` and `/messages`, normalizes system/developer messages, converts tools, and forwards the real Qwen alias upstream. It does not provide vision; use text-only coding requests.

## Troubleshooting order

1. `curl http://<model-host>:8024/v1/models`.
2. Send a short request with the exact model ID.
3. Check that Claude Code uses port `8025`, while Pi/Codex use `8024/v1`.
4. Check the adapter logs for HTTP status and upstream errors.
5. Only then test a tool call or a long coding prompt.
