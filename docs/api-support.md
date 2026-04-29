# API support matrix

What hippo_leap supports today vs the full OpenAI API surface.

## Endpoints

| Endpoint | Status | Notes |
|---|---|---|
| `GET /health` | Supported | Custom health check (not part of OpenAI API) |
| `GET /v1/models` | Supported | Returns loaded model info |
| `POST /v1/chat/completions` | Partial | Non-streaming only |
| `POST /v1/responses` | Not yet | See v2.5 roadmap |
| `POST /v1/embeddings` | Not yet | Would need embedding model support |
| `GET /v1/models/:id` | Not yet | |
| `POST /v1/audio/transcriptions` | Not yet | Out of scope |
| `POST /v1/fine_tuning/jobs` | Not yet | Out of scope |

## POST /v1/chat/completions

### Request fields (CreateChatCompletionReq)

| Field | Status | Notes |
|---|---|---|
| `messages` | Supported | Extracts text from last user message |
| `model` | Accepted | Echoed back in response, not validated against loaded model |
| `max_tokens` | Supported | Falls back to 256 if not set |
| `stream` | Partial | Recognized — returns 400 error. Streaming not yet implemented |
| `temperature` | Not yet | Always greedy argmax (temperature=0 behavior) |
| `top_p` | Not yet | |
| `frequency_penalty` | Not yet | |
| `presence_penalty` | Not yet | |
| `stop` | Not yet | Only stops on EOS token |
| `seed` | Not yet | Deterministic by default (argmax) |
| `n` | Not yet | Always generates 1 completion |
| `logprobs` | Not yet | |
| `top_logprobs` | Not yet | |
| `logit_bias` | Not yet | |
| `response_format` | Not yet | |
| `tools` | Not yet | No function calling |
| `tool_choice` | Not yet | |
| `user` | Ignored | Accepted but not used |

### Message content

| Content type | Status | Notes |
|---|---|---|
| Text (`type: "text"`) | Supported | Extracts from last user message |
| Image (`type: "image_url"`) | Not yet | Would need vision model |
| System messages | Partial | Present in messages but only last user text is extracted |
| Multi-turn conversation | Not yet | Only uses last user message, ignores conversation history |
| Tool results (`role: "tool"`) | Not yet | |

### Response fields (CreateChatCompletionResp)

| Field | Status | Notes |
|---|---|---|
| `id` | Supported | `chatcmpl-{epoch_timestamp}` |
| `object` | Supported | Always `chat.completion` |
| `created` | Supported | Unix timestamp |
| `model` | Supported | Echoes request model |
| `choices` | Supported | Always exactly 1 choice |
| `choices[].message.content` | Supported | Generated text |
| `choices[].message.role` | Supported | Always `assistant` |
| `choices[].finish_reason` | Partial | Always `stop` (doesn't distinguish length vs EOS) |
| `choices[].delta` | Not yet | Streaming only |
| `choices[].log_probs` | Not yet | |
| `system_fingerprint` | Supported | Always `hippo_leap` |
| `usage.prompt_tokens` | Supported | Actual tokenized count |
| `usage.total_tokens` | Supported | prompt + completion |
| `usage.completion_tokens` | Missing | Not in openai_leap Usage type (only prompt + total) |

## Sampling

| Feature | Status | Notes |
|---|---|---|
| Greedy (argmax) | Supported | Default and only mode |
| Temperature sampling | Not yet | |
| Top-p (nucleus) | Not yet | |
| Top-k | Not yet | |
| Repetition penalty | Not yet | |
| Min-p | Not yet | |

## Server features

| Feature | Status | Notes |
|---|---|---|
| Concurrent requests | Partial | Lock-based — requests queue, one inference at a time |
| Request batching | Not yet | Each request runs independently |
| Prompt caching | Not yet | KV cache resets every request |
| Continuous batching | Not yet | |
| Streaming (SSE) | Not yet | |
| CORS headers | Not yet | |
| Authentication | Not yet | No API key validation |
| Rate limiting | Not yet | |
| Request logging | Not yet | |
| Graceful shutdown | Not yet | |

## Conversation handling

Currently the server is **stateless** — each request tokenizes the full prompt
from scratch and resets the KV cache. There is no conversation memory between
requests.

### What "multi-turn" means today

The server extracts only the **last user message** text. System messages and
conversation history in the `messages` array are ignored. To have a coherent
conversation, the client would need to concatenate all prior context into a
single user message.

### What proper multi-turn needs

1. **Full message history** — concatenate all messages (system + user +
   assistant turns) into the prompt, not just the last user message
2. **Prompt caching** — reuse KV cache from the shared prefix of a
   conversation instead of recomputing from scratch
3. **Session management** — track conversations by ID so the KV cache
   persists across HTTP requests
