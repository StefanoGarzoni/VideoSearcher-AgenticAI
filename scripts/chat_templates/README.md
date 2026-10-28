# Vendored chat templates

`llama.cpp/` is gitignored (a local build), so the tool-use chat templates the
comparison harness needs are vendored here and referenced by `TEMPLATES_DIR` in
`scripts/run_model_comparison.py`.

| File | Source | Used by |
|---|---|---|
| `NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use.jinja` | llama.cpp bundled template (`llama.cpp/models/templates/`) — generic Hermes `<tool_call>` format | command-r, command-r7b, phi4-mini, deepseek-v2-lite |
| `Mistral-Small-3.2-relaxed.jinja` | **Custom** — a copy of llama.cpp's `Mistral-Small-3.2-24B-Instruct-2506.jinja` with the strict "tool-call IDs must be 9 chars" `raise_exception` replaced by a deterministic 9-char ID sanitizer (the OpenAI Agents SDK emits longer IDs) | mistral-small |

To use a stock llama.cpp template for a new model, copy it here and point the
model's `chat_template` in the REGISTRY at it. See
[`docs/MODEL_COMPARISON.md`](../../docs/MODEL_COMPARISON.md).

> Not every fix needs a template file: hybrid-thinking models (e.g. **Qwen3.6**) keep
> their native template and instead pass kwargs to it — set
> `chat_template_kwargs={"enable_thinking": False}` on the REGISTRY entry, which the
> harness forwards as `--chat-template-kwargs` at server launch.
