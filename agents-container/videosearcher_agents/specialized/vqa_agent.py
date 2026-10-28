"""
VQA Agent — Visual Question Answering via local VLM.

The model runs directly in this process (no MCP server).
Swap the model by setting VQA_MODEL_NAME env var and rebuilding the container.
"""

import os
import sys
import glob
import logging
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from agents import Agent, function_tool
from videosearcher_agents.llm_utils import build_llm, resolve_llm_config

logger = logging.getLogger(__name__)

# VLMs-Testing-Framework source baked into the image at /vlm_src
sys.path.insert(0, "/vlm_src")

VQA_MODEL_NAME = os.getenv("VQA_MODEL_NAME", "microsoft/Phi-4-reasoning-vision-15B")

# CodeCarbon Layer 1 (LLM use — VQA model). The VQA model runs in-process on the
# GPU, so EmissionsTracker can wrap the inference directly. Energy lands in the
# shared emissions dir (mounted to logs/emissions) under project "vqa_phi4_inference".
EMISSIONS_DIR = os.getenv("EMISSIONS_DIR", "/app/emissions")
try:
    from codecarbon import EmissionsTracker as _EmissionsTracker
    _CODECARBON_OK = True
except Exception:  # pragma: no cover - codecarbon optional
    _CODECARBON_OK = False


def _energy_tracker(project_name: str):
    """Return a started EmissionsTracker, or None if unavailable. Never raises."""
    if not _CODECARBON_OK:
        return None
    try:
        os.makedirs(EMISSIONS_DIR, exist_ok=True)
        t = _EmissionsTracker(
            project_name=project_name,
            output_dir=EMISSIONS_DIR,
            log_level="error",
            save_to_file=True,
        )
        t.start()
        return t
    except Exception as exc:
        logger.warning("Energy tracker (%s) failed to start: %s", project_name, exc)
        return None

# Lazy-loaded singleton — model loads on first tool call, not at import time
_vlm = None


def _ensure_phi4_cache_patched() -> bool:
    """
    Patch Phi-4-reasoning-vision-15B's custom modeling code to work with
    transformers >= 4.40 (DynamicCache).

    # TODO: this patch addresses DynamicCache tuple-indexing incompatibility in
    # transformers >= 4.40 and SigLIP2 in >= 5.5. Verify and remove when upgrading
    # transformers past those versions.

    The upstream file modeling_phi4_visionr.py uses the legacy tuple format:
        target_shape = past_key_values[-1][-1].shape[-2] + 1
    which crashes with 'DynamicCache' object is not subscriptable on newer
    transformers versions. This patch replaces the lookup with a compatible
    expression that works on both old tuples and new DynamicCache objects.

    Idempotent: returns True only if a patch was actually applied during this
    call (caller uses this signal to reload the model). Only touches files
    whose path contains the Phi-4 model id.
    """
    hf_home = os.environ.get("HF_HOME", "/app/hf_cache")
    pattern = os.path.join(
        hf_home,
        "modules/transformers_modules/microsoft/Phi_hyphen_4_hyphen_reasoning_hyphen_vision_hyphen_15B/*/modeling_phi4_visionr.py",
    )
    old = "target_shape = past_key_values[-1][-1].shape[-2] + 1"
    new = (
        'target_shape = (past_key_values.get_seq_length() '
        'if hasattr(past_key_values, "get_seq_length") '
        'else past_key_values[-1][-1].shape[-2]) + 1'
    )

    any_applied = False
    for fp in glob.glob(pattern):
        try:
            with open(fp, "r") as f:
                src = f.read()
            if new in src:
                continue  # already patched
            if old not in src:
                logger.warning("Phi-4 patch target not found in %s (upstream changed?)", fp)
                continue
            with open(fp, "w") as f:
                f.write(src.replace(old, new))
            logger.info("Applied Phi-4 DynamicCache patch to %s", fp)
            any_applied = True
        except Exception as e:
            logger.warning("Failed to apply Phi-4 patch to %s: %s", fp, e)
    return any_applied


def _ensure_siglip2_shim() -> None:
    """
    Make filter_out_non_signature_kwargs accessible from siglip2's
    image_processing_siglip2 module.

    Phi-4's custom processor imports the decorator from
    transformers.models.siglip2.image_processing_siglip2, but in
    transformers >= 5.5 the helper lives in transformers.utils and is no
    longer re-exported from the siglip2 module. Decorator lookups at class
    definition time in modeling_phi4_visionr.py fail with AttributeError.

    We attach the decorator to the module object before the modeling file is
    imported. Idempotent — no-op if already present.
    """
    try:
        import transformers.models.siglip2.image_processing_siglip2 as siglip2_ips
    except ImportError:
        return
    if hasattr(siglip2_ips, "filter_out_non_signature_kwargs"):
        return
    try:
        from transformers.utils import filter_out_non_signature_kwargs
    except ImportError:
        logger.warning("transformers.utils.filter_out_non_signature_kwargs unavailable")
        return
    siglip2_ips.filter_out_non_signature_kwargs = filter_out_non_signature_kwargs
    logger.info("Installed siglip2 filter_out_non_signature_kwargs shim")


def _load_phi4_vlm():
    """Load Phi-4 in bfloat16 directly to CUDA.

    The GB10 has 128 GB unified memory. With ~106 GB available (total system
    usage is ~14 GB at rest) the model loads without memory pressure.
    bitsandbytes is NOT used: it falls back to CPU on GB10 (unrecognised arch)
    and the CPU-side quantization temporarily doubles memory usage.
    bfloat16 + device_map="cuda:0" loads cleanly without bitsandbytes.

    low_cpu_mem_usage=True loads one shard at a time through CPU then moves
    it to CUDA, keeping instantaneous CPU RAM usage minimal.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM
    _ensure_siglip2_shim()

    print(f"Loading {VQA_MODEL_NAME} in bfloat16 to CUDA (~29 GB CUDA)…")
    processor = AutoProcessor.from_pretrained(VQA_MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        VQA_MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()

    # Build a duck-type object compatible with _phi4_predict().
    class _VLM:
        pass
    vlm = _VLM()
    vlm.model = model
    vlm.processor = processor
    vlm.device = "cuda:0"
    return vlm


def _drop_phi4_modules():
    """Remove cached phi4 modules from sys.modules so a re-import picks up the
    patched file."""
    for mod_name in list(sys.modules.keys()):
        lname = mod_name.lower()
        if "phi4_visionr" in lname or "phi_hyphen_4" in lname:
            del sys.modules[mod_name]


def _get_vlm():
    """Lazy-load Phi-4 VQA model (singleton)."""
    global _vlm
    if _vlm is not None:
        return _vlm

    import torch

    # Patch + load + re-patch on first-ever download
    _ensure_phi4_cache_patched()
    _vlm = _load_phi4_vlm()
    if _ensure_phi4_cache_patched():
        logger.info("Phi-4 model file was just patched — reloading model")
        del _vlm
        _vlm = None
        torch.cuda.empty_cache()
        _drop_phi4_modules()
        _vlm = _load_phi4_vlm()

    return _vlm


def _phi4_predict(vlm, img, question: str, max_new_tokens: int) -> str:
    """
    Phi-4-reasoning-vision inference using the format from the model's own
    prepare_inputs_for_generation helper:
      1. Prepend <image>\\n to the prompt
      2. Wrap in user message and apply the TOKENIZER's chat template
      3. Pass through processor with text + images
    HuggingFaceVLM.predict() does not work for this model because it calls
    processor.apply_chat_template() (missing chat_template attribute).
    """
    import torch

    tokenizer = vlm.processor.tokenizer
    prompt = f"<image>\n{question}"
    messages = [{"role": "user", "content": prompt}]
    full_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = vlm.processor(text=full_prompt, images=[img], return_tensors="pt")
    inputs = {
        k: (v.to(vlm.device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
    }

    with torch.no_grad():
        out = vlm.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()


@function_tool
async def answer_visual_question_tool(
    frame_path: str,
    question: str,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    """
    Answer a question about a video frame using a local VLM.

    Args:
        frame_path:      Absolute path to the frame image (e.g. /data/frames/.../frame_000001.jpg)
        question:        The question to ask about the image (natural language)
        max_new_tokens:  Max tokens to generate (default 512)

    Returns:
        success, frame_path, question, answer, model
    """
    frame_p = Path(frame_path)
    if not frame_p.exists():
        # LLMs sometimes mistype the hex folder name; fall back to the last extracted frame.
        last_frame_file = Path("/tmp/last_extracted_frame.txt")
        if last_frame_file.exists():
            fallback = Path(last_frame_file.read_text().strip())
            if fallback.exists():
                frame_p = fallback
            else:
                return {"success": False, "error": f"Frame not found: {frame_path}"}
        else:
            return {"success": False, "error": f"Frame not found: {frame_path}"}

    try:
        import asyncio
        import torch
        from PIL import Image

        img = Image.open(frame_p).convert("RGB")
        vlm = _get_vlm()
        predict_fn = _phi4_predict

        # Run blocking inference in a thread to avoid blocking the event loop.
        # Layer 1 energy: start the tracker HERE (event-loop / main thread) — not
        # inside the worker thread, where CodeCarbon's signal handlers fail.
        loop = asyncio.get_event_loop()
        _tracker = _energy_tracker("vqa_phi4_inference")
        try:
            answer = await loop.run_in_executor(
                None, lambda: predict_fn(vlm, img, question, max_new_tokens)
            )
        finally:
            if _tracker is not None:
                try:
                    _tracker.stop()
                except Exception:
                    pass
    except Exception as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "frame_path": frame_path,
        "question": question,
        "answer": answer,
        "model": VQA_MODEL_NAME,
    }


class VqaAgent:
    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        llm_model_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_base_url: Optional[str] = None,
    ):
        cfg = resolve_llm_config(
            agent_type="specialized",
            llm_api_key=llm_api_key,
            llm_model_name=llm_model_name,
            llm_provider=llm_provider,
            llm_base_url=llm_base_url,
        )
        model = build_llm(cfg)

        self.agent = Agent(
            name="VqaAgent",
            model=model,
            instructions=(
                "You answer visual questions about image frames using a local "
                "vision-language model (Phi-4).\n"
                "ALWAYS call answer_visual_question_tool(frame_path, question) "
                "immediately on the first turn. Never ask the user (or the "
                "orchestrator) for clarification.\n"
                "Extract frame_path and question from the input verbatim. If "
                "the question is vague like 'what do you see?' or 'describe "
                "this frame', pass it through as-is — the VLM handles vague "
                "questions fine.\n"
                "If the input gives only a path and no explicit question, use "
                "'Describe this image in detail.' as the question.\n"
                "After the tool returns, report the answer field verbatim. Do "
                "not paraphrase, summarise, or add caveats. Do not fabricate "
                "visual details the tool did not return."
            ),
            tools=[answer_visual_question_tool],
        )
