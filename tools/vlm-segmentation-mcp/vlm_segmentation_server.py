"""
VLM Segmentation MCP Server (StreamableHTTP)

Runs GroundingDINO + SAM2 pipeline from VLMs-Testing-Framework directly.

Tools:
- segment_frame
- segment_image

Input:
- frame_path / image_path
- text_query
- box_threshold
- text_threshold
- segmentation_type
- output_mode:
    - "masks": saves and returns mask PNGs, preserving the old behavior
    - "segmented_image": saves and returns a segmented image with overlay

Output:
- masks mode:
    mask_paths, scores, labels, boxes
- segmented_image mode:
    segmented_image_path, scores, labels, boxes
"""

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import numpy as np

from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http import TransportSecuritySettings

# VLMs-Testing-Framework source is mounted at /vlm_src
sys.path.insert(0, "/vlm_src")
from SegmentationPipelines import GroundingDinoSam2Pipeline
from Samples import SegmentationSample

try:
    from codecarbon import EmissionsTracker as _EmissionsTracker
    _CODECARBON_OK = True
except Exception:
    _CODECARBON_OK = False

_EMISSIONS_DIR = os.getenv("EMISSIONS_DIR", "/app/emissions")

def _energy_tracker(project_name):
    """Start a CodeCarbon tracker, or return None if unavailable. Never raises."""
    if not _CODECARBON_OK:
        return None
    try:
        os.makedirs(_EMISSIONS_DIR, exist_ok=True)
        t = _EmissionsTracker(project_name=project_name, output_dir=_EMISSIONS_DIR, log_level="error")
        t.start()
        return t
    except Exception:
        return None


logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

PORT = int(os.getenv("PORT", "8008"))
HOST = os.getenv("HOST", "0.0.0.0")
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
SEG_DIR = DATA_DIR / "segmentation"
SEG_DIR.mkdir(parents=True, exist_ok=True)

# Weights live at /app/weights (named Docker volume — persists SAM2 download)
WEIGHTS_DIR = Path("/app/weights")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="VLM Segmentation MCP Server")

mcp = FastMCP(
    "vlm-segmentation-http",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

# Align FastMCP streamable HTTP path with our mount to avoid "/mcp/mcp"
mcp.settings.streamable_http_path = "/"

_mcp_session_ctx = None
_pipeline: Optional[GroundingDinoSam2Pipeline] = None
_model_loaded: bool = False


@app.on_event("startup")
async def startup():
    global _mcp_session_ctx, _pipeline, _model_loaded

    # Start MCP session manager
    _mcp_session_ctx = mcp.session_manager.run()
    await _mcp_session_ctx.__aenter__()

    # Resolve GroundingDINO config from the installed package
    import groundingdino

    gdino_config = str(
        Path(groundingdino.__file__).parent / "config" / "GroundingDINO_SwinT_OGC.py"
    )
    gdino_ckpt = str(WEIGHTS_DIR / "groundingdino_swint_ogc.pth")

    sam2_ckpt = str(WEIGHTS_DIR / "sam2_hiera_tiny.pt")
    # Both checkpoints are baked into the image at build time (see the Dockerfile);
    # Docker copies them into a fresh vlm_weights volume from the image layer on first
    # boot, so no runtime download happens. The only way SAM2 can be missing here is a
    # STALE vlm_weights volume left over from an image built before the weights were
    # baked in — Docker never repopulates a volume that already has content. Surface
    # that with an actionable message instead of the old silent FileNotFoundError,
    # which left model_loaded=false and hung the launcher at "11/12 servers ready".
    if not Path(sam2_ckpt).exists():
        logger.error(
            "SAM2 checkpoint missing at %s. This image bakes it in, so a stale "
            "vlm_weights volume is shadowing it. Reset the volume and rebuild:\n"
            "  docker compose -f docker-compose.yml down\n"
            "  docker volume rm guarnizoscappaturagarzoni_vlm_interaction_vlm_weights\n"
            "  ./local-ai-service/run_agentic_system.sh --build",
            sam2_ckpt,
        )
    logger.info("Loading GroundingDINO + SAM2 pipeline...")
    logger.info("  GroundingDINO config : %s", gdino_config)
    logger.info("  GroundingDINO ckpt   : %s", gdino_ckpt)
    logger.info("  SAM2 checkpoint      : %s", sam2_ckpt)
    logger.info("  Weights dir          : %s", WEIGHTS_DIR)

    try:
        # Use tiny SAM2 model to fit within available VRAM alongside other services.
        # Switch to sam2_hiera_large.pt / sam2_hiera_l.yaml when more VRAM is free.
        _pipeline = GroundingDinoSam2Pipeline(
            grounding_dino_config=gdino_config,
            grounding_dino_checkpoint=gdino_ckpt,
            sam2_checkpoint=sam2_ckpt,
            sam2_config="sam2_hiera_t.yaml",
            device="cuda",
        )
        _pipeline.load_model()
        _model_loaded = True
        logger.info("Pipeline loaded successfully.")
    except Exception as exc:
        logger.error("Failed to load segmentation pipeline: %s", exc, exc_info=True)
        # _model_loaded stays False; /health will report it; tool calls return clear error


@app.on_event("shutdown")
async def shutdown():
    global _mcp_session_ctx

    if _mcp_session_ctx is not None:
        await _mcp_session_ctx.__aexit__(None, None, None)
        _mcp_session_ctx = None


@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "model_loaded": _model_loaded})


def _mask_to_uint8(mask: Any, image_size: tuple[int, int]) -> np.ndarray:
    """
    Converts a mask returned by the segmentation pipeline to a 0/255 uint8 mask.

    image_size is PIL format: (width, height).
    """
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    mask_arr = np.asarray(mask)
    mask_arr = np.squeeze(mask_arr)

    if mask_arr.ndim > 2:
        mask_arr = mask_arr[0]

    mask_arr = (mask_arr > 0).astype(np.uint8) * 255

    mask_img = Image.fromarray(mask_arr, mode="L")
    if mask_img.size != image_size:
        mask_img = mask_img.resize(image_size, Image.Resampling.NEAREST)

    return np.array(mask_img, dtype=np.uint8)


def _build_combined_mask(masks: Any, image_size: tuple[int, int]) -> Optional[Image.Image]:
    """
    Builds a single union mask from all instance masks.
    """
    if masks is None or len(masks) == 0:
        return None

    combined_mask = None

    for mask in masks:
        mask_arr = _mask_to_uint8(mask, image_size)

        if combined_mask is None:
            combined_mask = mask_arr
        else:
            combined_mask = np.maximum(combined_mask, mask_arr)

    if combined_mask is None:
        return None

    return Image.fromarray(combined_mask, mode="L")


def _save_segmented_image(
    img: Image.Image,
    masks: Any,
    output_path: Path,
) -> Optional[str]:
    """
    Saves one segmented image with a transparent red overlay over all masks.

    This function does not save individual masks.
    """
    combined_mask = _build_combined_mask(masks, img.size)

    if combined_mask is None:
        return None

    base = img.convert("RGBA")

    red_layer = Image.new("RGBA", base.size, (255, 0, 0, 115))
    transparent_layer = Image.new("RGBA", base.size, (255, 0, 0, 0))
    overlay = Image.composite(red_layer, transparent_layer, combined_mask)

    segmented = Image.alpha_composite(base, overlay).convert("RGB")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    segmented.save(output_path)

    return str(output_path)


def _save_masks(
    masks: Any,
    image_size: tuple[int, int],
    out_dir: Path,
) -> list[str]:
    """
    Saves individual mask PNGs, preserving the original behavior.
    """
    mask_paths = []

    if masks is None or len(masks) == 0:
        return mask_paths

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, mask in enumerate(masks):
        mask_arr = _mask_to_uint8(mask, image_size)
        mask_img = Image.fromarray(mask_arr, mode="L")

        mask_path = out_dir / f"mask_{i:03d}.png"
        mask_img.save(mask_path)
        mask_paths.append(str(mask_path))

    return mask_paths


async def _segment_impl(
    image_path: str,
    text_query: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    segmentation_type: str = "instance",
    output_mode: str = "masks",
    output_image_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if not _model_loaded:
        return {
            "success": False,
            "error": "Model not loaded yet — try again in a few seconds",
        }

    if output_mode not in {"masks", "segmented_image"}:
        return {
            "success": False,
            "error": "output_mode must be one of: masks, segmented_image",
        }

    image_p = Path(image_path)
    if not image_p.exists():
        return {"success": False, "error": f"Image not found: {image_path}"}

    run_id = uuid.uuid4().hex[:8]
    base_dir = Path(output_dir) if output_dir else SEG_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir = base_dir / f"{image_p.stem}_{run_id}"

    img = Image.open(image_p).convert("RGB")
    text_queries = [query.strip() for query in text_query.split(",") if query.strip()]

    if not text_queries:
        return {
            "success": False,
            "error": "text_query must contain at least one non-empty object description",
        }

    try:
        result = _pipeline.predict(
            SegmentationSample(
                sample_id=run_id,
                image=img,
                text_queries=text_queries,
            ),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
    except Exception as exc:
        logger.exception("Pipeline predict failed")
        return {"success": False, "error": str(exc)}

    masks = result.get("masks")
    scores = result.get("scores", np.array([]))
    labels = result.get("labels", [])
    boxes = result.get("boxes", np.array([]))

    scores_list = [float(score) for score in scores] if len(scores) > 0 else []
    boxes_list = boxes.tolist() if hasattr(boxes, "tolist") and len(boxes) > 0 else []

    response: Dict[str, Any] = {
        "success": True,
        "image_path": image_path,
        "frame_path": image_path,
        "text_query": text_query,
        "output_mode": output_mode,
        "scores": scores_list,
        "labels": list(labels),
        "boxes": boxes_list,
        "output_dir": str(out_dir),
        "params": {
            "box_threshold": box_threshold,
            "text_threshold": text_threshold,
            "segmentation_type": segmentation_type,
        },
    }

    if output_mode == "masks":
        mask_paths = _save_masks(
            masks=masks,
            image_size=img.size,
            out_dir=out_dir,
        )
        response["mask_paths"] = mask_paths
        return response

    if output_mode == "segmented_image":
        if output_image_path:
            segmented_image_path = Path(output_image_path)
            # A relative path used to resolve against the container CWD (/app,
            # the bind-mounted source tree) instead of the shared output volume,
            # writing an unreachable artifact and returning a relative path no
            # other container could resolve. Anchor it under out_dir.
            if not segmented_image_path.is_absolute():
                segmented_image_path = out_dir / segmented_image_path
        else:
            segmented_image_path = out_dir / "segmented_image.png"

        saved_segmented_image_path = _save_segmented_image(
            img=img,
            masks=masks,
            output_path=segmented_image_path,
        )

        response["segmented_image_path"] = saved_segmented_image_path
        return response

    return response


@mcp.tool()
async def segment_frame(
    frame_path: str,
    text_query: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    segmentation_type: str = "instance",
    output_mode: str = "masks",
    output_image_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Segment objects in a video frame using GroundingDINO + SAM2.

    output_mode:
    - "masks": saves individual mask PNGs and returns mask_paths.
      Use this for internal pipelines, VQA, post-processing, or compatibility.
    - "segmented_image": saves a visual segmented image and returns segmented_image_path.
      Use this when the user asks for image/frame segmentation as the final output.

    output_dir: Optional output directory override (uses DATA_DIR/segmentation if omitted)
    """
    _tracker = _energy_tracker("image_segmentation")
    try:
        return await _segment_impl(
            image_path=frame_path,
            text_query=text_query,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            segmentation_type=segmentation_type,
            output_mode=output_mode,
            output_image_path=output_image_path,
            output_dir=output_dir,
        )
    finally:
        if _tracker is not None:
            try:
                _tracker.stop()
            except Exception:
                pass


@mcp.tool()
async def segment_image(
    image_path: str,
    text_query: str,
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    segmentation_type: str = "instance",
    output_mode: str = "masks",
    output_image_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Alias for segment_frame, using image_path instead of frame_path.

    output_dir: Optional output directory override (uses DATA_DIR/segmentation if omitted)
    """
    _tracker = _energy_tracker("image_segmentation")
    try:
        return await _segment_impl(
            image_path=image_path,
            text_query=text_query,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            segmentation_type=segmentation_type,
            output_mode=output_mode,
            output_image_path=output_image_path,
            output_dir=output_dir,
        )
    finally:
        if _tracker is not None:
            try:
                _tracker.stop()
            except Exception:
                pass


app.mount("/mcp", mcp.streamable_http_app())

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)