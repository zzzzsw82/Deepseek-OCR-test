from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_PROMPT = "<image>\n<|grounding|>Convert the document to markdown."
DEFAULT_MODEL_ID = "deepseek-ai/DeepSeek-OCR"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DeepSeek-OCR model on a folder of images and export Markdown + CSV summaries."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory where Markdown files and results.csv will be written.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model identifier to download from the hub.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Local path to a previously downloaded DeepSeek-OCR model. "
            "If provided, --model-id is ignored and the model is loaded offline."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force transformers to use only local files (offline mode).",
    )
    parser.add_argument(
        "--attn-impl",
        type=str,
        default="flash_attention_2",
        help=(
            "Attention implementation passed to AutoModel.from_pretrained. "
            "Set to '' to disable and fall back to the default implementation."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Prompt sent to the OCR model. Defaults to Markdown conversion.",
    )
    parser.add_argument(
        "--base-size",
        type=int,
        default=1024,
        help="Base image resolution used by DeepSeek-OCR inference.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Image size used by DeepSeek-OCR inference.",
    )
    parser.add_argument(
        "--no-crop",
        dest="crop_mode",
        action="store_false",
        help="Disable the crop_mode flag (enabled by default).",
    )
    parser.set_defaults(crop_mode=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Computation dtype used for the model.",
    )
    parser.add_argument(
        "--enable-preprocessing",
        action="store_true",
        help="Run GPU-accelerated preprocessing (deskew/crop/resize) before inference.",
    )
    parser.add_argument(
        "--preprocess-max-size",
        type=int,
        default=1536,
        help="Maximum longer-side resolution for preprocessed images to control VRAM (pixels).",
    )
    parser.add_argument(
        "--preprocess-min-size",
        type=int,
        default=1024,
        help="Target minimum shorter-side resolution when upscaling documents (pixels).",
    )
    parser.add_argument(
        "--preprocess-margin",
        type=float,
        default=0.02,
        help="Extra margin ratio to keep around detected content during cropping.",
    )
    parser.add_argument(
        "--preprocess-sharpen",
        type=float,
        default=0.15,
        help="Unsharp-mask strength applied after resizing (0 disables).",
    )
    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Persist preprocessed images alongside outputs for inspection.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity.",
    )
    return parser.parse_args(argv)


def find_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Image directory is not a folder: {folder}")

    images = [
        path
        for path in sorted(folder.rglob("*"))
        if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file()
    ]

    if not images:
        raise FileNotFoundError(
            f"No supported images found under {folder}. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )
    return images


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError(
        "CUDA device not available. DeepSeek-OCR's reference implementation requires an NVIDIA GPU."
    )


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model_and_tokenizer(args: argparse.Namespace):
    model_source = str(args.model_path) if args.model_path else args.model_id
    load_kwargs = dict(
        trust_remote_code=True,
        use_safetensors=True,
        local_files_only=args.local_files_only or args.model_path is not None,
    )

    if args.attn_impl:
        load_kwargs["_attn_implementation"] = args.attn_impl

    tokenizer = AutoTokenizer.from_pretrained(model_source, **load_kwargs)

    try:
        model = AutoModel.from_pretrained(model_source, **load_kwargs)
    except TypeError:
        # Some installs might not support the chosen attention implementation; retry without it.
        if "_attn_implementation" in load_kwargs:
            logging.warning(
                "Falling back to default attention implementation because %s is not supported.",
                args.attn_impl,
            )
            load_kwargs.pop("_attn_implementation")
            model = AutoModel.from_pretrained(model_source, **load_kwargs)
        else:
            raise

    device = resolve_device()
    dtype = dtype_from_name(args.dtype)
    model = model.eval().to(device)
    if dtype != torch.float32:
        model = model.to(dtype=dtype)

    return tokenizer, model, device


def run_inference(
    model,
    tokenizer,
    image_path: Path,
    prompt: str,
    output_dir: Path,
    base_size: int,
    image_size: int,
    crop_mode: bool,
    device: torch.device,
    args: argparse.Namespace,
    persist_dir: Optional[Path],
) -> str:
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:
        tmp_path = Path(tmp_dir)
        os.makedirs(tmp_path, exist_ok=True)
        image_for_inference = maybe_preprocess_image(
            image_path=image_path,
            scratch_dir=tmp_path,
            device=device,
            args=args,
            persist_dir=persist_dir,
        )
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(image_for_inference),
            output_path=str(tmp_path),
            base_size=base_size,
            image_size=image_size,
            crop_mode=crop_mode,
            eval_mode=True,
            save_results=False,
        )
    if not isinstance(result, str):
        raise RuntimeError(
            f"Inference did not return text for image {image_path}. Got: {type(result)}"
        )
    return result.strip()


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def release_model(model, tokenizer) -> None:
    """Move model back to CPU and reset CUDA cache to free VRAM."""
    if model is not None:
        try:
            model.to("cpu")
        except Exception:  # best-effort
            pass
    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def preprocess_image_gpu(
    image_path: Path,
    scratch_dir: Path,
    device: torch.device,
    max_size: int,
    min_size: int,
    margin_ratio: float,
    sharpen_strength: float,
) -> Path:
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        np_img = np.asarray(rgb, dtype=np.uint8)

    tensor = torch.from_numpy(np_img).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0
    del np_img
    height, width = tensor.shape[1:]

    with torch.inference_mode():
        gray = (0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]).unsqueeze(0).unsqueeze(0)
        downscale = max(1, min(height, width) // 512)
        if downscale > 1:
            pooled = F.avg_pool2d(gray, kernel_size=downscale, stride=downscale)
            mask = pooled < 0.98
            mask = F.interpolate(mask.float(), size=(height, width), mode="nearest").squeeze(0).squeeze(0) > 0
        else:
            mask = gray.squeeze(0).squeeze(0) < 0.98

        mask = mask.float().unsqueeze(0).unsqueeze(0)
        mask = F.max_pool2d(mask, kernel_size=31, stride=1, padding=15)
        mask = mask.squeeze(0).squeeze(0) > 0
        coords = torch.nonzero(mask)

        if coords.numel() > 0:
            y_min = int(coords[:, 0].min().item())
            y_max = int(coords[:, 0].max().item())
            x_min = int(coords[:, 1].min().item())
            x_max = int(coords[:, 1].max().item())
            margin_pixels = int(round(margin_ratio * max(height, width)))
            y_min = max(y_min - margin_pixels, 0)
            y_max = min(y_max + margin_pixels, height - 1)
            x_min = max(x_min - margin_pixels, 0)
            x_max = min(x_max + margin_pixels, width - 1)
            cropped = tensor[:, y_min : y_max + 1, x_min : x_max + 1]
        else:
            cropped = tensor

        min_side = min(cropped.shape[1], cropped.shape[2])
        max_side = max(cropped.shape[1], cropped.shape[2])

        scale_up = min_size / min_side if min_size and min_side < min_size else 1.0
        scale_down = max_size / max_side if max_size and max_side > max_size else 1.0
        scale = 1.0
        if scale_up > scale:
            scale = scale_up
        if scale_down < scale:
            scale = scale_down

        if abs(scale - 1.0) > 1e-3:
            target_h = max(int(round(cropped.shape[1] * scale)), 8)
            target_w = max(int(round(cropped.shape[2] * scale)), 8)
            cropped = F.interpolate(
                cropped.unsqueeze(0),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if sharpen_strength > 0:
            blurred = F.avg_pool2d(cropped.unsqueeze(0), kernel_size=5, stride=1, padding=2)
            enhanced = cropped.unsqueeze(0) + sharpen_strength * (cropped.unsqueeze(0) - blurred)
            cropped = enhanced.squeeze(0).clamp(0.0, 1.0)

    processed = (cropped.permute(1, 2, 0).clamp(0.0, 1.0) * 255).to(torch.uint8).cpu().numpy()
    processed_image = Image.fromarray(processed, mode="RGB")
    output_path = scratch_dir / f"{image_path.stem}_preprocessed.png"
    processed_image.save(output_path)
    del tensor, cropped, processed  # free GPU memory
    return output_path


def maybe_preprocess_image(
    image_path: Path,
    scratch_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    persist_dir: Optional[Path],
) -> Path:
    if not args.enable_preprocessing:
        return image_path

    preprocessed = preprocess_image_gpu(
        image_path=image_path,
        scratch_dir=scratch_dir,
        device=device,
        max_size=args.preprocess_max_size,
        min_size=args.preprocess_min_size,
        margin_ratio=args.preprocess_margin,
        sharpen_strength=max(args.preprocess_sharpen, 0.0),
    )

    if persist_dir is not None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(preprocessed, persist_dir / preprocessed.name)
    return preprocessed


def run_batch(args: argparse.Namespace, *, configure_logging: bool = False) -> dict:
    if configure_logging and not logging.getLogger().handlers:
        logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")

    output_dir = ensure_output_dir(args.output_dir)
    images = find_images(args.image_dir)
    tokenizer = model = None  # type: ignore[assignment]
    device_obj: Optional[torch.device] = None
    persist_dir = output_dir / "preprocessed" if args.save_preprocessed else None

    records = []
    markdown_paths: List[str] = []
    summary: Dict[str, Any]

    try:
        tokenizer, model, device_str = load_model_and_tokenizer(args)
        device_obj = torch.device(device_str)

        for image_path in tqdm(images, desc="Processing images"):
            try:
                markdown_text = run_inference(
                    model=model,
                    tokenizer=tokenizer,
                    image_path=image_path,
                    prompt=args.prompt,
                    output_dir=output_dir,
                    base_size=args.base_size,
                    image_size=args.image_size,
                    crop_mode=args.crop_mode,
                    device=device_obj,
                    args=args,
                    persist_dir=persist_dir,
                )
            except Exception as exc:
                logging.error("Failed to process %s: %s", image_path, exc)
                records.append(
                    {
                        "image": image_path.name,
                        "markdown_file": "",
                        "status": "error",
                        "error": str(exc),
                        "text": "",
                    }
                )
                continue

            markdown_file = output_dir / f"{image_path.stem}.md"
            markdown_file.write_text(markdown_text + "\n", encoding="utf-8")
            markdown_paths.append(str(markdown_file))
            records.append(
                {
                    "image": image_path.name,
                    "markdown_file": markdown_file.name,
                    "status": "ok",
                    "error": "",
                    "text": markdown_text,
                }
            )

        results_csv = output_dir / "results.csv"
        pd.DataFrame.from_records(records).to_csv(results_csv, index=False, encoding="utf-8")

        if configure_logging:
            logging.info("Processed %d images. Markdown and CSV saved to %s", len(records), output_dir)

        summary = {
            "records": records,
            "results_csv": results_csv,
            "markdown_files": markdown_paths,
            "output_dir": output_dir,
        }
    finally:
        if model is not None and tokenizer is not None:
            release_model(model, tokenizer)

    return summary


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_batch(args, configure_logging=True)


if __name__ == "__main__":
    main()
