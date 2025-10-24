from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import torch
from tqdm import tqdm
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
) -> str:
    with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:
        tmp_path = Path(tmp_dir)
        os.makedirs(tmp_path, exist_ok=True)
        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=str(image_path),
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


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s: %(message)s")

    output_dir = ensure_output_dir(args.output_dir)
    images = find_images(args.image_dir)
    tokenizer, model, _device = load_model_and_tokenizer(args)

    records = []

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
    logging.info("Processed %d images. Markdown and CSV saved to %s", len(records), output_dir)


if __name__ == "__main__":
    main()
