from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, validator

from main import parse_args, run_batch


class OCRToolInput(BaseModel):
    """Inputs required to run the DeepSeek-OCR batch pipeline."""

    image_dir: str = Field(..., description="Directory containing the input images.")
    output_dir: str = Field(..., description="Directory where Markdown/CSV outputs are stored.")
    model_path: Optional[str] = Field(None, description="Local DeepSeek-OCR model path.")
    model_id: Optional[str] = Field(None, description="Hugging Face model identifier to download if no local path.")
    local_files_only: bool = Field(
        True, description="Whether to restrict model loading to local files only."
    )
    prompt: Optional[str] = Field(
        None, description="Override the default DeepSeek OCR prompt for this run."
    )
    base_size: Optional[int] = Field(
        None, description="Override the base resolution (defaults to 1024 in the CLI)."
    )
    image_size: Optional[int] = Field(
        None, description="Override the image size (defaults to 640 in the CLI)."
    )
    crop_mode: Optional[bool] = Field(
        None, description="Explicitly enable/disable crop_mode (default is True)."
    )
    dtype: Optional[str] = Field(
        None, description="Computation dtype: one of bfloat16/float16/float32."
    )
    attn_impl: Optional[str] = Field(
        "eager",
        description="Attention implementation passed through to the model loader. Defaults to 'eager' for compatibility.",
    )
    enable_preprocessing: bool = Field(
        True, description="Enable GPU-based preprocessing before inference."
    )
    preprocess_max_size: Optional[int] = Field(
        1536, description="Maximum long-side resolution after preprocessing."
    )
    preprocess_min_size: Optional[int] = Field(
        1152, description="Minimum short-side resolution target during preprocessing."
    )
    preprocess_margin: float = Field(
        0.03, description="Margin ratio kept around detected content."
    )
    preprocess_sharpen: float = Field(
        0.2, description="Unsharp-mask strength applied after resizing."
    )
    save_preprocessed: bool = Field(
        False, description="If true, persist cleaned images next to the outputs."
    )
    log_level: str = Field(
        "INFO", description="Logging level used during the OCR run."
    )

    @validator("dtype")
    def _validate_dtype(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        allowed = {"bfloat16", "float16", "float32"}
        if value not in allowed:
            raise ValueError(f"dtype must be one of {allowed}")
        return value


def _make_namespace(data: OCRToolInput) -> Namespace:
    """Convert structured inputs into the argparse namespace used in main.run_batch."""

    cli_args: List[str] = [
        "--image-dir",
        data.image_dir,
        "--output-dir",
        data.output_dir,
        "--log-level",
        data.log_level.upper(),
    ]

    if data.model_path:
        cli_args.extend(["--model-path", data.model_path])
    elif data.model_id:
        cli_args.extend(["--model-id", data.model_id])

    if data.local_files_only:
        cli_args.append("--local-files-only")

    if data.prompt is not None:
        cli_args.extend(["--prompt", data.prompt])

    if data.base_size is not None:
        cli_args.extend(["--base-size", str(data.base_size)])

    if data.image_size is not None:
        cli_args.extend(["--image-size", str(data.image_size)])

    if data.crop_mode is False:
        cli_args.append("--no-crop")

    if data.dtype is not None:
        cli_args.extend(["--dtype", data.dtype])

    if data.attn_impl is not None:
        cli_args.extend(["--attn-impl", data.attn_impl])

    if data.enable_preprocessing:
        cli_args.append("--enable-preprocessing")

    if data.preprocess_max_size is not None:
        cli_args.extend(["--preprocess-max-size", str(data.preprocess_max_size)])
    if data.preprocess_min_size is not None:
        cli_args.extend(["--preprocess-min-size", str(data.preprocess_min_size)])

    cli_args.extend(
        [
            "--preprocess-margin",
            str(data.preprocess_margin),
            "--preprocess-sharpen",
            str(data.preprocess_sharpen),
        ]
    )

    if data.save_preprocessed:
        cli_args.append("--save-preprocessed")

    return parse_args(cli_args)


def _run_ocr_tool(**kwargs: Any) -> Dict[str, Any]:
    params = OCRToolInput(**kwargs)
    args = _make_namespace(params)
    summary = run_batch(args, configure_logging=False)
    return {
        "results_csv": str(summary["results_csv"]),
        "markdown_files": summary["markdown_files"],
        "records": summary["records"],
        "output_dir": str(summary["output_dir"]),
    }


def build_ocr_tool() -> StructuredTool:
    """Expose the DeepSeek OCR batch pipeline as a LangChain/ LangGraph tool."""

    return StructuredTool(
        name="deepseek_ocr_batch",
        description=(
            "Run DeepSeek-OCR on a directory of images, producing Markdown files "
            "and a results.csv summary."
        ),
        func=_run_ocr_tool,
        args_schema=OCRToolInput,
    )
