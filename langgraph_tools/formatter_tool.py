from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


DEFAULT_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "formatter_prompt.txt"


class FormatterToolInput(BaseModel):
    """Inputs for formatting Markdown OCR outputs into a CSV summary."""

    markdown_dir: str = Field(..., description="Directory containing Markdown files from OCR.")
    output_csv: Optional[str] = Field(
        None, description="Destination CSV file path. Defaults to <markdown_dir>/formatted.csv."
    )
    prompt_path: Optional[str] = Field(
        None,
        description="Path to the prompt template used for formatting. Defaults to prompts/formatter_prompt.txt.",
    )
    model: str = Field(
        "qwen3:14b", description="Ollama model name used for formatting."
    )
    base_url: Optional[str] = Field(
        None,
        description="Base URL for the Ollama server. Defaults to http://localhost:11434 if omitted.",
    )
    temperature: float = Field(
        0.0, description="Temperature passed to the Ollama chat model."
    )


def _collect_markdown(markdown_dir: Path) -> List[Path]:
    files = sorted(markdown_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No Markdown files found under {markdown_dir}.")
    return files


def _load_prompt(prompt_path: Optional[str]) -> str:
    path = Path(prompt_path).resolve() if prompt_path else DEFAULT_PROMPT_PATH
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found at {path}")
    return path.read_text(encoding="utf-8")


def _format_markdown_with_ollama(**kwargs: Any) -> Dict[str, Any]:
    params = FormatterToolInput(**kwargs)
    markdown_dir = Path(params.markdown_dir)
    markdown_files = _collect_markdown(markdown_dir)
    prompt_text = _load_prompt(params.prompt_path)

    # Concatenate markdown with metadata to help the formatter.
    combined_sections = []
    for md_path in markdown_files:
        combined_sections.append(f"# FILE: {md_path.name}\n\n{md_path.read_text(encoding='utf-8')}")
    combined_md = "\n\n".join(combined_sections)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
            ("human", "Markdown documents:\n\n{markdown_input}\n\nReturn the CSV text."),
        ]
    )

    model = ChatOllama(
        model=params.model,
        base_url=params.base_url,
        temperature=params.temperature,
    )

    formatted = model.invoke(
        prompt.format_prompt(markdown_input=combined_md).to_messages()
    )

    csv_text = formatted.content if hasattr(formatted, "content") else str(formatted)
    csv_text = csv_text.strip() + ("\n" if not csv_text.endswith("\n") else "")

    output_path = (
        Path(params.output_csv).resolve()
        if params.output_csv
        else (markdown_dir / "formatted.csv").resolve()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(csv_text, encoding="utf-8")

    return {
        "csv_path": str(output_path),
        "csv_text": csv_text,
        "markdown_files": [str(path) for path in markdown_files],
    }


def build_formatter_tool() -> StructuredTool:
    """Return a tool that converts Markdown outputs into CSV using a local Ollama model."""

    return StructuredTool(
        name="invoice_markdown_to_csv",
        description=(
            "Convert Markdown files produced by the OCR step into a structured CSV "
            "summary using a locally hosted Ollama model."
        ),
        func=_format_markdown_with_ollama,
        args_schema=FormatterToolInput,
    )
