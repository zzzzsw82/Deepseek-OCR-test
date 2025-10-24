from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="deepseek-ai/DeepSeek-OCR",
    local_dir="models/DeepSeek-OCR",
    resume_download=True,
    local_dir_use_symlinks=False,
)