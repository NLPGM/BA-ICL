
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"


from huggingface_hub import snapshot_download


model = "princeton-nlp/sup-simcse-bert-base-uncased"
local_dir=f"{model}"


snapshot_download(
  repo_id=model,
  local_dir=f"{local_dir}",
  local_dir_use_symlinks=False,
  max_workers=8,
)