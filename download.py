import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

# 指定模型 ID（例如：bert-base-uncased）
model_ids = [
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4",

    # "shuyuej/Meta-Llama-3.1-8B-Instruct-GPTQ",
    # "ISTA-DASLab/gemma-3-4b-it-GPTQ-4b-128g",
    # "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",


]



# root_llm_path="../../../LLMs" # on 4090
root_llm_path = "/mnt/data/LLMS"  # on L20

for model_id in model_ids:

    # 下载到本地目录（默认缓存路径或自定义路径）
    local_dir = f"{root_llm_path}/{model_id}"  # 可选：指定本地保存路径

    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # 避免创建符号链接，直接复制文件
        resume_download=True,          # 支持断点续传
        token=None                     # 如果是私有模型，填入你的 Hugging Face Token
    )