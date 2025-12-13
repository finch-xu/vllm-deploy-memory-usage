import os
from pydantic import BaseModel
from typing import Optional

import uvicorn
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse



# 1. 加载环境变量
load_dotenv()
SERVER_HF_TOKEN = os.getenv("HF_TOKEN")

# Pydantic 模型定义
class CalculateMemoryRequest(BaseModel):
    params_b: float
    precision: float
    ctx_len: int
    max_seqs: int
    batch_size: int
    total_vram: float
    tp_size: int
    hidden_size: int = 4096
    layers: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    is_mla: bool = False
    kv_lora_rank: int = 512
    qk_rope_dim: int = 64

app = FastAPI()

# 挂载静态文件目录（当前目录）
app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_val(obj, keys, default=0):
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return default

@app.get("/api/model_specs")
def get_model_specs(model_id: str, token: Optional[str] = None):
    headers = {}
    # if token:
    #     headers["Authorization"] = f"Bearer {token}"

    headers["Authorization"] = f"Bearer {SERVER_HF_TOKEN}"

    # 1. 直接调用 HuggingFace Model API (获取元数据 + Config)
    # 相比直接下载 config.json，这个 API 能提供 safetensors 的权威参数统计
    api_url = f"https://huggingface.co/api/models/{model_id}"

    try:
        resp = requests.get(api_url, headers=headers, timeout=10)
        if resp.status_code == 401:
            raise HTTPException(401, "权限不足 (Token Invalid)")
        if resp.status_code != 200:
            raise HTTPException(404, "模型未找到")

        data = resp.json()
    except Exception as e:
        raise HTTPException(500, f"HF API Error: {str(e)}")

    # 1.5 下载完整的 config.json (HF API 的 config 字段不完整)
    config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        config_resp = requests.get(config_url, headers=headers, timeout=10)
        if config_resp.status_code == 200:
            config = config_resp.json()
        else:
            # 如果下载失败，回退到 API 的 config 字段
            config = data.get("config", {})
    except:
        # 如果下载失败，回退到 API 的 config 字段
        config = data.get("config", {})
    
    # 2. 尝试从 safetensors 元数据获取"权威"的总参数量 (Total Params)
    # 这是 HF API 唯一能帮我们省事的地方，它告诉我们精确的物理参数量
    safetensors_info = data.get("safetensors", {})
    parameters_dict = safetensors_info.get("parameters", {})

    # 优先级：选择参数量最大的精度（通常是量化后的主要权重）
    # 例如 DeepSeek-V3.2: BF16=3.95B, F8_E4M3=681.41B, F32=0.04B
    # 应该选择 F8_E4M3=681.41B（这是实际加载到显存的总参数量）
    total_params_official = 0
    if parameters_dict:
        # 选择最大的参数量
        total_params_official = max(parameters_dict.values())

    # 如果 API 没给参数量，我们后续自己算
    official_params_b = total_params_official / 1e9 if total_params_official else 0

    # 3. 深入解析 Config (为了算 KV Cache 和 MoE)
    # 处理嵌套结构 (VL / DeepSeek)
    llm = config
    is_vl = False
    if "text_config" in config:
        llm = config["text_config"]
        is_vl = True
    elif "model_config" in config:
        llm = config["model_config"]

    # 提取维度信息
    h = get_val(llm, ['hidden_size', 'd_model'])
    layers = get_val(llm, ['num_hidden_layers', 'n_layer'])
    attn_heads = get_val(llm, ['num_attention_heads', 'n_head'])
    kv_heads = get_val(llm, ['num_key_value_heads', 'n_kv_head'], default=attn_heads)
    
    # DeepSeek MLA 检测
    kv_lora_rank = get_val(llm, ['kv_lora_rank'])
    rope_dim = get_val(llm, ['qk_rope_head_dim'])
    is_mla = kv_lora_rank > 0

    # MoE 检测
    n_experts = get_val(llm, ['num_experts', 'n_routed_experts'])
    is_moe = n_experts > 1

    # FP8 检测
    quant_config = config.get("quantization_config", {})
    is_fp8 = quant_config.get("quant_method") == "fp8"

    # 4. 如果 HF API 没给权威参数量，或者我们需要区分 MoE 激活参数，则手动计算
    # 注意：vLLM 显存占用看的是【总参数量】(加载所有专家)，不是激活参数量
    # 但如果是 Dense 模型，我们自己算的通常比 HF 给的更准（HF 有时会漏算 embedding）
    
    calc_params_b = 0
    vocab = get_val(llm, ['vocab_size'], 32000)
    
    # 手动计算逻辑 (同之前，略微简化)
    # Embedding
    calc_params_b += (vocab * h) / 1e9
    
    # Layers
    attn_w = 4 * h * h
    ffn_w = 0
    
    if is_moe:
        expert_size = get_val(llm, ['moe_intermediate_size']) or get_val(llm, ['intermediate_size'])
        n_shared = get_val(llm, ['n_shared_experts'])
        shared_size = get_val(llm, ['intermediate_size']) if n_shared > 0 else 0
        ffn_w = (n_experts * 3 * h * expert_size) + (n_shared * 3 * h * shared_size) + (n_experts * h)
    else:
        inter = get_val(llm, ['intermediate_size']) or (h * 4)
        ffn_w = 3 * h * inter

    calc_params_b += (layers * (attn_w + ffn_w)) / 1e9

    # 5. 决策：使用哪个参数量？
    # 如果 HF 给的数据和我们算的差异巨大（比如 MoE），通常我们算的更符合 vLLM 加载逻辑
    # 对于 MoE，HF 的 safetensors.parameters 通常是总参数量，这是正确的显存依据
    
    final_params_b = official_params_b if official_params_b > 0 else calc_params_b

    return {
        "model_id": data.get("modelId", model_id),
        "params_b": round(final_params_b, 2), # 总参数量 (B)
        "calc_params_b": round(calc_params_b, 2), # 校验用
        "hidden_size": h,
        "layers": layers,
        "kv_heads": kv_heads,
        "attn_heads": attn_heads,
        "head_dim": get_val(llm, ['head_dim', 'qk_nope_head_dim']) or (h // attn_heads if attn_heads else 0),
        "is_moe": is_moe,
        "num_experts": n_experts,
        "is_mla": is_mla,
        "kv_lora_rank": kv_lora_rank,
        "qk_rope_dim": rope_dim,
        "is_vl": is_vl,
        "is_fp8": is_fp8,
        "max_position_embeddings": get_val(llm, ['max_position_embeddings'], 8192),
        "vocab_size": vocab
    }

# 提供 index.html 文件
@app.get("/")
async def read_root():
    return FileResponse("index.html")

# 兼容前端的 API 端点
@app.get("/api/fetch_config")
def fetch_config(model_id: str, token: Optional[str] = None):
    """兼容前端的 API 端点，实际调用 get_model_specs"""
    return get_model_specs(model_id, token)

# 提供参数选项和默认值
@app.get("/api/get_options")
def get_options():
    """返回所有参数选项和默认值"""
    return {
        "precision": {
            "options": [
                {"value": "4", "label": "F32 (4 Bytes)"},
                {"value": "2", "label": "BF16 (2 Bytes)"},
                {"value": "2.1", "label": "FP16 (2 Bytes)"},
                {"value": "1", "label": "F8_E4M3 (1 Byte)"},
                {"value": "1.1", "label": "FP8 / INT8 (1 Byte)"},
                {"value": "0.5", "label": "AWQ / INT4 (0.5 Byte)"}
            ],
            "default": "2"
        },
        "ctx_len": {
            "options": [
                {"value": "8192", "label": "8K"},
                {"value": "32768", "label": "32K"},
                {"value": "65536", "label": "64K"},
                {"value": "131072", "label": "128K"},
                {"value": "262144", "label": "256K"}
            ],
            "default": "8192"
        },
        "max_seqs": {
            "options": [
                {"value": "1", "label": "1 (Minimal)"},
                {"value": "4", "label": "4"},
                {"value": "8", "label": "8"},
                {"value": "12", "label": "12"},
                {"value": "16", "label": "16"},
                {"value": "24", "label": "24"},
                {"value": "32", "label": "32"},
                {"value": "64", "label": "64"},
                {"value": "256", "label": "256 (Default)"}
            ],
            "default": "256"
        },
        "batch_size": {
            "default": 1,
            "min": 1
        },
        "total_vram": {
            "default": 80,
            "min": 1
        },
        "tp_size": {
            "options": [
                {"value": "1", "label": "1 GPU (单卡)"},
                {"value": "2", "label": "2 GPUs (TP=2)"},
                {"value": "4", "label": "4 GPUs (TP=4)"},
                {"value": "8", "label": "8 GPUs (TP=8)"},
                {"value": "16", "label": "16 GPUs (TP=16)"},
                {"value": "32", "label": "32 GPUs (TP=32)"},
                {"value": "64", "label": "64 GPUs (TP=64)"},
                {"value": "128", "label": "128 GPUs (TP=128)"}
            ],
            "default": "1"
        }
    }

@app.post("/api/calculate_memory")
def calculate_memory(request: CalculateMemoryRequest):
    """
    计算显存占用

    参数:
    - params_b: 模型参数量（Billion）
    - precision: 精度（字节数）
    - ctx_len: 上下文长度
    - max_seqs: 最大序列数
    - batch_size: 批次大小
    - total_vram: 单卡总显存（GB）
    - tp_size: Tensor Parallel 大小
    - hidden_size: 隐藏层大小
    - layers: 层数
    - kv_heads: KV heads 数量
    - head_dim: Head 维度
    - is_mla: 是否使用 MLA
    - kv_lora_rank: KV LoRA rank
    - qk_rope_dim: QK RoPE 维度
    """

    # 1. 计算总权重
    wGB_total = request.params_b * request.precision

    # 2. 计算 Overhead
    base = 2.0
    act_factor = (request.max_seqs * request.ctx_len * request.hidden_size) / 1.6e10
    oGB = base + act_factor

    # 3. 计算 KV Cache
    if request.is_mla:
        # MLA 压缩逻辑
        elem_size = request.kv_lora_rank + request.qk_rope_dim
        kv_bytes = request.layers * elem_size * 2 * request.ctx_len * request.batch_size
    else:
        # 标准 GQA 逻辑
        kv_bytes = 2 * request.layers * request.kv_heads * request.head_dim * 2 * request.ctx_len * request.batch_size

    kGB_total = kv_bytes / (1024**3)

    # 4. Tensor Parallel 分片计算
    wGB_per_gpu = wGB_total / request.tp_size
    kGB_per_gpu = kGB_total / request.tp_size
    oGB_per_gpu = oGB  # 每个 GPU 都需要完整的 Overhead
    tGB_per_gpu = wGB_per_gpu + oGB_per_gpu + kGB_per_gpu

    # 5. 计算利用率
    utilization = (tGB_per_gpu / request.total_vram) * 100

    # 6. 确定状态
    if utilization > 100:
        status = "error"
        status_text = "显存不足"
    elif utilization > 95:
        status = "warning"
        status_text = "显存紧张"
    else:
        status = "normal"
        status_text = "正常"

    return {
        "weights": {
            "total": round(wGB_total, 2),
            "per_gpu": round(wGB_per_gpu, 2)
        },
        "overhead": {
            "total": round(oGB, 2),
            "per_gpu": round(oGB_per_gpu, 2)
        },
        "kv_cache": {
            "total": round(kGB_total, 2),
            "per_gpu": round(kGB_per_gpu, 2)
        },
        "total_memory": {
            "total": round(tGB_per_gpu * request.tp_size, 2),
            "per_gpu": round(tGB_per_gpu, 2)
        },
        "utilization": round(utilization, 1),
        "status": status,
        "status_text": status_text,
        "tp_size": request.tp_size,
        "total_vram": request.total_vram
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)