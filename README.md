# vLLM 部署显存计算器

一个用于计算 vLLM 模型部署时显存占用的 Web 应用，帮助你在部署大语言模型前准确评估所需的 GPU 显存。

## 主要功能

### 1. 自动获取模型配置
- 通过 Hugging Face Model ID 自动获取模型参数
- 支持从 Hugging Face API 和 config.json 获取完整配置
- 自动识别模型类型：
  - Dense 模型（标准 Transformer）
  - MoE 模型（Mixture of Experts）
  - MLA 架构（Multi-head Latent Attention，如 DeepSeek）
  - 视觉语言模型（VL Models）
  - FP8 量化模型

### 2. 精确的显存计算
- **模型权重**：根据参数量和精度计算权重占用
- **KV Cache**：支持标准 GQA 和 MLA 压缩算法
- **系统开销**：包含激活值、CUDA 上下文等开销
- **Tensor Parallel**：支持多卡并行部署的显存分配计算

### 3. 灵活的参数配置
- 精度选择：F32、BF16、FP16、FP8、INT8、INT4 等
- 上下文长度：8K ~ 256K
- 批处理大小：自定义 batch size 和 max sequences
- 多卡部署：支持 1~128 GPU 的 Tensor Parallel 配置

### 4. 直观的 Web 界面
- 实时显存占用可视化
- 显存利用率状态提示（正常/紧张/不足）
- 支持自定义 Hugging Face Token（访问私有模型）

## 技术栈

- **后端**：FastAPI + Python 3.14
- **前端**：原生 HTML/CSS/JavaScript
- **部署**：Docker + Docker Compose
- **API 集成**：Hugging Face Model API

## 快速开始

### 前置要求

- Docker
- Docker Compose
- Hugging Face Token（可选，用于访问私有模型）

### 部署步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/yourusername/vllm-deploy-memory-usage.git
   cd vllm-deploy-memory-usage
   ```

2. **配置环境变量**
   ```bash
   cp .env.example .env
   ```

   编辑 `.env` 文件，添加你的 Hugging Face Token：
   ```
   HF_TOKEN=your_huggingface_token_here
   ```

   获取 Token：[https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

3. **构建并启动服务**
   ```bash
   docker compose up -d --build
   ```

4. **访问应用**

   打开浏览器访问：[http://localhost:8018](http://localhost:8018)

### 常用命令

```bash
# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 重新构建并启动
docker-compose up -d --build
```

## 使用示例

1. 在输入框中输入 Hugging Face Model ID，例如：
   - `Qwen/Qwen3-4B-Instruct-2507`
   - `deepseek-ai/DeepSeek-V3`
   - `Qwen/Qwen3-Next-80B-A3B-Thinking`
   - `Qwen/Qwen3-VL-8B-Instruct`

2. 点击"获取配置"按钮，系统会自动填充模型参数

3. 根据需要调整部署参数：
   - 选择精度（如 BF16）
   - 设置上下文长度（如 32K）
   - 配置 GPU 数量（Tensor Parallel）
   - 输入单卡显存大小（如 80GB）

4. 查看计算结果：
   - 权重占用
   - KV Cache 占用
   - 系统开销
   - 总显存占用
   - 显存利用率

## API 文档

### 获取模型配置
```http
GET /api/model_specs?model_id={model_id}&token={optional_token}
```

### 计算显存占用
```http
POST /api/calculate_memory
Content-Type: application/json

{
  "params_b": 70,
  "precision": 2,
  "ctx_len": 32768,
  "max_seqs": 256,
  "batch_size": 1,
  "total_vram": 80,
  "tp_size": 1,
  "hidden_size": 8192,
  "layers": 80,
  "kv_heads": 8,
  "head_dim": 128,
  "is_mla": false,
  "kv_lora_rank": 0,
  "qk_rope_dim": 0
}
```

### 获取参数选项
```http
GET /api/get_options
```

## 项目结构

```
.
├── main.py                 # FastAPI 后端应用
├── index.html             # 前端界面
├── requirements.txt       # Python 依赖
├── Dockerfile            # Docker 镜像配置
├── docker-compose.yml    # Docker Compose 配置
├── .env.example          # 环境变量模板
└── README.md             # 项目文档
```

## 开发模式

如需在开发时实时更新代码，可以取消 `docker-compose.yml` 中的 volumes 注释：

```yaml
volumes:
  - ./main.py:/app/main.py
  - ./index.html:/app/index.html
```

然后重启服务：
```bash
docker-compose restart
```

## 注意事项

1. **Hugging Face Token**：访问私有模型或提高 API 限额时需要配置
2. **显存计算**：结果为理论估算值，实际部署时可能有 ±5% 的误差
3. **MoE 模型**：计算的是总参数量（所有专家），而非激活参数量
4. **时区设置**：容器默认使用 Asia/Shanghai 时区

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！