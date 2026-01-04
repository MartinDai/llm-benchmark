# llm-benchmark

LLM基准测试工具，用于测试不同LLM模型的并发性能，包括生成速度、首次token时间等指标。

## 功能特性

- 🔄 **并发测试**：支持多线程并发请求，模拟真实负载场景
- ⚡ **实时监控**：实时显示测试进度、生成速度和完成情况
- 📊 **详细指标**：
  - 平均生成速度 (tokens/s)
  - P50/P99 分位数速度
  - TTFT (首次token时间)
  - 每token延迟 (TGL)
  - 平均输出长度
- 💾 **结果保存**：自动保存所有请求的响应结果
- 🎯 **灵活配置**：支持自定义模型、API端点、并发数等参数

## 安装

### 环境要求

- Python 3.9+
- 依赖管理工具：`uv`

### 安装依赖

```bash
# 安装 uv（如果尚未安装）
pip install uv

# 安装项目依赖
uv sync
```

## 使用方法

### 基本用法

```bash
python benchmark.py --base-url http://localhost:11434/v1 --model qwen2.5:7b --concurrency 2 --requests 4
```

### 参数说明

| 参数名 | 类型 | 默认值 | 描述 |
|--------|------|--------|------|
| `--base-url` | string | `http://localhost:11434/v1` | API端点地址 |
| `--model` | string | `qwen2.5:7b` | 测试模型名称 |
| `--api-key` | string | `xxx` | API密钥 |
| `--concurrency` | int | `2` | 并发请求数 |
| `--requests` | int | `4` | 总请求数 |
| `--output-dir` | string | `outputs` | 响应保存目录 |

### 示例

#### 测试本地Ollama模型

```bash
python benchmark.py --base-url http://localhost:11434/v1 --model llama3:8b --concurrency 5 --requests 20
```

#### 测试远程API

```bash
python benchmark.py --base-url https://api.example.com/v1 --model gpt-4o-mini --api-key sk-xxx --concurrency 10 --requests 50
```

## 测试结果说明

测试完成后，程序会输出详细的统计信息：

```
================================================================================
基准测试完成: qwen2.5:7b
成功请求: 4/4
平均输出长度: 500.0 tokens
平均速度: 38.56 tokens/s (P50: 38.67, P99: 38.89)
平均TTFT: 0.123s
平均每token延迟: 0.026s
所有响应已保存至: /path/to/outputs
================================================================================
```

- **成功请求**：成功完成的请求数/总请求数
- **平均输出长度**：所有成功请求的平均token数
- **平均速度**：每秒生成的平均token数
- **P50/P99**：50%和99%分位数的生成速度
- **TTFT**：从请求发出到收到第一个token的时间
- **每token延迟**：生成每个后续token的平均延迟时间

## 项目结构

```
llm-benchmark/
├── benchmark.py          # 主程序文件
├── pyproject.toml        # 依赖库配置
└── README.md             # 项目说明文档
```

## 工作原理

1. **初始化**：创建异步客户端和信号量控制并发
2. **并发请求**：使用asyncio.gather发起多个并发请求
3. **流式处理**：通过流式API接收模型响应
4. **实时监控**：记录生成速度和进度
5. **结果统计**：计算各项性能指标
6. **结果保存**：将响应内容保存到文件

## 注意事项

1. 确保测试环境的网络稳定，避免网络因素影响测试结果
2. 对于远程API，注意遵守API提供商的速率限制
3. 测试结果仅供参考，实际性能可能因硬件、负载等因素而异
4. 高并发测试可能会对目标服务器造成较大压力，请谨慎使用
