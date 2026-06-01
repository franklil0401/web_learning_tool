# 学生听课实时问答助手

面向课堂听课场景的实时问答助手。监听课堂声音，实时转写老师讲课内容，建立本地检索索引，学生随时提问时根据已记录的课堂片段生成回答。

## 快速开始

### 1. 安装依赖

```bash
pip install soundcard numpy sounddevice websockets flask python-dotenv openai
```

### 2. 配置 API

将 `.env.example` 复制为 `.env`，填入火山方舟 API Key 和豆包流式 ASR 凭证：

```env
ARK_API_KEY=你的火山方舟APIKey
TA_LLM_PROVIDER=doubao
TA_DOUBAO_ANSWER_MODEL=doubao-seed-2-0-lite-260428
TA_TRANSCRIPTION_PROVIDER=doubao_streaming
TA_DOUBAO_ASR_APP_KEY=你的ASR_App_Key
TA_DOUBAO_ASR_ACCESS_KEY=你的ASR_Access_Key
TA_DOUBAO_ASR_RESOURCE_ID=volc.bigasr.sauc.duration
```

### 3. 启动

**Web UI（推荐）：**

```bash
python -m app.web_main
```

浏览器打开 `http://127.0.0.1:5001`，三栏界面：左控制、中问答、右实时转写。

**命令行：**

```bash
python -m app.main
```

## 功能

- 实时监听课堂声音（电脑系统声音 / 麦克风 / 混合模式）
- 豆包大模型流式 ASR 实时转写
- 课堂内容自动入库并建立向量索引
- 学生提问时检索相关片段，调用大模型生成回答
- 回答附带参考课堂片段和相似度分数
- 手动输入课堂内容（演示/调试用）
- 一键清空课程历史，避免课程间内容串扰

## 音频来源

| 模式 | 适用场景 | 说明 |
|------|----------|------|
| 电脑声音 | Bilibili、网课平台 | 默认模式，录电脑系统输出 |
| 麦克风 | 线下课堂 | 录麦克风输入 |
| 混合模式 | 腾讯会议 + 耳机 | 同时录系统声音和麦克风，适合需要同时听到别人和自己的场景 |

## 项目结构

详细说明见 [项目结构.md](项目结构.md)。

```
app/          应用层（音频监听、ASR、LLM、Web 服务）
rag/          RAG 检索增强（切分、入库、检索）
static/       Web 前端（三栏暗色 SPA）
data/         运行时数据（课堂记录 + 向量索引）
models/       本地 Vosk 语音模型（回退用）
```

## 配置参数

常用环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TA_TRANSCRIPTION_PROVIDER` | `auto` | 语音转写：`doubao_streaming` / `doubao` / `openai` / `vosk` |
| `TA_LLM_PROVIDER` | `auto` | 回答模型：`doubao` / `openai` / `deepseek` / `dashscope` |
| `TA_AUDIO_SOURCE` | `system` | 音频来源：`system` / `microphone` / `mixed` |
| `TA_CHUNK_MAX_CHARS` | `800` | 文本切分最大字符数 |
| `TA_CHUNK_OVERLAP` | `120` | 切分重叠字符数 |
| `TA_RETRIEVE_TOP_K` | `5` | 检索召回数量 |
| `TA_MIN_SIMILARITY` | `0.05` | 最低相似度阈值 |

完整配置说明见 `.env.example`。

## 详细文档

- [需求文档.md](需求文档.md) — 项目目的、需求、工作流程、注意事项、交互界面
- [项目结构.md](项目结构.md) — 目录结构与文件职责说明

## 测试

```bash
python -m unittest discover -s tests
```
