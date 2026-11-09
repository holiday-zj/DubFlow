# DubFlow — 视频出海一站式 AI 本地化工具

面向内容创作者/跨境卖家：上传视频 + 选择目标语言，自动完成
**音轨提取 → Whisper 语音识别 → 机器翻译 → 神经配音 → (口型同步) → 合成输出**。

对外提供三种接入方式：
1. **Web UI**（浏览器上传视频、看进度、下载成片）
2. **REST API**（`POST /api/jobs` 异步任务接口，供业务系统集成）
3. **MCP 服务**（stdio，供 Claude Desktop / Hermes Agent 等 AI 客户端直接调用）

## 流水线架构

每个环节是独立组件，可单独替换（参照 model-compose 的设计思路）：

| 阶段 | 实现 | 说明 |
|---|---|---|
| 1. 音频提取 | FFmpeg | 抽成 16kHz 单声道 WAV（语音模型标准输入） |
| 2. STT | faster-whisper（CTranslate2） | tiny/base/small/medium/large-v3 可选，默认 small；CPU int8 可跑；带 VAD |
| 3. 翻译 | MarianMT（默认）/ NLLB-200 | 本地免费；商业 API（DeepL 等）预留接口 |
| 4. TTS | edge-tts（默认，免费神经声线） | 按字幕时间轴逐段合成并 atempo 变速对齐槽位；GPT-SoVITS/CosyVoice 声音克隆通过 HTTP 服务对接（`DUBFLOW_TTS_URL`） |
| 4a. BGM 保留 | demucs htdemucs（可选，默认开） | 人声/背景分离，只替换人声；背景乐在说话时 sidechain 自动闪避（ducking），可调背景音量 0-200% |
| 5. 口型同步 | 外部 HTTP 服务（可选） | `DUBFLOW_LIPSYNC_URL` 指向 LatentSync 1.5（Apache-2.0 可商用）或 Wav2Lip（仅内部测试，非商业许可）；本机无 GPU 时自动跳过 |
| 6. 合成 | FFmpeg | `-map 0:v -map 1:a -map 2:s` 嵌入字幕轨（mov_text），或烧录字幕（libx264 重编码） |

任务队列：单 worker 串行执行（CPU 机器），任务持久化到 `data/jobs/<id>/job.json`，重启后恢复历史。

## 目录

```
dubflow/
├── app/
│   ├── config.py      # 全部配置（模型、声线、环境变量）
│   ├── pipeline.py    # 各阶段实现（可独立调用）
│   ├── worker.py      # 后台任务执行器（5阶段状态机）
│   ├── jobs.py        # 任务存储 + 队列
│   └── main.py        # FastAPI（REST + Web UI + 静态产物）
├── mcp_server.py      # MCP stdio 服务（6个工具）
├── web/index.html     # 单页前端（上传/进度/成片播放/字幕对照表）
├── bin/               # ffmpeg/ffprobe 静态二进制
├── scripts/test_mcp.py# MCP 端到端测试
└── data/jobs/<id>/    # input.mp4, audio_16k.wav, source.srt, translated.srt,
                       # dubbed_audio.wav, output.mp4, job.json
```

## 快速开始

```bash
cd ~/dubflow
source .venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8090
# Web UI:  http://<host>:8090/
# API docs: http://<host>:8090/docs
```

首次运行会自动下载 Whisper small（~480MB）与 Marian 模型（~300MB/语向）到 `models/`。

systemd 部署：`systemctl --user enable --now dubflow`（单元见 `scripts/dubflow.service`）。

## REST API

### POST /api/jobs （multipart）
| 字段 | 必填 | 说明 |
|---|---|---|
| video | ✔ | 视频文件（≤500MB，mp4/mov/mkv…） |
| tgt_lang | ✔ | 目标语言：en/zh/ja/ko/es/fr/de/ru/pt/it/ar/id/vi/th/tr/hi |
| src_lang | | 源语言，留空自动检测 |
| whisper_model | | tiny/base/small(默认)/medium/large-v3 |
| translate_engine | | marian(默认)/nllb |
| gender | | f(默认)/m（edge-tts 声线性别） |
| voice | | 直接指定 edge-tts 声线名，覆盖 gender |
| burn_subs | | true=字幕烧录进画面 |
| lipsync | | true=调用外部口型同步服务（未配置则跳过） |
| keep_bgm | | true=demucs 人声分离，保留原背景音乐/音效并混回（CPU 上约 1.3x 视频时长） |
| bgm_volume | | 背景音量倍数 0.0-2.0，默认 1.0 |

返回 `{"job_id": "...", "status_url": "/api/jobs/<id>", "result_url": "/api/jobs/<id>/result"}`

### GET /api/jobs/<id> — 状态/进度/日志/分段字幕对照
### GET /api/jobs/<id>/result — 完成后返回产物链接 + segments（未完成返回 202）
### GET /api/jobs — 任务列表
### GET /api/meta — 支持的语言/模型/声线
### GET /jobs/<id>/<file> — 下载产物（output.mp4、translated.srt、dubbed_audio.wav…）

示例：
```bash
curl -X POST http://localhost:8090/api/jobs \
  -F "video=@my_video.mp4" -F "tgt_lang=en" -F "whisper_model=small"
curl http://localhost:8090/api/jobs/<job_id>/result
```

鉴权（生产环境务必开启）：设 `DUBFLOW_API_KEY=*** `X-API-Key` 或 `Authorization: Bearer` 头。

## MCP 服务

启动：`python -m mcp_server`（stdio）。工具：

| 工具 | 功能 |
|---|---|
| dubflow_capabilities | 列出支持的语言/模型/引擎 |
| dubflow_submit | 提交本地视频路径 → job_id |
| dubflow_status | 查询进度/日志/产物 |
| dubflow_wait | 阻塞至完成或超时 |
| dubflow_transcribe | 仅 STT：视频 → 带时间戳分段 + SRT 文件 |
| dubflow_translate | 仅翻译：文本列表 → 译文 |

客户端配置（Claude Desktop / Hermes）：
```json
{"mcpServers": {"dubflow": {
  "command": "/home/ec2-user/dubflow/.venv/bin/python",
  "args": ["-m", "mcp_server"],
  "cwd": "/home/ec2-user/dubflow"}}}
```

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| DUBFLOW_API_KEY | 空=免鉴权 | REST API 密钥 |
| DUBFLOW_WHISPER_MODEL | small | 默认 Whisper 档位 |
| DUBFLOW_LIPSYNC_URL | 空=跳过 | 口型同步 HTTP 服务（LatentSync/Wav2Lip，如 psyb0t/flickies API） |
| DUBFLOW_TTS_URL | 空 | GPT-SoVITS/CosyVoice 声音克隆服务（预留） |
| DEEPL_API_KEY | 空 | DeepL 商业翻译（预留） |
| DUBFLOW_MAX_UPLOAD_MB | 500 | 上传上限 |

## 商业化注意事项

- **口型同步许可**：Wav2Lip 基于 LRS2 训练，**非商业许可**，只能内部测试；商用必须用 LatentSync 1.5（Apache-2.0）或取得授权。本服务通过 `DUBFLOW_LIPSYNC_URL` 解耦，切换引擎不影响其他环节。
- **声音克隆**：GPT-SoVITS/CosyVoice 均要求取得被克隆者授权；商用前需加用户协议确认。
- **edge-tts**：免费但属微软在线服务，无正式商用授权条款；正式产品建议替换为 GPT-SoVITS（6GB 显存）/CosyVoice（4GB）自托管，或商业 TTS API。
- 本仓库所有本地模型推理均为 CPU 路径；GPU 机器上把 `DUBFLOW_DEVICE=cuda`、`DUBFLOW_COMPUTE=float16` 即可加速。

## 测试

```bash
source .venv/bin/activate
python scripts/test_mcp.py          # MCP 全链路（capabilities/translate/submit/wait）
# REST 全链路见 README 顶部 curl 示例；测试视频生成脚本 scripts/make_sample.py
```
