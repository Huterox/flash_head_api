# Once FlashHead API

基于 [SoulX-FlashHead](https://github.com/SoulX-Org/FlashHead) 的说话头部视频生成服务。上传一张人像图片和一段音频，自动裁剪头肩区域并生成 512×512 的说话头部视频。服务层对齐 Once Edge 算法服务标准架构。

## 功能特性

- **双模式推理**：lite（单卡蒸馏模型，4 步推理）/ pro（多卡序列并行，20 步推理）
- **自动人脸裁剪**：基于 MediaPipe 的 CPU 人脸检测，自动定位并裁剪头肩正方形区域；也支持前端手动指定裁剪区域
- **异步任务队列**：Redis 队列 + 单线程池串行调度，提交即返回 task_id，轮询获取进度
- **管理面板**：内置 Vue 3 单页面，可视化管理任务与文件
- **网关对接**：支持向 Once Edge 网关自动注册节点 + 心跳保活

## 项目结构

```
once_flash_head_api/
├── config/                         # 配置层
│   ├── config.yml                  #   唯一配置文件
│   ├── schema.py                   #   Pydantic 配置模型 (AppConfig)
│   └── loader.py                   #   get_config() 单例加载器
├── state/                          # 状态层（数据库 / 队列 / 调度）
│   ├── db_engine.py                #   SQLAlchemy 引擎 (PostgreSQL)
│   ├── db_models.py                #   ORM 模型: Task, UploadedFile
│   ├── db_operations.py            #   数据库 CRUD
│   ├── redis_client.py             #   Redis 客户端（任务队列 + 进度）
│   └── scheduler.py                #   单线程池任务调度器
├── schema/                         # 数据模型
│   ├── enums.py                    #   TaskStatus 枚举
│   └── request_entities.py         #   请求体定义 (SynthesizeRequest 等)
├── service/                        # 服务层
│   ├── app.py                      #   FastAPI 应用 + 生命周期管理
│   ├── dependencies.py             #   API Key 认证依赖
│   └── routes/                     #   路由
│       ├── task_api.py             #     任务：合成 / 查询 / 下载
│       ├── file_api.py             #     文件：上传
│       └── system_api.py           #     系统：健康检查、调度器状态
├── cores/                          # 核心适配
│   └── pipeline_adapter.py         #   FlashHead 推理适配器（初始化 / 预处理 / 推理 / 编码）
├── utils/                          # 工具
│   ├── result.py                   #   统一响应格式 R
│   └── file_manager.py             #   文件上传管理
├── flash_head/                     # 核心算法（SoulX-FlashHead 源码，不修改）
│   ├── inference.py                #   推理入口
│   ├── audio_analysis/             #   wav2vec2 音频特征提取
│   ├── ltx_video/                  #   LTX-Video VAE & Transformer
│   ├── wan/                        #   WAN VAE 模块
│   ├── src/                        #   FlashHead 模型 + 分布式
│   ├── utils/                      #   人脸裁剪、工具函数
│   └── configs/                    #   推理参数 (infer_params.yaml)
├── checkpoint/                     # 模型权重（需自行下载）
│   ├── SoulX-FlashHead-1_3B/      #   FlashHead 1.3B 模型
│   └── wav2vec2-base-960h/         #   wav2vec2 音频编码器
├── libs/                           # 外部工具
│   └── ffmpeg.exe                  #   FFmpeg 可执行文件
├── templates/
│   └── index.html                  # Vue 3 管理面板
├── base.py                         # 路径常量
├── start_api.py                    # 启动入口
├── requirements.txt                # Python 依赖
└── cache/                          # 运行时缓存（自动创建）
    ├── uploads/                    #   上传文件暂存
    └── out/                        #   输出视频
```

## 环境要求

| 依赖 | 版本 |
|------|------|
| Python | 3.10 |
| CUDA | 12.8+ |
| PyTorch | 2.7.1 |
| PostgreSQL | 12+ |
| Redis | 5+ |
| FFmpeg | 4.4+ |

> GPU 显存需求：lite 模式约 8 GB，pro 模式需要两张 GPU。

## 快速开始

### 1. 创建环境

```bash
conda create -n flashhead python=3.10
conda activate flashhead
```

### 2. 安装依赖

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install ninja && pip install flash_attn==2.8.0.post2 --no-build-isolation
```

### 3. 准备模型权重

将模型文件放置到 `checkpoint/` 目录下：

```
checkpoint/
├── SoulX-FlashHead-1_3B/    # FlashHead 模型权重
└── wav2vec2-base-960h/       # wav2vec2 音频编码器
```

### 4. 修改配置

编辑 `config/config.yml`，按实际环境修改以下字段：

```yaml
database:
  host: "your-pg-host"
  password: "your-pg-password"

redis:
  host: "your-redis-host"
  password: "your-redis-password"

flashhead:
  mode: lite                  # lite 或 pro
  ckpt_dir: "/absolute/path/to/checkpoint/SoulX-FlashHead-1_3B"
  wav2vec_dir: "/absolute/path/to/checkpoint/wav2vec2-base-960h"

server:
  port: 8100
  api_key: "your-api-key"

ffmpeg_path: "/absolute/path/to/ffmpeg"
cache_dir: "/absolute/path/to/cache"
out_dir: "/absolute/path/to/cache/out"
```

> 所有路径必须使用**绝对路径**。

### 5. 启动服务

```bash
python start_api.py
```

服务启动后：
- API 文档：`http://localhost:8100/docs`
- 管理面板：`http://localhost:8100/`

## API 接口

所有业务接口需携带 Header `X-API-Key`。

### 上传文件

```
POST /api/files/upload
Content-Type: multipart/form-data
```

支持格式：`.png` `.jpg` `.jpeg` `.wav` `.mp3` `.m4a`

### 提交合成任务

```
POST /api/tasks/synthesize
Content-Type: application/json

{
  "image_file_id": "上传返回的 file_id",
  "audio_file_id": "上传返回的 file_id",
  "crop_region": [x, y, w, h]       // 可选，不传则自动人脸检测裁剪
}
```

### 查询任务状态

```
GET /api/tasks/{task_id}
```

### 下载结果视频

```
GET /api/tasks/{task_id}/download
```

## 推理模式

| 模式 | 模型 | 推理步数 | GPU 需求 | 启动方式 |
|------|------|---------|---------|---------|
| lite | 蒸馏模型 | 4 steps | 单卡 | `python start_api.py` |
| pro | 预训练模型 | 20 steps | 双卡 | 自动 torchrun 拉起 |

在 `config.yml` 中设置 `flashhead.mode` 切换模式。

## 鸣谢

- [SoulX-FlashHead](https://github.com/SoulX-Org/FlashHead) — 核心说话头部生成算法
- [LTX-Video](https://github.com/Lightricks/LTX-Video) — 视频 VAE & Transformer 架构
- [wav2vec 2.0](https://github.com/facebookresearch/fairseq) — 音频特征提取
- [xfuser](https://github.com/xdit-project/xDiT) — 序列并行推理加速
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) — CPU 人脸检测
- [Once Edge](https://github.com/Starter-Kit-Org) — 算法服务标准架构规范

## 许可证

本项目服务层代码遵循 MIT 许可证。`flash_head/` 目录下的核心算法代码请遵循 SoulX-FlashHead 原始许可协议。
