"""
FlashHead 推理适配器
封装 pipeline 初始化、图片预处理、音频分块推理、视频编码全流程
pro 模式下通过 dist.broadcast 同步多 rank 执行 pipeline
"""
import os
import uuid
import threading
import subprocess
import numpy as np
import cv2
import librosa
from collections import deque
from typing import Optional, List

from config import get_config
from loguru import logger

_pipeline = None
_infer_params = None
_pipeline_lock = threading.Lock()
_is_pro = False


def init_pipeline():
    """启动时初始化 FlashHead pipeline（单例）"""
    global _pipeline, _infer_params, _is_pro
    with _pipeline_lock:
        if _pipeline is not None:
            return
        cfg = get_config().flashhead
        nproc = len(cfg.pro_device_ids.split(",")) if cfg.mode == "pro" else 1
        _is_pro = cfg.mode == "pro" and nproc > 1

        if cfg.mode == "pro":
            world_size = nproc
            model_type = "pro"
        else:
            world_size = 1
            model_type = "lite"

        # 根据配置决定是否禁用 torch.compile（在导入 flash_head 之前设置）
        if not cfg.torch_compile:
            os.environ["TORCHDYNAMO_DISABLE"] = "1"

        from flash_head.inference import get_pipeline, get_infer_params
        _pipeline = get_pipeline(
            world_size=world_size,
            ckpt_dir=cfg.ckpt_dir,
            model_type=model_type,
            wav2vec_dir=cfg.wav2vec_dir,
        )
        _infer_params = get_infer_params()
        logger.info(f"Pipeline 初始化完成 [mode={cfg.mode}]: {_infer_params['height']}x{_infer_params['width']}")


# ==================== Pro 模式多 rank 同步 ====================
# 信号约定（broadcast 一个 int tensor）:
#   0 = 退出
#   1 = prepare_params (get_base_data)
#   2 = run_pipeline (get_audio_embedding + generate)

def run_worker_loop():
    """rank != 0 的 worker 循环，等待 rank 0 的信号同步执行 pipeline"""
    import torch
    import torch.distributed as dist
    from flash_head.inference import get_base_data, get_audio_embedding, run_pipeline

    device = _pipeline.device
    logger.info(f"[Worker] rank={dist.get_rank()} 进入 worker 循环")

    while True:
        # 等待信号
        signal = torch.zeros(1, dtype=torch.long, device=device)
        dist.broadcast(signal, src=0)
        cmd = signal.item()

        if cmd == 0:
            logger.info("[Worker] 收到退出信号")
            break
        elif cmd == 1:
            # prepare_params: 接收 crop_path 字符串长度 + 内容
            str_len = torch.zeros(1, dtype=torch.long, device=device)
            dist.broadcast(str_len, src=0)
            str_tensor = torch.zeros(str_len.item(), dtype=torch.uint8, device=device)
            dist.broadcast(str_tensor, src=0)
            crop_path = bytes(str_tensor.cpu().tolist()).decode("utf-8")
            get_base_data(_pipeline, crop_path, base_seed=42, use_face_crop=False)
        elif cmd == 2:
            # run_pipeline: 接收 audio_embedding shape + data
            shape_tensor = torch.zeros(4, dtype=torch.long, device=device)
            dist.broadcast(shape_tensor, src=0)
            shape = tuple(shape_tensor.cpu().tolist())
            audio_embedding = torch.zeros(shape, dtype=torch.float32, device=device)
            dist.broadcast(audio_embedding, src=0)
            # 同步执行推理（内部有 NCCL 通信）
            run_pipeline(_pipeline, audio_embedding)


def _broadcast_signal(cmd: int):
    """rank 0 发送信号"""
    import torch
    import torch.distributed as dist
    signal = torch.tensor([cmd], dtype=torch.long, device=_pipeline.device)
    dist.broadcast(signal, src=0)


def _broadcast_prepare_params(crop_path: str):
    """rank 0 广播 prepare_params 参数并执行"""
    import torch
    import torch.distributed as dist
    from flash_head.inference import get_base_data

    _broadcast_signal(1)
    path_bytes = crop_path.encode("utf-8")
    str_len = torch.tensor([len(path_bytes)], dtype=torch.long, device=_pipeline.device)
    dist.broadcast(str_len, src=0)
    str_tensor = torch.tensor(list(path_bytes), dtype=torch.uint8, device=_pipeline.device)
    dist.broadcast(str_tensor, src=0)
    get_base_data(_pipeline, crop_path, base_seed=42, use_face_crop=False)


def _broadcast_run_pipeline(audio_embedding):
    """rank 0 广播 audio_embedding 并同步执行推理"""
    import torch
    import torch.distributed as dist
    from flash_head.inference import run_pipeline

    _broadcast_signal(2)
    shape_tensor = torch.tensor(list(audio_embedding.shape), dtype=torch.long, device=_pipeline.device)
    # 补齐到 4 维
    while shape_tensor.shape[0] < 4:
        shape_tensor = torch.cat([shape_tensor, torch.tensor([1], dtype=torch.long, device=_pipeline.device)])
    dist.broadcast(shape_tensor, src=0)
    ae = audio_embedding.to(torch.float32).contiguous().to(_pipeline.device)
    dist.broadcast(ae, src=0)
    return run_pipeline(_pipeline, audio_embedding)


# ==================== 图片裁剪 ====================

def _crop_image(image_path: str, crop_region: Optional[List[int]] = None) -> str:
    cfg = get_config()
    face_ratio = cfg.flashhead.face_ratio

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"无法读取图片: {image_path}")
    img_h, img_w = img_bgr.shape[:2]

    if crop_region and len(crop_region) == 4:
        x1, y1, x2, y2 = crop_region
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
    else:
        from flash_head.utils.cpu_face_handler import CPUFaceHandler
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        detector = CPUFaceHandler()
        boxes, _ = detector(img_rgb)
        if not boxes or len(boxes) == 0:
            raise ValueError("未检测到人脸")
        fx1, fy1, fx2, fy2 = boxes[0][0]*img_w, boxes[0][1]*img_h, boxes[0][2]*img_w, boxes[0][3]*img_h
        cx, cy = (fx1+fx2)/2, (fy1+fy2)/2
        fw = fx2 - fx1
        nw = fw * face_ratio
        x1 = int(max(0, cx - nw*0.5))
        x2 = int(min(img_w, cx + nw*0.5))
        y1 = int(max(0, cy - nw*0.55))
        y2 = int(min(img_h, cy + nw*0.45))

    crop = img_bgr[y1:y2, x1:x2]
    tmp_dir = os.path.join(cfg.cache_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{uuid.uuid4()}.png")
    cv2.imwrite(tmp_path, crop)
    logger.info(f"裁剪区域: [{x1},{y1},{x2},{y2}] = {x2-x1}x{y2-y1}")
    return tmp_path


# ==================== 合成主流程 ====================

def synthesize(image_path: str, audio_path: str, output_path: str,
               crop_region: Optional[List[int]] = None,
               progress_callback=None) -> str:
    global _pipeline, _infer_params
    if _pipeline is None:
        raise RuntimeError("Pipeline 未初始化")

    from flash_head.inference import get_base_data, get_audio_embedding, run_pipeline
    cfg = get_config()

    # 1. 裁剪图片
    crop_path = _crop_image(image_path, crop_region)

    # 2. 设置条件图（pro 模式需要广播给其他 rank）
    if _is_pro:
        _broadcast_prepare_params(crop_path)
    else:
        get_base_data(_pipeline, crop_path, base_seed=42, use_face_crop=False)

    # 3. 音频参数
    tgt_fps = _infer_params['tgt_fps']
    frame_num = _infer_params['frame_num']
    sample_rate = _infer_params['sample_rate']
    motion_frames_num = _infer_params['motion_frames_num']
    gen_frame_num = frame_num - motion_frames_num
    height = _infer_params['height']
    width = _infer_params['width']

    # 4. 加载音频并分块
    speech_array, _ = librosa.load(audio_path, sr=sample_rate)
    cached_audio_duration = _infer_params['cached_audio_duration']
    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num
    human_speech_slice_len = gen_frame_num * sample_rate // tgt_fps
    total_samples = (len(speech_array) // human_speech_slice_len) * human_speech_slice_len
    speech_slices = speech_array[:total_samples].reshape(-1, human_speech_slice_len)
    total_chunks = len(speech_slices)

    # 5. FFmpeg 编码进程
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        cfg.ffmpeg_path, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(tgt_fps),
        '-i', '-', '-i', audio_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-shortest', '-bf', '0', '-v', 'warning',
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

    # 6. 逐 chunk 推理
    logger.info(f"开始推理: {total_chunks} chunks, {height}x{width}")
    try:
        for chunk_idx in range(total_chunks):
            audio_dq.extend(speech_slices[chunk_idx].tolist())
            audio_array = np.array(audio_dq)
            audio_embedding = get_audio_embedding(_pipeline, audio_array, audio_start_idx, audio_end_idx)

            # pro 模式广播 audio_embedding 同步推理，lite 模式直接推理
            if _is_pro:
                video = _broadcast_run_pipeline(audio_embedding)
            else:
                video = run_pipeline(_pipeline, audio_embedding)

            gen_frames = video.cpu().numpy().astype(np.uint8)
            for i in range(gen_frames.shape[0]):
                proc.stdin.write(gen_frames[i].tobytes())
            if progress_callback:
                progress_callback(chunk_idx + 1, total_chunks)
            logger.info(f"chunk {chunk_idx+1}/{total_chunks}")
    finally:
        proc.stdin.close()
        proc.wait()

    # 7. 清理临时裁剪图
    try:
        os.remove(crop_path)
    except OSError:
        pass

    logger.info(f"合成完成: {output_path}")
    return output_path
