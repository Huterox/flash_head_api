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
from typing import Optional, List, Tuple

from config import get_config
from loguru import logger


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """
    将十六进制颜色转换为 BGR 元组（OpenCV 格式）

    Args:
        hex_color: 十六进制颜色字符串，如 "#00FF00" 或 "00FF00"

    Returns:
        BGR 元组，如 (0, 255, 0)
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV 使用 BGR 顺序

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

def _crop_image(image_path: str, crop_region: Optional[List[int]] = None) -> Tuple[str, str, List[int]]:
    """
    裁剪图片为正方形区域

    Args:
        image_path: 原图路径
        crop_region: 可选的裁剪区域 [x1, y1, x2, y2]

    Returns:
        Tuple[裁剪后路径, 原图路径, 裁剪坐标[x1, y1, x2, y2]]
    """
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
    return tmp_path, image_path, [x1, y1, x2, y2]


# ==================== 背景移除 ====================

def _remove_background_from_video(
    video_path: str,
    audio_path: str,
    bg_color: Tuple[int, int, int] = (0, 255, 0),
    progress_callback=None
) -> str:
    """
    对已生成的视频进行背景移除处理（替换为绿幕）

    Args:
        video_path: 视频路径（会被替换）
        audio_path: 音频路径
        bg_color: 背景颜色 BGR 元组，默认绿色 (0, 255, 0)
        progress_callback: 进度回调函数

    Returns:
        处理后的视频路径（与输入相同）
    """
    cfg = get_config()
    rvm_cfg = cfg.rvm

    logger.info(f"[背景移除] 配置检查: enabled={rvm_cfg.enabled}, variant={rvm_cfg.variant}")
    logger.info(f"[背景移除] 模型路径: {rvm_cfg.checkpoint}")
    logger.info(f"[背景移除] 背景颜色: {bg_color} (BGR)")

    if not rvm_cfg.enabled:
        logger.warning("❌ RVM 未启用（config.yml 中 rvm.enabled=false），跳过背景移除")
        return video_path

    logger.info(f"[背景移除] 开始初始化 RVM 处理器...")
    # 初始化 RVM 处理器
    from cores.rvm_processor import RVMProcessor
    try:
        processor = RVMProcessor(
            checkpoint_path=rvm_cfg.checkpoint,
            variant=rvm_cfg.variant,
            device=rvm_cfg.device
        )
        logger.info(f"[背景移除] ✅ RVM 处理器初始化成功")
    except Exception as e:
        logger.error(f"[背景移除] ❌ RVM 处理器初始化失败: {e}")
        raise

    # 打开原视频
    logger.info(f"[背景移除] 打开视频文件: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"[背景移除] 视频信息: {frame_count} 帧, {width}x{height}, {fps} fps")

    # 使用 FFmpeg 管道直接编码，避免 VideoWriter 二次压缩
    final_output = video_path.replace('.mp4', '_bg_removed_final.mp4')

    # FFmpeg 命令：从管道读取原始帧，高质量编码
    ffmpeg_cmd = [
        cfg.ffmpeg_path, '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',  # 从 stdin 读取
        '-i', audio_path,
        '-c:v', 'libx264',
        '-preset', 'slow',  # 更慢但质量更好
        '-crf', '18',  # 视觉无损质量（原来 23）
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',  # 提高音频码率
        '-shortest',
        '-v', 'warning',
        final_output
    ]

    logger.info(f"[背景移除] 启动 FFmpeg 管道编码（CRF=18, preset=slow）...")
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    logger.info(f"[背景移除] 开始逐帧处理...")

    # 逐帧处理并直接写入 FFmpeg 管道
    frame_idx = 0
    downsample_ratio = rvm_cfg.downsample_ratio

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # RVM 处理
            processed_frame = processor.process_single_frame(
                frame,
                bg_color=bg_color,
                downsample_ratio=downsample_ratio
            )

            # 直接写入 FFmpeg 管道
            proc.stdin.write(processed_frame.tobytes())
            frame_idx += 1

            # 进度回调
            if progress_callback and frame_idx % 5 == 0:
                progress_callback(frame_idx, frame_count, "bg_remove", "背景移除")

    finally:
        cap.release()
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        logger.error(f"[背景移除] ❌ FFmpeg 编码失败: {stderr}")
        raise RuntimeError(f"FFmpeg 编码失败: {stderr}")

    logger.info(f"[背景移除] ✅ 逐帧处理完成，共处理 {frame_idx} 帧")
    logger.info(f"[背景移除] ✅ FFmpeg 高质量编码完成")

    # 替换原视频
    logger.info(f"[背景移除] 替换原视频: {video_path}")
    try:
        os.remove(video_path)
        os.rename(final_output, video_path)
        logger.info(f"[背景移除] ✅ 视频替换成功")
    except OSError as e:
        logger.error(f"[背景移除] ❌ 替换视频失败: {e}")
        raise

    logger.info(f"[背景移除] ========== 背景移除完成 ==========")
    logger.info(f"[背景移除] 最终输出: {video_path}")
    return video_path


# ==================== 贴回原图 ====================

def _paste_back_video(
    generated_video_path: str,
    original_image_path: str,
    crop_coords: List[int],
    audio_path: str,
    output_path: str,
    progress_callback=None
) -> str:
    """
    将生成的 512×512 大头视频逐帧贴回原图

    Args:
        generated_video_path: 生成的 512×512 视频路径
        original_image_path: 原图路径
        crop_coords: 裁剪坐标 [x1, y1, x2, y2]（用户选择或自动检测的区域）
        audio_path: 音频路径
        output_path: 最终输出路径

    Returns:
        输出视频路径

    注意：
        需要反向计算 resize_and_centercrop 的实际使用区域，因为 FlashHead 会对裁剪图进行
        缩放+中心裁剪到 512×512，生成的视频对应的是裁剪后的中心区域
    """
    cfg = get_config()
    x1, y1, x2, y2 = crop_coords
    crop_w, crop_h = x2 - x1, y2 - y1

    # 读取原图作为静止背景
    bg_img = cv2.imread(original_image_path)
    if bg_img is None:
        raise ValueError(f"无法读取原图: {original_image_path}")
    orig_h, orig_w = bg_img.shape[:2]

    # 计算 resize_and_centercrop 的实际使用区域
    # FlashHead 会将裁剪图缩放+中心裁剪到 512×512
    target_size = 512
    scale_h = target_size / crop_h
    scale_w = target_size / crop_w
    scale = max(scale_h, scale_w)  # 保证至少一边填满

    # 缩放后的尺寸
    scaled_h = int(crop_h * scale)
    scaled_w = int(crop_w * scale)

    # 中心裁剪的偏移量（在缩放后的图片上）
    crop_offset_y = (scaled_h - target_size) // 2
    crop_offset_x = (scaled_w - target_size) // 2

    # 反向映射到原始裁剪区域的坐标（在原图坐标系中）
    # 512×512 视频对应的是裁剪区域中的一个子区域
    actual_x1 = x1 + int(crop_offset_x / scale)
    actual_y1 = y1 + int(crop_offset_y / scale)
    actual_x2 = actual_x1 + int(target_size / scale)
    actual_y2 = actual_y1 + int(target_size / scale)

    # 确保不超出原图边界
    actual_x1 = max(0, actual_x1)
    actual_y1 = max(0, actual_y1)
    actual_x2 = min(orig_w, actual_x2)
    actual_y2 = min(orig_h, actual_y2)

    actual_w = actual_x2 - actual_x1
    actual_h = actual_y2 - actual_y1

    # 打开生成的视频
    cap = cv2.VideoCapture(generated_video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {generated_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建临时输出（无音频）
    temp_output = output_path.replace('.mp4', '_composite_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (orig_w, orig_h))

    logger.info(f"开始贴回原图：原图尺寸 {orig_w}x{orig_h}")
    logger.info(f"  用户裁剪区域: [{x1},{y1},{x2},{y2}] ({crop_w}x{crop_h})")
    logger.info(f"  实际贴回区域: [{actual_x1},{actual_y1},{actual_x2},{actual_y2}] ({actual_w}x{actual_h})")

    # 逐帧处理
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 将 512×512 生成帧缩放到实际贴回区域大小
        resized_frame = cv2.resize(frame, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)

        # 复制背景图
        composite = bg_img.copy()

        # 贴回到实际区域
        composite[actual_y1:actual_y2, actual_x1:actual_x2] = resized_frame

        out.write(composite)
        frame_idx += 1

        # 进度回调
        if progress_callback and frame_idx % 5 == 0:
            progress_callback(frame_idx, frame_count, "paste_back", "贴回原图")

    cap.release()
    out.release()

    logger.info(f"贴回完成 {frame_idx} 帧，开始添加音频")

    # 使用 FFmpeg 添加音频并重新编码
    final_cmd = [
        cfg.ffmpeg_path, '-y',
        '-i', temp_output,
        '-i', audio_path,
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-shortest',
        '-v', 'warning',
        output_path
    ]

    try:
        subprocess.run(final_cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg 编码失败: {e.stderr.decode()}")
        raise

    # 删除临时文件
    try:
        os.remove(temp_output)
    except OSError:
        pass

    logger.info(f"贴回原图完成，输出视频：{output_path}")
    return output_path


# ==================== 合成主流程 ====================

def synthesize(image_path: str, audio_path: str, output_path: str,
               crop_region: Optional[List[int]] = None,
               restore_to_original: bool = False,
               bg_remove: bool = False,
               bg_color: str = "#00FF00",
               progress_callback=None) -> str:
    global _pipeline, _infer_params
    if _pipeline is None:
        raise RuntimeError("Pipeline 未初始化")

    from flash_head.inference import get_base_data, get_audio_embedding, run_pipeline
    cfg = get_config()

    logger.info(f"========== 合成任务开始 ==========")
    logger.info(f"参数: restore_to_original={restore_to_original}, bg_remove={bg_remove}, bg_color={bg_color}")
    logger.info(f"输出路径: {output_path}")

    # 进度分配：合成占 80%，背景移除占 20%（如果启用）
    # 如果没有背景移除，合成占 100%
    synthesis_weight = 0.8 if bg_remove else 1.0
    bg_remove_weight = 0.2 if bg_remove else 0.0
    logger.info(f"进度分配: 合成={synthesis_weight*100}%, 背景移除={bg_remove_weight*100}%")

    def report_progress(current, total, stage, stage_name):
        """统一的进度报告函数，自动计算总体进度百分比"""
        if not progress_callback:
            return

        stage_percent = (current / total * 100) if total > 0 else 0

        # 根据阶段计算总体进度
        if stage == "inference" or stage == "paste_back":
            # 合成阶段（包括推理和贴回原图）
            overall_percent = stage_percent * synthesis_weight
        elif stage == "bg_remove":
            # 背景移除阶段
            overall_percent = synthesis_weight * 100 + stage_percent * bg_remove_weight
        else:
            overall_percent = stage_percent

        progress_callback(
            current, total,
            stage=stage,
            stage_name=stage_name,
            percent=round(overall_percent, 1)
        )

    # 1. 裁剪图片（返回三个值：裁剪后路径、原图路径、裁剪坐标）
    crop_path, orig_path, crop_coords = _crop_image(image_path, crop_region)

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

    # 5. 确定输出路径（如果需要贴回原图，先输出到临时路径）
    if restore_to_original:
        temp_video_path = output_path.replace('.mp4', '_head_only.mp4')
    else:
        temp_video_path = output_path

    # 6. FFmpeg 编码进程
    os.makedirs(os.path.dirname(temp_video_path), exist_ok=True)
    cmd = [
        cfg.ffmpeg_path, '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(tgt_fps),
        '-i', '-', '-i', audio_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-shortest', '-bf', '0', '-v', 'warning',
        temp_video_path,
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
            report_progress(chunk_idx + 1, total_chunks, "inference", "视频推理")
            logger.info(f"chunk {chunk_idx+1}/{total_chunks}")
    finally:
        proc.stdin.close()
        proc.wait()

    # 7. 如果启用贴回原图，将大头视频贴回到原图
    if restore_to_original:
        logger.info("========== 阶段 7: 贴回原图 ==========")
        logger.info(f"贴回原图已启用，开始处理")
        _paste_back_video(temp_video_path, orig_path, crop_coords, audio_path, output_path, report_progress)
        # 删除临时大头视频
        try:
            os.remove(temp_video_path)
        except OSError:
            pass
        logger.info("贴回原图完成")
    else:
        logger.info("========== 阶段 7: 贴回原图 ==========")
        logger.info("贴回原图未启用，跳过")

    # 8. 如果启用背景移除，处理最终视频
    logger.info(f"========== 阶段 8: 背景移除 ==========")
    logger.info(f"bg_remove 参数值: {bg_remove} (类型: {type(bg_remove).__name__})")

    if bg_remove:
        logger.info(f"✅ 背景移除已启用，开始处理视频: {output_path}")
        logger.info(f"背景颜色: {bg_color} (十六进制)")
        # 转换十六进制颜色为 BGR 元组
        bg_color_bgr = hex_to_bgr(bg_color)
        logger.info(f"转换后的 BGR 颜色: {bg_color_bgr}")
        try:
            _remove_background_from_video(output_path, audio_path, bg_color_bgr, report_progress)
            logger.info("✅ 背景移除处理完成")
        except Exception as e:
            logger.error(f"❌ 背景移除失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    else:
        logger.info("❌ 背景移除未启用，跳过此阶段")

    # 9. 清理临时裁剪图
    try:
        os.remove(crop_path)
    except OSError:
        pass

    logger.info(f"合成完成: {output_path}")
    return output_path
