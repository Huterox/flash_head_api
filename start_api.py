"""FlashHead API 启动入口"""
import os
import sys
import subprocess
from pathlib import Path

# 项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from config import get_config

cfg = get_config()


def _is_torchrun():
    """判断当前是否由 torchrun 拉起"""
    return "LOCAL_RANK" in os.environ


if __name__ == "__main__":
    nproc = len(cfg.flashhead.pro_device_ids.split(",")) if cfg.flashhead.mode == "pro" else 1

    if cfg.flashhead.mode == "pro" and nproc > 1 and not _is_torchrun():
        # pro 多卡模式首次启动：用 torchrun 重新拉起自身
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.flashhead.pro_device_ids
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "86400"
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            "--master_port=29500",
            __file__,
        ]
        print(f"[PRO] 启动 {nproc} 卡序列并行: CUDA_VISIBLE_DEVICES={cfg.flashhead.pro_device_ids}")
        subprocess.run(cmd, cwd=str(project_root))
    elif cfg.flashhead.mode == "pro" and nproc > 1 and _is_torchrun():
        # pro 多卡模式 torchrun 子进程
        rank = int(os.environ.get("LOCAL_RANK", 0))
        if rank == 0:
            # rank 0: 启动 API 服务（pipeline 在 lifespan 中初始化）
            from service.app import main_service
            main_service()
        else:
            # 其他 rank: 初始化 pipeline 后进入 worker 循环
            # 等待 rank 0 的 broadcast 信号同步执行推理
            from cores.pipeline_adapter import init_pipeline, run_worker_loop
            init_pipeline()
            run_worker_loop()
    elif cfg.flashhead.mode == "pro" and nproc == 1:
        # pro 单卡模式：使用 Model_Pro 但不走分布式
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.flashhead.pro_device_ids
        from service.app import main_service
        main_service()
    else:
        # lite 模式：单卡直接启动
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_ids
        from service.app import main_service
        main_service()
