"""
LaMa inpainting のラッパー。CUDA 未対応環境では map_location='cpu' でフォールバック。
"""
import logging

logger = logging.getLogger(__name__)

def create_simple_lama():
    """SimpleLama を生成。CUDA 未対応時は map_location='cpu' で読み込む。"""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_load = torch.jit.load
    def patched_load(path, *args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = "cpu"
        return orig_load(path, *args, **kwargs)
    torch.jit.load = patched_load
    try:
        from simple_lama_inpainting import SimpleLama
        return SimpleLama(device=device)
    finally:
        torch.jit.load = orig_load
