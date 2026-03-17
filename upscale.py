import argparse
import subprocess
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _enhance_with_progress(upsampler, img, scale, progress_callback):
    """enhance を実行し、progress_callback があればタイル進捗を通知"""
    if progress_callback and upsampler.tile_size > 0:
        from tqdm import tqdm
        import realesrgan.utils as ru

        tile_pass = [0]  # 1=色情報, 2=透過情報（RGBA時のみ）

        class ProgressTqdm(tqdm):
            def update(self, n=1):
                super().update(n)
                if self.total and self.n <= self.total:
                    if self.n == 1 and tile_pass[0] == 1:
                        tile_pass[0] = 2
                    elif self.n == self.total:
                        tile_pass[0] = 1
                    phase = "（透過部分）" if tile_pass[0] == 2 else ""
                    progress_callback(3, f"区画 {self.n}/{self.total} をAIで拡大中{phase}（1区画あたり約5秒）", {"current": int(self.n), "total": int(self.total), "phase": tile_pass[0]})

        orig_tqdm = ru.tqdm
        ru.tqdm = ProgressTqdm
        try:
            return upsampler.enhance(img, outscale=scale)
        finally:
            ru.tqdm = orig_tqdm
    return upsampler.enhance(img, outscale=scale)


def _format_size(size_bytes):
    """バイト数を読みやすい形式に変換"""
    for unit in ("B", "KB", "MB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}GB"

# 自動最適化時の長辺の最大px（Web用途を想定）
AUTO_OPTIMIZE_MAX_SIDE = 1920

def optimize(input_path, output_path, max_width=None, max_height=None, quality=85, auto=True):
    """画像を軽量化する（リサイズ・圧縮）
    auto=True かつ max_width/max_height 未指定時は長辺1920pxに自動リサイズ
    """
    print("[1/3] 画像を読み込み中...")
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {input_path}")

    h, w = img.shape[:2]
    input_size = os.path.getsize(input_path)
    print(f"      入力: {w}x{h}px ({_format_size(input_size)})")

    # リサイズ（自動 or 手動指定）
    if max_width or max_height:
        # 手動指定
        print("[2/3] リサイズ中（指定サイズ）...")
        scale = 1.0
        if max_width and w > max_width:
            scale = min(scale, max_width / w)
        if max_height and h > max_height:
            scale = min(scale, max_height / h)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"      {w}x{h} → {new_w}x{new_h}px")
        else:
            print("      リサイズ不要")
    elif auto:
        # 自動最適化：長辺が閾値を超えていれば縮小
        max_side = max(w, h)
        if max_side > AUTO_OPTIMIZE_MAX_SIDE:
            print(f"[2/3] リサイズ中（自動: 長辺{AUTO_OPTIMIZE_MAX_SIDE}px以下）...")
            scale = AUTO_OPTIMIZE_MAX_SIDE / max_side
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"      {w}x{h} → {new_w}x{new_h}px")
        else:
            print("[2/3] リサイズ不要（既に最適サイズ）")
    else:
        print("[2/3] リサイズスキップ")

    print("[3/3] 保存中...")
    ext = Path(output_path).suffix.lower()

    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == ".png":
        cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])  # 最大圧縮
    elif ext in (".webp",):
        cv2.imwrite(output_path, img, [cv2.IMWRITE_WEBP_QUALITY, quality])
    else:
        cv2.imwrite(output_path, img)

    output_size = os.path.getsize(output_path)
    ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0
    print(f"      保存しました: {output_path}")
    if ratio > 0:
        print(f"      出力: {_format_size(output_size)} (約{ratio:.0f}%削減)")
    elif ratio < 0:
        print(f"      出力: {_format_size(output_size)} (約{-ratio:.0f}%増加)")
    else:
        print(f"      出力: {_format_size(output_size)}")

    return {
        "input_size": input_size,
        "output_size": output_size,
        "reduction_percent": round(ratio, 1),
    }


def optimize_video(input_path, output_path, max_width=None, max_height=None, crf=18):
    """MP4等の動画を軽量化する（品質を損なわない設定）
    ffmpeg が必要: brew install ffmpeg
    crf=18: 視覚的にロスレスに近い品質
    """
    if not shutil.which("ffmpeg"):
        raise ValueError("ffmpeg がインストールされていません。macOS: brew install ffmpeg")

    input_path = Path(input_path)
    output_path = Path(output_path)
    input_size = input_path.stat().st_size
    print(f"[1/2] 動画を読み込み中... ({_format_size(input_size)})")

    cmd = ["ffmpeg", "-y", "-i", str(input_path)]
    if max_width or max_height:
        w = f"min(iw\\,{max_width})" if max_width else "iw"
        h = f"min(ih\\,{max_height})" if max_height else "ih"
        cmd.extend(["-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease"])
    cmd.extend([
        "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(output_path),
    ])

    print("[2/2] エンコード中（品質維持モード CRF18）...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg エラー: {result.stderr[-500:] if result.stderr else '不明'}")

    output_size = output_path.stat().st_size
    ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0
    print(f"      保存しました: {output_path} ({_format_size(output_size)}, 約{ratio:.0f}%削減)")

    return {
        "input_size": input_size,
        "output_size": output_size,
        "reduction_percent": round(ratio, 1),
    }


def crop(input_path, output_path, x_pct, y_pct, w_pct, h_pct, quality=95):
    """画像を切り抜く（割合指定 0-100）
    x_pct, y_pct: 左上の位置（%）
    w_pct, h_pct: 幅・高さ（%）
    """
    print("[1/3] 画像を読み込み中...")
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {input_path}")

    h, w = img.shape[:2]
    print(f"      入力: {w}x{h}px")

    x = int(w * x_pct / 100)
    y = int(h * y_pct / 100)
    cw = int(w * w_pct / 100)
    ch = int(h * h_pct / 100)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    cw = max(1, min(cw, w - x))
    ch = max(1, min(ch, h - y))

    print("[2/3] 切り抜き中...")
    cropped = img[y:y + ch, x:x + cw]
    print(f"      {w}x{h} → {cw}x{ch}px")

    print("[3/3] 保存中...")
    ext = Path(output_path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(output_path, cropped, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == ".png":
        cv2.imwrite(output_path, cropped, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    elif ext in (".webp",):
        cv2.imwrite(output_path, cropped, [cv2.IMWRITE_WEBP_QUALITY, quality])
    else:
        cv2.imwrite(output_path, cropped)
    print(f"      保存しました: {output_path}")


_rembg_session = None


def _get_rembg_session():
    """GPU 利用可能なら CUDA でセッション作成（キャッシュ）"""
    global _rembg_session
    if _rembg_session is not None:
        return _rembg_session
    try:
        import onnxruntime as ort
        from rembg.session_factory import new_session
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            _rembg_session = new_session("u2net", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        else:
            _rembg_session = new_session("u2net")
    except Exception:
        from rembg.session_factory import new_session
        _rembg_session = new_session("u2net")
    return _rembg_session


def remove_background(input_path, output_path):
    """画像の背景を削除する（rembg 使用）
    出力は透過 PNG のみ。GPU 利用可能時は自動で CUDA 使用
    """
    from PIL import Image
    from rembg import remove as rembg_remove

    session = _get_rembg_session()
    print("[1/3] 画像を読み込み中...")
    input_img = Image.open(input_path).convert("RGB")
    w, h = input_img.size
    print(f"      入力: {w}x{h}px")

    print("[2/3] 背景削除中...")
    output_img = rembg_remove(input_img, session=session)

    print("[3/3] 保存中...")
    output_img.save(output_path, "PNG")
    print(f"      保存しました: {output_path}")


def remove_background_bright(input_path, output_path, min_rgb=240):
    """明るい色（白〜薄いグレー）を透過。スクリーンショット・オフホワイト背景向け。
    R,G,B がすべて min_rgb 以上のピクセルを透明にする。緑ボタンなどは R が低いので残る。
    """
    from PIL import Image
    import numpy as np

    input_path = Path(input_path) if not isinstance(input_path, Path) else input_path
    input_size = input_path.stat().st_size if input_path.exists() else 0
    print(f"[1/3] 画像を読み込み中... {input_path.name} ({input_size} bytes)")

    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"      入力: {w}x{h}px")

    print("[2/3] 明るい色を透過中...")
    rgb = arr[:, :, :3]
    bright = (rgb[:, :, 0] >= min_rgb) & (rgb[:, :, 1] >= min_rgb) & (rgb[:, :, 2] >= min_rgb)
    arr[:, :, 3] = np.where(bright, 0, 255).astype(np.uint8)

    kept = int(np.sum(arr[:, :, 3] == 255))
    removed = int(np.sum(arr[:, :, 3] == 0))
    total = h * w
    print(f"      残す: {kept}px, 透過: {removed}px (合計{total}px)")

    if kept == 0:
        # 全部透過 = 閾値が緩すぎた可能性。min_rgb を 250 に上げて厳格に（白に近いものだけ透過）
        print("      警告: 残るピクセルが0。min_rgb を 250 に上げて再試行（白に近いもののみ透過）...")
        min_rgb = 250
        bright = (rgb[:, :, 0] >= min_rgb) & (rgb[:, :, 1] >= min_rgb) & (rgb[:, :, 2] >= min_rgb)
        arr[:, :, 3] = np.where(bright, 0, 255).astype(np.uint8)
        kept = int(np.sum(arr[:, :, 3] == 255))
        print(f"      再試行後 残す: {kept}px")

    print("[3/3] 保存中...")
    Image.fromarray(arr).save(output_path, "PNG")
    print(f"      保存しました: {output_path}")


def remove_background_by_color(input_path, output_path, threshold=25, sample_margin=0.02, bg_color_hex=None):
    """単色背景を削除（四隅をサンプリング or 指定色で類似色を透明に）
    バナーや白背景の画像向け。AI で全部消える場合の代替手段。
    threshold: 色の類似度（0-100、大きいほど多くのピクセルが透明になる）
    sample_margin: 四隅のサンプル範囲（画像サイズに対する割合、0.02=2%）
    bg_color_hex: 背景色を指定（例: #FFFFFF, FFFFFF）。None なら四隅から自動検出
    """
    from PIL import Image
    import numpy as np
    import re

    print("[1/3] 画像を読み込み中...")
    img = Image.open(input_path).convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]
    print(f"      入力: {w}x{h}px")

    if bg_color_hex:
        # 指定色をパース（#FFFFFF, FFFFFF, rgb(255,255,255) など）
        hex_val = bg_color_hex.strip().lstrip("#")
        m = re.match(r"^([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$", hex_val)
        if m:
            r, g, b = int(m.group(1), 16), int(m.group(2), 16), int(m.group(3), 16)
            bg_color = np.array([r, g, b, 255], dtype=np.float32)
            print(f"      背景色指定: #{hex_val}")
        else:
            raise ValueError(f"背景色の形式が不正です: {bg_color_hex} （例: #FFFFFF または FFFFFF）")
    else:
        # 四隅をサンプリング（マージン内のピクセル平均）
        m = max(1, int(min(w, h) * sample_margin))
        corners = [
            arr[:m, :m].reshape(-1, 4).mean(axis=0),   # 左上
            arr[:m, -m:].reshape(-1, 4).mean(axis=0),  # 右上
            arr[-m:, :m].reshape(-1, 4).mean(axis=0),  # 左下
            arr[-m:, -m:].reshape(-1, 4).mean(axis=0), # 右下
        ]
        bg_color = np.array(corners).mean(axis=0).astype(np.float32)
    # RGB のみで距離計算（Alpha は無視）
    threshold_val = threshold * (255 / 100) * np.sqrt(3)  # 正規化

    print("[2/3] 単色背景を削除中...")
    rgb = arr[:, :, :3].astype(np.float32)
    bg_rgb = bg_color[:3]
    dist = np.sqrt(np.sum((rgb - bg_rgb) ** 2, axis=2))
    mask = dist > threshold_val
    arr[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)

    print("[3/3] 保存中...")
    Image.fromarray(arr).save(output_path, "PNG")
    print(f"      保存しました: {output_path}")


def convert(input_path, output_path, quality=95):
    """画像の形式を変換する（例: jpg→png, png→jpg）"""
    print("[1/2] 画像を読み込み中...")
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {input_path}")

    h, w = img.shape[:2]
    print(f"      入力: {w}x{h}px ({Path(input_path).suffix})")

    print("[2/2] 保存中...")
    ext = Path(output_path).suffix.lower()

    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == ".png":
        cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    elif ext in (".webp",):
        cv2.imwrite(output_path, img, [cv2.IMWRITE_WEBP_QUALITY, quality])
    else:
        cv2.imwrite(output_path, img)

    print(f"      保存しました: {output_path}")

def upscale(input_path, output_path, mode="photo", scale=4, target_width=None, target_height=None, progress_callback=None):
    """画像をアップスケール。progress_callback(step, msg, extra) で進捗を通知"""
    def report(step, msg, extra=None):
        print(f"[{step}/4] {msg}")
        if progress_callback:
            progress_callback(step, msg, extra or {})

    report(1, "画像ファイルを開いています...")
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {input_path}")
    h, w = img.shape[:2]
    print(f"      入力: {w}x{h}px")
    if progress_callback:
        progress_callback(1, f"読み込み完了: {w}×{h}px の画像", {"w": w, "h": h})

    # 目標サイズが元より小さい場合はリサイズのみ（モデル不要）
    if target_width is not None and target_height is not None:
        tw, th = int(target_width), int(target_height)
        if tw <= w and th <= h:
            report(4, "拡大した画像をファイルに保存しています...")
            output = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
            cv2.imwrite(output_path, output)
            print(f"      保存しました: {output_path}")
            return

    import torch
    use_gpu = torch.cuda.is_available()
    tile_size = 0 if use_gpu else 256  # GPU時はタイル不要で高速化

    report(2, "AIモデルを読み込んでいます（約64MB・初回は時間がかかります）...")
    if mode == "anime":
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=6, num_grow_ch=32, scale=4)
        model_path = "weights/RealESRGAN_x4plus_anime_6B.pth"
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        model_path = "weights/RealESRGAN_x4plus.pth"

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=tile_size,
        tile_pad=10,
        pre_pad=0,
        half=use_gpu       # GPU時はfp16で高速化
    )
    if progress_callback:
        progress_callback(2, "AIモデル読み込み完了", {})
    print("      モデル読み込み完了")

    # 目標サイズから倍率を決定（縮小は上で return 済み）
    if target_width is not None and target_height is not None:
        tw, th = int(target_width), int(target_height)
        # アップスケール: 2x で足りれば2x、それ以外は4x
        report(3, "アップスケール処理中...")
        scale = 2 if (2 * w >= tw and 2 * h >= th) else 4
        output, _ = _enhance_with_progress(upsampler, img, scale, progress_callback)
        out_h, out_w = output.shape[:2]
        if out_w != tw or out_h != th:
            output = cv2.resize(output, (tw, th), interpolation=cv2.INTER_LANCZOS4)
    else:
        report(3, "画像を拡大しています（区画ごとにAI処理・数分かかります）...")
        output, _ = _enhance_with_progress(upsampler, img, scale, progress_callback)

    report(4, "拡大した画像をファイルに保存しています...")
    cv2.imwrite(output_path, output)
    print(f"      保存しました: {output_path}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "optimize":
        # optimize: 軽量化（画像・動画）
        parser = argparse.ArgumentParser(description="画像・動画を軽量化")
        parser.add_argument("input", help="入力ファイルパス")
        parser.add_argument("output", help="出力ファイルパス")
        parser.add_argument("--max-width", "-W", type=int, help="最大幅")
        parser.add_argument("--max-height", "-H", type=int, help="最大高さ")
        parser.add_argument("--no-auto", action="store_true", help="自動リサイズを無効化（画像のみ）")
        parser.add_argument("--quality", "-q", type=int, default=85,
                           help="JPEG/WebPの品質 (1-100, 画像のみ)")
        parser.add_argument("--crf", type=int, default=18,
                           help="動画の品質 (0-51, 18=高品質, 動画のみ)")
        args = parser.parse_args(sys.argv[2:])
        ext = Path(args.input).suffix.lower()
        if ext in VIDEO_EXTENSIONS:
            optimize_video(args.input, args.output, args.max_width, args.max_height, args.crf)
        else:
            auto = not args.no_auto and not (args.max_width or args.max_height)
            optimize(args.input, args.output, args.max_width, args.max_height, args.quality, auto=auto)
    elif len(sys.argv) >= 2 and sys.argv[1] == "removebg":
        # removebg: 背景削除
        parser = argparse.ArgumentParser(description="画像の背景を削除")
        parser.add_argument("input", help="入力画像パス")
        parser.add_argument("output", help="出力画像パス（PNG）")
        args = parser.parse_args(sys.argv[2:])
        remove_background(args.input, args.output)
    elif len(sys.argv) >= 2 and sys.argv[1] == "crop":
        # crop: トリミング（割合指定）
        parser = argparse.ArgumentParser(description="画像を切り抜き")
        parser.add_argument("input", help="入力画像パス")
        parser.add_argument("output", help="出力画像パス")
        parser.add_argument("--x", type=float, default=0, help="左上 X (%)")
        parser.add_argument("--y", type=float, default=0, help="左上 Y (%)")
        parser.add_argument("--w", type=float, default=100, help="幅 (%)")
        parser.add_argument("--h", type=float, default=100, help="高さ (%)")
        parser.add_argument("--quality", "-q", type=int, default=95)
        args = parser.parse_args(sys.argv[2:])
        crop(args.input, args.output, args.x, args.y, args.w, args.h, args.quality)
    elif len(sys.argv) >= 2 and sys.argv[1] == "convert":
        # convert: 形式変換
        parser = argparse.ArgumentParser(description="画像の形式を変換（jpg→png など）")
        parser.add_argument("input", help="入力画像パス")
        parser.add_argument("output", help="出力画像パス")
        parser.add_argument("--quality", "-q", type=int, default=95,
                           help="JPEG/WebPの品質 (1-100, デフォルト: 95)")
        args = parser.parse_args(sys.argv[2:])  # "convert" をスキップ
        convert(args.input, args.output, args.quality)
    else:
        # upscale: アップスケール（デフォルト・従来の使い方）
        parser = argparse.ArgumentParser(description="画像をアップスケール")
        parser.add_argument("input", help="入力画像パス")
        parser.add_argument("output", help="出力画像パス")
        parser.add_argument("--mode", choices=["photo", "anime"], default="photo")
        parser.add_argument("--scale", type=int, default=4)
        args = parser.parse_args()
        upscale(args.input, args.output, args.mode, args.scale)