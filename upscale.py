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


def remove_background(input_path, output_path):
    """画像の背景を削除する（rembg 使用）
    出力は透過 PNG のみ
    """
    from PIL import Image
    from rembg import remove as rembg_remove

    print("[1/3] 画像を読み込み中...")
    input_img = Image.open(input_path).convert("RGB")
    w, h = input_img.size
    print(f"      入力: {w}x{h}px")

    print("[2/3] 背景削除中...")
    output_img = rembg_remove(input_img)

    print("[3/3] 保存中...")
    output_img.save(output_path, "PNG")
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

def upscale(input_path, output_path, mode="photo", scale=4):
    print("[1/4] モデルを読み込み中...")
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
        tile=256,        # CPU用にタイル分割処理
        tile_pad=10,
        pre_pad=0,
        half=False       # CPUはfloat32
    )
    print("      モデル読み込み完了")

    print("[2/4] 画像を読み込み中...")
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    print(f"      入力: {w}x{h}px")

    print("[3/4] アップスケール処理中...")
    output, _ = upsampler.enhance(img, outscale=scale)

    print("[4/4] 保存中...")
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