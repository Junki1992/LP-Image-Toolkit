"""
CTA アニメーション — 画像に動きを付けて GIF またはソースコードを生成
"""
from __future__ import annotations

import math
from typing import Optional
from io import BytesIO
from pathlib import Path

from PIL import Image
import numpy as np

EFFECTS = {
    "pulse": {"label": "パルス", "desc": "拡大縮小＋明るさ変化"},
    "glow": {"label": "グロー", "desc": "外側に光が広がる"},
    "bounce": {"label": "バウンス", "desc": "上下に弾む"},
    "shake": {"label": "シェイク", "desc": "左右に揺れる"},
    "float": {"label": "フロート", "desc": "ゆっくり上下に浮く"},
    "wiggle": {"label": "ウィグル", "desc": "小刻みに揺れる"},
    "fade": {"label": "フェード", "desc": "透明度が脈打つ"},
    "rotate": {"label": "ローテート", "desc": "軽く回転"},
    "swing": {"label": "スイング", "desc": "振り子のように揺れる"},
    "heartbeat": {"label": "ハートビート", "desc": "心拍のような動き"},
    "rubber": {"label": "ラバー", "desc": "ゴムのように伸び縮み"},
    "breathe": {"label": "ブリーズ", "desc": "ゆっくり膨らむ・縮む"},
    "attention": {"label": "アテンション", "desc": "一瞬大きく動いて注目を集める"},
    "shine": {"label": "シャイン", "desc": "光が走る（JS）"},
    "ripple": {"label": "リップル", "desc": "クリックで波紋が広がる（JS）"},
    "tilt": {"label": "チルト", "desc": "3Dで傾いて浮く"},
}


def _ensure_rgba(img: Image.Image) -> Image.Image:
    """RGBA に統一"""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return img


def _apply_pulse_frame(img: np.ndarray, t: float, scale_min: float = 0.95, scale_max: float = 1.05, brightness_min: float = 0.9, brightness_max: float = 1.1) -> np.ndarray:
    """パルス: 1周期で scale と brightness が変化"""
    # t: 0..1
    phase = math.sin(t * 2 * math.pi)
    scale = scale_min + (scale_max - scale_min) * (phase + 1) / 2
    brightness = brightness_min + (brightness_max - brightness_min) * (phase + 1) / 2
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    pil = Image.fromarray(img)
    scaled = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # 中央に配置
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = (w - new_w) // 2
    y = (h - new_h) // 2
    out.paste(scaled, (x, y), scaled)
    arr = np.array(out)
    # brightness (RGB only)
    arr[:, :, :3] = np.clip(arr[:, :, :3].astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return arr


def _apply_glow_frame(img: np.ndarray, t: float, intensity: float = 1.2) -> np.ndarray:
    """グロー: box-shadow 風に外側が明るくなる（簡易版: 明るさパルス）"""
    phase = math.sin(t * 2 * math.pi)
    brightness = 0.9 + 0.3 * (phase + 1) / 2
    arr = img.copy().astype(np.float32)
    arr[:, :, :3] = np.clip(arr[:, :, :3] * brightness, 0, 255)
    return arr.astype(np.uint8)


def _apply_bounce_frame(img: np.ndarray, t: float, amp: float = 8) -> np.ndarray:
    """バウンス: 上下に弾む"""
    # イージングでバウンス風
    phase = math.sin(t * 2 * math.pi)
    offset_y = int(amp * phase)
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    src_y0 = max(0, -offset_y)
    src_y1 = min(h, h - offset_y)
    dst_y0 = max(0, offset_y)
    dst_y1 = min(h, h + offset_y)
    out[dst_y0:dst_y1, :] = img[src_y0:src_y1, :]
    return out


def _apply_shake_frame(img: np.ndarray, t: float, amp: float = 4) -> np.ndarray:
    """シェイク: 左右に揺れる"""
    phase = math.sin(t * 2 * math.pi * 2)  # 速め
    offset_x = int(amp * phase)
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    src_x0 = max(0, -offset_x)
    src_x1 = min(w, w - offset_x)
    dst_x0 = max(0, offset_x)
    dst_x1 = min(w, w + offset_x)
    out[:, dst_x0:dst_x1] = img[:, src_x0:src_x1]
    return out


def _apply_float_frame(img: np.ndarray, t: float, amp: float = 6) -> np.ndarray:
    """フロート: ゆっくり上下に浮く"""
    phase = math.sin(t * 2 * math.pi * 0.5)  # ゆっくり
    offset_y = int(amp * phase)
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    src_y0 = max(0, -offset_y)
    src_y1 = min(h, h - offset_y)
    dst_y0 = max(0, offset_y)
    dst_y1 = min(h, h + offset_y)
    out[dst_y0:dst_y1, :] = img[src_y0:src_y1, :]
    return out


def _apply_wiggle_frame(img: np.ndarray, t: float, amp: float = 3) -> np.ndarray:
    """ウィグル: 小刻みに揺れる"""
    phase = math.sin(t * 2 * math.pi * 4)
    offset_x = int(amp * phase)
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    src_x0 = max(0, -offset_x)
    src_x1 = min(w, w - offset_x)
    dst_x0 = max(0, offset_x)
    dst_x1 = min(w, w + offset_x)
    out[:, dst_x0:dst_x1] = img[:, src_x0:src_x1]
    return out


def _apply_fade_frame(img: np.ndarray, t: float) -> np.ndarray:
    """フェード: 透明度パルス"""
    phase = math.sin(t * 2 * math.pi)
    alpha = 0.7 + 0.3 * (phase + 1) / 2
    arr = img.copy().astype(np.float32)
    arr[:, :, 3] = arr[:, :, 3] * alpha
    return np.clip(arr, 0, 255).astype(np.uint8)


def _apply_rotate_frame(img: np.ndarray, t: float, deg: float = 5) -> np.ndarray:
    """ローテート: 軽く回転"""
    angle = deg * math.sin(t * 2 * math.pi) * math.pi / 180
    pil = Image.fromarray(img)
    rotated = pil.rotate(-angle * 180 / math.pi, resample=Image.Resampling.BICUBIC, expand=False)
    return np.array(rotated)


def _apply_swing_frame(img: np.ndarray, t: float, deg: float = 8) -> np.ndarray:
    """スイング: 振り子のように揺れる"""
    phase = math.sin(t * 2 * math.pi)
    angle = deg * phase * math.pi / 180
    pil = Image.fromarray(img)
    rotated = pil.rotate(-angle * 180 / math.pi, resample=Image.Resampling.BICUBIC, expand=False)
    return np.array(rotated)


def _apply_heartbeat_frame(img: np.ndarray, t: float) -> np.ndarray:
    """ハートビート: ドクン・ドクン"""
    # 0-0.15: 拡大, 0.15-0.3: 縮小, 0.3-0.45: 小拡大, 0.45-0.6: 縮小, 0.6-1: 静止
    if t < 0.15:
        scale = 1 + 0.08 * (t / 0.15)
    elif t < 0.3:
        scale = 1.08 - 0.08 * ((t - 0.15) / 0.15)
    elif t < 0.45:
        scale = 1 + 0.04 * ((t - 0.3) / 0.15)
    elif t < 0.6:
        scale = 1.04 - 0.04 * ((t - 0.45) / 0.15)
    else:
        scale = 1.0
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    pil = Image.fromarray(img)
    scaled = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x, y = (w - new_w) // 2, (h - new_h) // 2
    out.paste(scaled, (x, y), scaled)
    return np.array(out)


def _apply_rubber_frame(img: np.ndarray, t: float) -> np.ndarray:
    """ラバー: ゴム伸縮"""
    phase = math.sin(t * 2 * math.pi)
    scale_x = 1 + 0.08 * phase
    scale_y = 1 - 0.04 * phase
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale_x), int(h * scale_y)
    pil = Image.fromarray(img)
    scaled = pil.resize((max(1, new_w), max(1, new_h)), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x, y = (w - scaled.width) // 2, (h - scaled.height) // 2
    out.paste(scaled, (x, y), scaled)
    return np.array(out)


def _apply_breathe_frame(img: np.ndarray, t: float) -> np.ndarray:
    """ブリーズ: ゆっくり膨張"""
    phase = math.sin(t * 2 * math.pi * 0.4)
    scale = 0.97 + 0.06 * (phase + 1) / 2
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    pil = Image.fromarray(img)
    scaled = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x, y = (w - new_w) // 2, (h - new_h) // 2
    out.paste(scaled, (x, y), scaled)
    return np.array(out)


def _apply_attention_frame(img: np.ndarray, t: float) -> np.ndarray:
    """アテンション: 0-0.2で一瞬拡大、その後戻る"""
    if t < 0.2:
        scale = 1 + 0.12 * (1 - t / 0.2)
    else:
        scale = 1.0
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    pil = Image.fromarray(img)
    scaled = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x, y = (w - new_w) // 2, (h - new_h) // 2
    out.paste(scaled, (x, y), scaled)
    return np.array(out)


def _apply_shine_frame(img: np.ndarray, t: float) -> np.ndarray:
    """シャイン: 光が左から右へ走る（GIF近似）"""
    h, w = img.shape[:2]
    arr = img.copy().astype(np.float32)
    x_center = int(w * t)
    for x in range(w):
        dist = abs(x - x_center)
        boost = max(0, 1.3 - dist / (w * 0.3))
        arr[:, x, :3] = np.clip(arr[:, x, :3] * boost, 0, 255)
    return arr.astype(np.uint8)


def _apply_ripple_frame(img: np.ndarray, t: float) -> np.ndarray:
    """リップル: 中心から波紋（GIF近似: 明るさの波）"""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    yg, xg = np.ogrid[:h, :w]
    d = np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2)
    wave = np.sin(d / 10 - t * 2 * math.pi * 3) * 0.1 + 1
    arr = img.copy().astype(np.float32)
    arr[:, :, :3] = np.clip(arr[:, :, :3] * wave[:, :, np.newaxis], 0, 255)
    return arr.astype(np.uint8)


def _apply_tilt_frame(img: np.ndarray, t: float) -> np.ndarray:
    """チルト: 3D風に傾く（GIF近似: せん断）"""
    phase = math.sin(t * 2 * math.pi)
    skew = 0.02 * phase
    h, w = img.shape[:2]
    pil = Image.fromarray(img)
    coeffs = (1, skew, 0, -skew, 1, 0)
    try:
        transformed = pil.transform((w, h), Image.Transform.AFFINE, coeffs)
    except AttributeError:
        transformed = pil.transform((w, h), Image.AFFINE, coeffs)
    except Exception:
        transformed = pil
    return np.array(transformed)


def _get_effect_func(effect: str):
    f = {
        "pulse": _apply_pulse_frame,
        "glow": _apply_glow_frame,
        "bounce": _apply_bounce_frame,
        "shake": _apply_shake_frame,
        "float": _apply_float_frame,
        "wiggle": _apply_wiggle_frame,
        "fade": _apply_fade_frame,
        "rotate": _apply_rotate_frame,
        "swing": _apply_swing_frame,
        "heartbeat": _apply_heartbeat_frame,
        "rubber": _apply_rubber_frame,
        "breathe": _apply_breathe_frame,
        "attention": _apply_attention_frame,
        "shine": _apply_shine_frame,
        "ripple": _apply_ripple_frame,
        "tilt": _apply_tilt_frame,
    }.get(effect)
    if not f:
        raise ValueError(f"未知のエフェクト: {effect}")
    return f


def generate_gif(
    image_path: str,
    effect: str,
    output_path: str,
    duration: float = 1.5,
    fps: int = 20,
    loop: int = 0,
) -> None:
    """
    画像にエフェクトを適用して GIF を生成
    duration: 1周期の秒数
    fps: フレーム数/秒
    loop: 0=無限
    """
    pil = Image.open(image_path)
    pil = _ensure_rgba(pil)
    img = np.array(pil)
    func = _get_effect_func(effect)
    frames = []
    num_frames = max(8, int(duration * fps))
    for i in range(num_frames):
        t = i / num_frames
        frame = func(img, t)
        frames.append(Image.fromarray(frame))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 * duration / num_frames) * num_frames // num_frames or 50,
        loop=loop,
    )


def generate_code(
    image_path: str,
    effect: str,
    image_filename: Optional[str] = None,
) -> str:
    """
    指定エフェクトの HTML + CSS ソースコードを生成
    image_filename: 画像のファイル名（コード内で参照、None の場合はプレースホルダ）
    """
    fname = image_filename or "cta-image.png"
    if effect == "shine":
        return _generate_shine_code(fname)
    if effect == "ripple":
        return _generate_ripple_code(fname)
    effect_props, effect_keyframes = _get_effect_css(effect)
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CTA アニメーション</title>
  <style>
    .cta-wrap {{
      display: inline-block;
      {effect_props}
    }}
    {effect_keyframes}
    .cta-wrap img {{
      display: block;
      max-width: 100%;
      height: auto;
    }}
  </style>
</head>
<body>
  <div class="cta-wrap">
    <img src="{fname}" alt="CTA">
  </div>
</body>
</html>
"""


def _get_effect_css(effect: str):
    """エフェクトごとの CSS。(animation プロパティ, @keyframes ブロック)"""
    data = {
        "pulse": ("animation: cta-pulse 1.5s ease-in-out infinite;",
                  "@keyframes cta-pulse {\n      0%, 100% { transform: scale(1); filter: brightness(1); }\n      50% { transform: scale(1.05); filter: brightness(1.1); }\n    }"),
        "glow": ("animation: cta-glow 1.5s ease-in-out infinite;",
                 "@keyframes cta-glow {\n      0%, 100% { filter: drop-shadow(0 0 4px rgba(255,255,255,0.5)); }\n      50% { filter: drop-shadow(0 0 12px rgba(255,255,255,0.9)); }\n    }"),
        "bounce": ("animation: cta-bounce 1s ease-in-out infinite;",
                   "@keyframes cta-bounce {\n      0%, 100% { transform: translateY(0); }\n      50% { transform: translateY(-8px); }\n    }"),
        "shake": ("animation: cta-shake 0.5s ease-in-out infinite;",
                  "@keyframes cta-shake {\n      0%, 100% { transform: translateX(0); }\n      25% { transform: translateX(-4px); }\n      75% { transform: translateX(4px); }\n    }"),
        "float": ("animation: cta-float 2s ease-in-out infinite;",
                 "@keyframes cta-float {\n      0%, 100% { transform: translateY(0); }\n      50% { transform: translateY(-6px); }\n    }"),
        "wiggle": ("animation: cta-wiggle 0.4s ease-in-out infinite;",
                   "@keyframes cta-wiggle {\n      0%, 100% { transform: rotate(-2deg); }\n      50% { transform: rotate(2deg); }\n    }"),
        "fade": ("animation: cta-fade 1.5s ease-in-out infinite;",
                 "@keyframes cta-fade {\n      0%, 100% { opacity: 1; }\n      50% { opacity: 0.75; }\n    }"),
        "rotate": ("animation: cta-rotate 2s ease-in-out infinite;",
                   "@keyframes cta-rotate {\n      0%, 100% { transform: rotate(-5deg); }\n      50% { transform: rotate(5deg); }\n    }"),
        "swing": ("animation: cta-swing 1s ease-in-out infinite;",
                  "@keyframes cta-swing {\n      0%, 100% { transform: rotate(-8deg); }\n      50% { transform: rotate(8deg); }\n    }"),
        "heartbeat": ("animation: cta-heartbeat 1.2s ease-in-out infinite;",
                      "@keyframes cta-heartbeat {\n      0%, 100% { transform: scale(1); }\n      14% { transform: scale(1.08); }\n      28% { transform: scale(1); }\n      42% { transform: scale(1.04); }\n      56% { transform: scale(1); }\n    }"),
        "rubber": ("animation: cta-rubber 1s ease-in-out infinite;",
                   "@keyframes cta-rubber {\n      0%, 100% { transform: scaleX(1) scaleY(1); }\n      30% { transform: scaleX(1.08) scaleY(0.96); }\n      65% { transform: scaleX(0.96) scaleY(1.04); }\n    }"),
        "breathe": ("animation: cta-breathe 3s ease-in-out infinite;",
                    "@keyframes cta-breathe {\n      0%, 100% { transform: scale(0.97); }\n      50% { transform: scale(1.03); }\n    }"),
        "attention": ("animation: cta-attention 2s ease-out infinite;",
                      "@keyframes cta-attention {\n      0%, 100% { transform: scale(1); }\n      10% { transform: scale(1.12); }\n      20% { transform: scale(1); }\n    }"),
        "tilt": ("animation: cta-tilt 2s ease-in-out infinite;",
                 "@keyframes cta-tilt {\n      0%, 100% { transform: perspective(400px) rotateX(-5deg) translateY(0); }\n      50% { transform: perspective(400px) rotateX(-5deg) translateY(-8px); }\n    }"),
    }
    return data.get(effect, ("", ""))


def _generate_shine_code(fname: str) -> str:
    """シャイン: 光が走る（CSS 疑似要素）"""
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CTA アニメーション - シャイン</title>
  <style>
    .cta-wrap {{
      display: inline-block;
      position: relative;
      overflow: hidden;
    }}
    .cta-wrap::after {{
      content: '';
      position: absolute;
      top: 0; left: -100%;
      width: 50%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
      animation: cta-shine 2s ease-in-out infinite;
    }}
    @keyframes cta-shine {{
      0% {{ left: -100%; }}
      50% {{ left: 150%; }}
      100% {{ left: 150%; }}
    }}
    .cta-wrap img {{
      display: block;
      max-width: 100%;
      height: auto;
    }}
  </style>
</head>
<body>
  <div class="cta-wrap">
    <img src="{fname}" alt="CTA">
  </div>
</body>
</html>
"""


def _generate_ripple_code(fname: str) -> str:
    """リップル: クリックで波紋（JS）"""
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CTA アニメーション - リップル</title>
  <style>
    .cta-wrap {{
      display: inline-block;
      position: relative;
      overflow: hidden;
      cursor: pointer;
    }}
    .cta-wrap img {{
      display: block;
      max-width: 100%;
      height: auto;
    }}
    .cta-ripple {{
      position: absolute;
      border-radius: 50%;
      background: rgba(255,255,255,0.5);
      transform: scale(0);
      animation: cta-ripple 0.6s ease-out forwards;
      pointer-events: none;
    }}
    @keyframes cta-ripple {{
      to {{ transform: scale(4); opacity: 0; }}
    }}
  </style>
</head>
<body>
  <div class="cta-wrap" id="ctaRipple">
    <img src="{fname}" alt="CTA">
  </div>
  <script>
    const wrap = document.getElementById('ctaRipple');
    wrap.addEventListener('click', function(e) {{
      const rect = wrap.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const ripple = document.createElement('span');
      ripple.className = 'cta-ripple';
      ripple.style.width = ripple.style.height = '100px';
      ripple.style.left = (x - 50) + 'px';
      ripple.style.top = (y - 50) + 'px';
      wrap.appendChild(ripple);
      setTimeout(() => ripple.remove(), 600);
    }});
  </script>
</body>
</html>
"""
