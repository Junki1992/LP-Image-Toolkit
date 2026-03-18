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
    "shine": {"label": "シャイン", "desc": "光が走る"},
    "ripple": {"label": "リップル", "desc": "クリックで波紋（JS）"},
    "tilt": {"label": "チルト", "desc": "3Dで傾いて浮く"},
    "magnetic": {"label": "マグネティック", "desc": "カーソルに吸い寄せられる（JS）"},
    "parallax": {"label": "パララックス", "desc": "マウスで3D傾き（JS）"},
    "spotlight": {"label": "スポットライト", "desc": "カーソルに光が追従（JS）"},
    "particle": {"label": "パーティクル", "desc": "クリックで粒子が飛ぶ（JS）"},
    "cursor_glow": {"label": "カーソルグロー", "desc": "カーソルに光が追従（JS）"},
}

# カーソル操作に依存するためGIF出力不可
CTA_EFFECTS_NO_GIF = frozenset({"magnetic", "parallax", "spotlight", "cursor_glow"})


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
        "magnetic": _apply_pulse_frame,
        "parallax": _apply_pulse_frame,
        "spotlight": _apply_pulse_frame,
        "particle": _apply_pulse_frame,
        "cursor_glow": _apply_pulse_frame,
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
    loop_interval: bool = False,
    loop_pause: float = 0,
) -> None:
    """
    画像にエフェクトを適用して GIF を生成
    duration: 1周期の秒数
    loop_interval: True なら アニメ後に静止
    loop_pause: 静止秒数（loop_interval 時のみ）
    """
    pil = Image.open(image_path)
    pil = _ensure_rgba(pil)
    img = np.array(pil)
    func = _get_effect_func(effect)
    frames = []
    total_duration = duration + loop_pause if loop_interval and loop_pause > 0 else duration
    animate_end_pct = (duration / total_duration) * 100 if loop_interval and loop_pause > 0 else 100
    num_frames = max(8, int(total_duration * fps))
    for i in range(num_frames):
        t = i / num_frames
        if loop_interval and loop_pause > 0 and t >= animate_end_pct / 100:
            frame = func(img, 0)  # 静止
        elif loop_interval and loop_pause > 0:
            t_anim = t / (animate_end_pct / 100)  # 0〜animate_end_pct% を 0〜1 にマップ
            frame = func(img, t_anim)
        else:
            frame = func(img, t)
        frames.append(Image.fromarray(frame))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 * total_duration / num_frames) * num_frames // num_frames or 50,
        loop=loop,
    )


# 生成コード内の画像パス（ユーザーが任意に設定するプレースホルダ）
CODE_IMG_PLACEHOLDER = "YOUR_IMAGE.png"
CODE_IMG_PLACEHOLDER_GIF = "YOUR_IMAGE.gif"

# スピード・強さ・ループのデフォルト（数値で細かく指定）
SPEED_DEFAULT = 1.5
STRENGTH_DEFAULT = 1.0
LOOP_DEFAULT = 0  # 0=常に動く、>0=静止秒数


def generate_code(
    image_path: str,
    effect: str,
    image_filename: Optional[str] = None,
    speed: float = 1.5,
    strength: float = 1.0,
    loop: float = 0,
) -> str:
    """
    指定エフェクトの HTML + CSS ソースコードを生成
    speed: 1周期の秒数（0.3〜5）
    strength: 振幅の倍率（0.2〜2）
    loop: 静止秒数、0=常に動く
    """
    fname = image_filename if image_filename else CODE_IMG_PLACEHOLDER
    params = {"speed": speed, "strength": strength, "loop": loop}
    if effect == "shine":
        return _generate_shine_code(fname, params)
    if effect == "ripple":
        return _generate_ripple_code(fname, params)
    if effect == "magnetic":
        return _generate_magnetic_code(fname, params)
    if effect == "parallax":
        return _generate_parallax_code(fname, params)
    if effect == "spotlight":
        return _generate_spotlight_code(fname, params)
    if effect == "particle":
        return _generate_particle_code(fname, params)
    if effect == "cursor_glow":
        return _generate_cursor_glow_code(fname, params)
    effect_props, effect_keyframes = _get_effect_css(effect, params)
    return f"""<!-- 画像パス（YOUR_IMAGE.png）を任意のパスに変更してください -->
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
<div class="cta-wrap">
  <img src="{fname}" alt="CTA">
</div>"""


def _get_effect_css(effect: str, params: dict):
    """エフェクトごとの CSS。(animation プロパティ, @keyframes ブロック)"""
    speed = float(params.get("speed", SPEED_DEFAULT))
    strength = float(params.get("strength", STRENGTH_DEFAULT))
    loop = float(params.get("loop", LOOP_DEFAULT))
    mult = max(0.2, min(2.0, strength))
    speed = max(0.3, min(5.0, speed))
    loop = max(0, min(10, loop))
    dur = f"{speed}s"
    use_interval = loop > 0
    if use_interval:
        total_sec = speed + loop
        animate_end_pct = (speed / total_sec) * 100
        loop_dur = f"{total_sec}s"
    else:
        animate_end_pct = 100
        loop_dur = dur

    def _amp(v: float) -> float:
        """scale 系: 1から離れる量を乗算"""
        return 1 + (v - 1) * mult if v != 1 else 1

    def _px(v: float) -> str:
        return f"{int(v * mult)}px"

    def _deg(v: float) -> str:
        return f"{v * mult:.1f}deg"

    # 数値（振幅）
    s_pulse = _amp(1.05)
    b_pulse = _amp(1.1)
    glow_peak = 4 + int(8 * mult)
    bounce_px = _px(8)
    shake_px = _px(4)
    float_px = _px(6)
    wiggle_deg = _deg(2)
    fade_op = 1 - 0.25 * mult
    rotate_deg = _deg(5)
    swing_deg = _deg(8)
    hb_p1 = _amp(1.08)
    hb_p2 = _amp(1.04)
    rubber_p1 = _amp(1.08)
    rubber_p2 = 1 / rubber_p1 if rubber_p1 != 1 else 1
    breathe_lo = _amp(0.97)
    breathe_hi = _amp(1.03)
    att_p = _amp(1.12)
    tilt_deg = _deg(5)
    tilt_px = _px(8)

    if use_interval:
        end = animate_end_pct
        mid = end / 2
        end4 = end / 4
        end34 = end * 3 / 4
        end14 = end * 14 / 100
        end28 = end * 28 / 100
        end42 = end * 42 / 100
        end56 = end * 56 / 100
        end10 = end * 10 / 100
        end20 = end * 20 / 100
        end30 = end * 30 / 100
        end65 = end * 65 / 100
        data = {
            "pulse": (f"animation: cta-pulse {loop_dur} ease-in-out infinite;",
                      f"@keyframes cta-pulse {{\n      0% {{ transform: scale(1); filter: brightness(1); }}\n      {mid}% {{ transform: scale({s_pulse}); filter: brightness({b_pulse}); }}\n      {end}%, 100% {{ transform: scale(1); filter: brightness(1); }}\n    }}"),
            "glow": (f"animation: cta-glow {loop_dur} ease-in-out infinite;",
                     f"@keyframes cta-glow {{\n      0% {{ filter: drop-shadow(0 0 4px rgba(255,255,255,0.5)); }}\n      {mid}% {{ filter: drop-shadow(0 0 {glow_peak}px rgba(255,255,255,0.9)); }}\n      {end}%, 100% {{ filter: drop-shadow(0 0 4px rgba(255,255,255,0.5)); }}\n    }}"),
            "bounce": (f"animation: cta-bounce {loop_dur} ease-in-out infinite;",
                      f"@keyframes cta-bounce {{\n      0% {{ transform: translateY(0); }}\n      {mid}% {{ transform: translateY(-{bounce_px}); }}\n      {end}%, 100% {{ transform: translateY(0); }}\n    }}"),
            "shake": (f"animation: cta-shake {loop_dur} ease-in-out infinite;",
                     f"@keyframes cta-shake {{\n      0% {{ transform: translateX(0); }}\n      {end4:.1f}% {{ transform: translateX(-{shake_px}); }}\n      {end34:.1f}% {{ transform: translateX({shake_px}); }}\n      {end:.1f}%, 100% {{ transform: translateX(0); }}\n    }}"),
            "float": (f"animation: cta-float {loop_dur} ease-in-out infinite;",
                     f"@keyframes cta-float {{\n      0% {{ transform: translateY(0); }}\n      {mid}% {{ transform: translateY(-{float_px}); }}\n      {end}%, 100% {{ transform: translateY(0); }}\n    }}"),
            "wiggle": (f"animation: cta-wiggle {loop_dur} ease-in-out infinite;",
                      f"@keyframes cta-wiggle {{\n      0% {{ transform: rotate(-{wiggle_deg}); }}\n      {mid}% {{ transform: rotate({wiggle_deg}); }}\n      {end}%, 100% {{ transform: rotate(-{wiggle_deg}); }}\n    }}"),
            "fade": (f"animation: cta-fade {loop_dur} ease-in-out infinite;",
                    f"@keyframes cta-fade {{\n      0% {{ opacity: 1; }}\n      {mid}% {{ opacity: {fade_op}; }}\n      {end}%, 100% {{ opacity: 1; }}\n    }}"),
            "rotate": (f"animation: cta-rotate {loop_dur} ease-in-out infinite;",
                      f"@keyframes cta-rotate {{\n      0% {{ transform: rotate(-{rotate_deg}); }}\n      {mid}% {{ transform: rotate({rotate_deg}); }}\n      {end}%, 100% {{ transform: rotate(-{rotate_deg}); }}\n    }}"),
            "swing": (f"animation: cta-swing {loop_dur} ease-in-out infinite;",
                     f"@keyframes cta-swing {{\n      0% {{ transform: rotate(-{swing_deg}); }}\n      {mid}% {{ transform: rotate({swing_deg}); }}\n      {end}%, 100% {{ transform: rotate(-{swing_deg}); }}\n    }}"),
            "heartbeat": (f"animation: cta-heartbeat {loop_dur} ease-out infinite;",
                         f"@keyframes cta-heartbeat {{\n      0% {{ transform: scale(1); }}\n      {end14:.1f}% {{ transform: scale({hb_p1}); }}\n      {end28:.1f}% {{ transform: scale(1); }}\n      {end42:.1f}% {{ transform: scale({hb_p2}); }}\n      {end:.1f}%, 100% {{ transform: scale(1); }}\n    }}"),
            "rubber": (f"animation: cta-rubber {loop_dur} ease-in-out infinite;",
                      f"@keyframes cta-rubber {{\n      0% {{ transform: scaleX(1) scaleY(1); }}\n      {end30:.1f}% {{ transform: scaleX({rubber_p1}) scaleY({1/rubber_p1:.2f}); }}\n      {end65:.1f}% {{ transform: scaleX({1/rubber_p1:.2f}) scaleY({rubber_p1}); }}\n      {end:.1f}%, 100% {{ transform: scaleX(1) scaleY(1); }}\n    }}"),
            "breathe": (f"animation: cta-breathe {loop_dur} ease-in-out infinite;",
                       f"@keyframes cta-breathe {{\n      0% {{ transform: scale({breathe_lo}); }}\n      {mid}% {{ transform: scale({breathe_hi}); }}\n      {end}%, 100% {{ transform: scale({breathe_lo}); }}\n    }}"),
            "attention": (f"animation: cta-attention {loop_dur} ease-out infinite;",
                         f"@keyframes cta-attention {{\n      0% {{ transform: scale(1); }}\n      {end10:.1f}% {{ transform: scale({att_p}); }}\n      {end20:.1f}% {{ transform: scale(1); }}\n      {end:.1f}%, 100% {{ transform: scale(1); }}\n    }}"),
            "tilt": (f"animation: cta-tilt {loop_dur} ease-in-out infinite;",
                    f"@keyframes cta-tilt {{\n      0% {{ transform: perspective(400px) rotateX(-{tilt_deg}) translateY(0); }}\n      {mid}% {{ transform: perspective(400px) rotateX(-{tilt_deg}) translateY(-{tilt_px}); }}\n      {end}%, 100% {{ transform: perspective(400px) rotateX(-{tilt_deg}) translateY(0); }}\n    }}"),
        }
    else:
        data = {
            "pulse": (f"animation: cta-pulse {dur} ease-in-out infinite;",
                      f"@keyframes cta-pulse {{\n      0%, 100% {{ transform: scale(1); filter: brightness(1); }}\n      50% {{ transform: scale({s_pulse}); filter: brightness({b_pulse}); }}\n    }}"),
            "glow": (f"animation: cta-glow {dur} ease-in-out infinite;",
                     f"@keyframes cta-glow {{\n      0%, 100% {{ filter: drop-shadow(0 0 4px rgba(255,255,255,0.5)); }}\n      50% {{ filter: drop-shadow(0 0 {glow_peak}px rgba(255,255,255,0.9)); }}\n    }}"),
            "bounce": (f"animation: cta-bounce {dur} ease-in-out infinite;",
                      f"@keyframes cta-bounce {{\n      0%, 100% {{ transform: translateY(0); }}\n      50% {{ transform: translateY(-{bounce_px}); }}\n    }}"),
            "shake": (f"animation: cta-shake {dur} ease-in-out infinite;",
                     f"@keyframes cta-shake {{\n      0%, 100% {{ transform: translateX(0); }}\n      25% {{ transform: translateX(-{shake_px}); }}\n      75% {{ transform: translateX({shake_px}); }}\n    }}"),
            "float": (f"animation: cta-float {dur} ease-in-out infinite;",
                     f"@keyframes cta-float {{\n      0%, 100% {{ transform: translateY(0); }}\n      50% {{ transform: translateY(-{float_px}); }}\n    }}"),
            "wiggle": (f"animation: cta-wiggle {dur} ease-in-out infinite;",
                      f"@keyframes cta-wiggle {{\n      0%, 100% {{ transform: rotate(-{wiggle_deg}); }}\n      50% {{ transform: rotate({wiggle_deg}); }}\n    }}"),
            "fade": (f"animation: cta-fade {dur} ease-in-out infinite;",
                    f"@keyframes cta-fade {{\n      0%, 100% {{ opacity: 1; }}\n      50% {{ opacity: {fade_op}; }}\n    }}"),
            "rotate": (f"animation: cta-rotate {dur} ease-in-out infinite;",
                      f"@keyframes cta-rotate {{\n      0%, 100% {{ transform: rotate(-{rotate_deg}); }}\n      50% {{ transform: rotate({rotate_deg}); }}\n    }}"),
            "swing": (f"animation: cta-swing {dur} ease-in-out infinite;",
                     f"@keyframes cta-swing {{\n      0%, 100% {{ transform: rotate(-{swing_deg}); }}\n      50% {{ transform: rotate({swing_deg}); }}\n    }}"),
            "heartbeat": (f"animation: cta-heartbeat {dur} ease-out infinite;",
                         f"@keyframes cta-heartbeat {{\n      0%, 100% {{ transform: scale(1); }}\n      14% {{ transform: scale({hb_p1}); }}\n      28% {{ transform: scale(1); }}\n      42% {{ transform: scale({hb_p2}); }}\n      56% {{ transform: scale(1); }}\n    }}"),
            "rubber": (f"animation: cta-rubber {dur} ease-in-out infinite;",
                      f"@keyframes cta-rubber {{\n      0%, 100% {{ transform: scaleX(1) scaleY(1); }}\n      30% {{ transform: scaleX({rubber_p1}) scaleY({1/rubber_p1:.2f}); }}\n      65% {{ transform: scaleX({1/rubber_p1:.2f}) scaleY({rubber_p1}); }}\n    }}"),
            "breathe": (f"animation: cta-breathe {dur} ease-in-out infinite;",
                       f"@keyframes cta-breathe {{\n      0%, 100% {{ transform: scale({breathe_lo}); }}\n      50% {{ transform: scale({breathe_hi}); }}\n    }}"),
            "attention": (f"animation: cta-attention {dur} ease-out infinite;",
                         f"@keyframes cta-attention {{\n      0%, 100% {{ transform: scale(1); }}\n      10% {{ transform: scale({att_p}); }}\n      20% {{ transform: scale(1); }}\n    }}"),
            "tilt": (f"animation: cta-tilt {dur} ease-in-out infinite;",
                    f"@keyframes cta-tilt {{\n      0%, 100% {{ transform: perspective(400px) rotateX(-{tilt_deg}) translateY(0); }}\n      50% {{ transform: perspective(400px) rotateX(-{tilt_deg}) translateY(-{tilt_px}); }}\n    }}"),
        }
    return data.get(effect, ("", ""))


def _generate_shine_code(fname: str, params: dict) -> str:
    """シャイン: 光が走る（CSS 疑似要素）"""
    speed = max(0.3, min(5.0, float(params.get("speed", SPEED_DEFAULT))))
    loop = max(0, min(10, float(params.get("loop", LOOP_DEFAULT))))
    mult = max(0.2, min(2.0, float(params.get("strength", STRENGTH_DEFAULT))))
    opacity = min(0.9, 0.2 + 0.4 * mult)
    if loop > 0:
        total_sec = speed + loop
        animate_end_pct = (speed / total_sec) * 100
        dur = f"{total_sec}s"
        mid = animate_end_pct / 2
        kf = f"  0% {{ left: -100%; }}\n      {mid:.1f}% {{ left: 150%; }}\n      {animate_end_pct:.1f}%, 100% {{ left: -100%; }}"
    else:
        dur = f"{speed}s"
        kf = "  0% {{ left: -100%; }}\n      50% {{ left: 150%; }}\n      100% {{ left: 150%; }}"
    return f"""<!-- 画像パス（{fname}）を任意のパスに変更してください -->
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
  background: linear-gradient(90deg, transparent, rgba(255,255,255,{opacity}), transparent);
  animation: cta-shine {dur} ease-in-out infinite;
}}
@keyframes cta-shine {{
{kf}
}}
.cta-wrap img {{
  display: block;
  max-width: 100%;
  height: auto;
}}
</style>
<div class="cta-wrap">
  <img src="{fname}" alt="CTA">
</div>"""


def _generate_ripple_code(fname: str, params: dict) -> str:
    """リップル: クリックで波紋（JS）"""
    return f"""<!-- 画像パス（{fname}）を任意のパスに変更してください -->
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
<div class="cta-wrap" data-cta-ripple>
  <img src="{fname}" alt="CTA">
</div>
<script>
(function() {{
  document.querySelectorAll('[data-cta-ripple]').forEach(function(wrap) {{
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
    setTimeout(function() {{ ripple.remove(); }}, 600);
  }});
  }});
}})();
</script>"""


def _generate_magnetic_code(fname: str, params: dict) -> str:
    """マグネティック: カーソルに吸い寄せられる"""
    mult = max(0.2, min(2.0, float(params.get("strength", STRENGTH_DEFAULT))))
    strength_val = int(20 * mult)
    return f"""<!-- 画像パス（{fname}）を任意のパスに変更してください -->
<style>
.cta-wrap {{
  display: inline-block;
  position: relative;
  cursor: pointer;
  transition: transform 0.15s ease-out;
}}
.cta-wrap img {{
  display: block;
  max-width: 100%;
  height: auto;
}}
</style>
<div class="cta-wrap" data-cta-magnetic>
  <img src="{fname}" alt="CTA">
</div>
<script>
(function() {{
  document.querySelectorAll('[data-cta-magnetic]').forEach(function(wrap) {{
    var strength = {strength_val};
    wrap.addEventListener('mousemove', function(e) {{
      var rect = wrap.getBoundingClientRect();
      var cx = rect.left + rect.width / 2;
      var cy = rect.top + rect.height / 2;
      var dx = (e.clientX - cx) / rect.width * strength;
      var dy = (e.clientY - cy) / rect.height * strength;
      wrap.style.transform = 'translate(' + dx + 'px, ' + dy + 'px)';
    }});
    wrap.addEventListener('mouseleave', function() {{
      wrap.style.transform = 'translate(0, 0)';
    }});
  }});
}})();
</script>"""


def _generate_parallax_code(fname: str, params: dict) -> str:
    """パララックス: マウスで3D傾き"""
    mult = max(0.2, min(2.0, float(params.get("strength", STRENGTH_DEFAULT))))
    max_rotate = int(12 * mult)
    return f"""<!-- 画像パス（{fname}）を任意のパスに変更してください -->
<style>
.cta-wrap {{
  display: inline-block;
  position: relative;
  cursor: pointer;
  transform-style: preserve-3d;
  perspective: 1000px;
  transition: transform 0.1s ease-out;
}}
.cta-wrap img {{
  display: block;
  max-width: 100%;
  height: auto;
}}
</style>
<div class="cta-wrap" data-cta-parallax>
  <img src="{fname}" alt="CTA">
</div>
<script>
(function() {{
  document.querySelectorAll('[data-cta-parallax]').forEach(function(wrap) {{
    var maxRotate = {max_rotate};
    wrap.addEventListener('mousemove', function(e) {{
      var rect = wrap.getBoundingClientRect();
      var x = (e.clientX - rect.left) / rect.width - 0.5;
      var y = (e.clientY - rect.top) / rect.height - 0.5;
      var rotateY = x * maxRotate;
      var rotateX = -y * maxRotate;
      wrap.style.transform = 'perspective(1000px) rotateX(' + rotateX + 'deg) rotateY(' + rotateY + 'deg)';
    }});
    wrap.addEventListener('mouseleave', function() {{
      wrap.style.transform = 'perspective(1000px) rotateX(0) rotateY(0)';
    }});
  }});
}})();
</script>"""


def _generate_spotlight_code(fname: str, params: dict) -> str:
    """スポットライト: カーソルに光が追従"""
    mult = max(0.2, min(2.0, float(params.get("strength", STRENGTH_DEFAULT))))
    size = int(150 * mult)
    return f"""<!-- 画像パス（{fname}）を任意のパスに変更してください -->
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
.cta-spotlight {{
  position: absolute;
  width: {size}px;
  height: {size}px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,0.5) 0%, transparent 70%);
  pointer-events: none;
  transform: translate(-50%, -50%);
}}
</style>
<div class="cta-wrap" data-cta-spotlight>
  <img src="{fname}" alt="CTA">
  <span class="cta-spotlight"></span>
</div>
<script>
(function() {{
  document.querySelectorAll('[data-cta-spotlight]').forEach(function(wrap) {{
    var el = wrap.querySelector('.cta-spotlight');
    if (!el) return;
    wrap.addEventListener('mousemove', function(e) {{
      var rect = wrap.getBoundingClientRect();
      el.style.left = (e.clientX - rect.left) + 'px';
      el.style.top = (e.clientY - rect.top) + 'px';
    }});
    wrap.addEventListener('mouseleave', function() {{
      el.style.left = '-999px';
      el.style.top = '-999px';
    }});
  }});
}})();
</script>"""


def _generate_particle_code(fname: str, params: dict) -> str:
    """パーティクル: クリックで粒子が飛ぶ"""
    mult = max(0.2, min(2.0, float(params.get("strength", STRENGTH_DEFAULT))))
    count = max(6, min(24, int(12 * mult)))
    dist_base = int(60 * mult)
    return f"""<!-- 画像パス（{fname}）を任意のパスに変更してください -->
<style>
.cta-wrap {{
  display: inline-block;
  position: relative;
  overflow: visible;
  cursor: pointer;
}}
.cta-wrap img {{
  display: block;
  max-width: 100%;
  height: auto;
}}
.cta-particle {{
  position: absolute;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: rgba(255,255,255,0.9);
  pointer-events: none;
  animation: cta-particle-fly 0.6s ease-out forwards;
}}
@keyframes cta-particle-fly {{
  to {{ transform: translate(var(--tx), var(--ty)) scale(0); opacity: 0; }}
}}
</style>
<div class="cta-wrap" data-cta-particle>
  <img src="{fname}" alt="CTA">
</div>
<script>
(function() {{
  document.querySelectorAll('[data-cta-particle]').forEach(function(wrap) {{
    wrap.addEventListener('click', function(e) {{
      var rect = wrap.getBoundingClientRect();
      var count = {count};
      for (var i = 0; i < count; i++) {{
        var angle = (i / count) * Math.PI * 2;
        var dist = {dist_base} + Math.random() * 40;
        var tx = Math.cos(angle) * dist;
        var ty = Math.sin(angle) * dist;
        var p = document.createElement('span');
        p.className = 'cta-particle';
        p.style.left = (e.clientX - rect.left) + 'px';
        p.style.top = (e.clientY - rect.top) + 'px';
        p.style.setProperty('--tx', tx + 'px');
        p.style.setProperty('--ty', ty + 'px');
        wrap.appendChild(p);
        setTimeout(function() {{ p.remove(); }}, 600);
      }}
    }});
  }});
}})();
</script>"""


def _generate_cursor_glow_code(fname: str, params: dict) -> str:
    """カーソルグロー: カーソルに光が追従"""
    mult = max(0.2, min(2.0, float(params.get("strength", STRENGTH_DEFAULT))))
    size = int(120 * mult)
    return f"""<!-- 画像パス（{fname}）を任意のパスに変更してください -->
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
.cta-cursor-glow {{
  position: absolute;
  width: {size}px;
  height: {size}px;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0.1) 40%, transparent 70%);
  pointer-events: none;
  transform: translate(-50%, -50%);
  filter: blur(8px);
}}
</style>
<div class="cta-wrap" data-cta-cursor-glow>
  <img src="{fname}" alt="CTA">
  <span class="cta-cursor-glow"></span>
</div>
<script>
(function() {{
  document.querySelectorAll('[data-cta-cursor-glow]').forEach(function(wrap) {{
    var el = wrap.querySelector('.cta-cursor-glow');
    if (!el) return;
    wrap.addEventListener('mousemove', function(e) {{
      var rect = wrap.getBoundingClientRect();
      el.style.left = (e.clientX - rect.left) + 'px';
      el.style.top = (e.clientY - rect.top) + 'px';
    }});
    wrap.addEventListener('mouseleave', function() {{
      el.style.left = '-999px';
      el.style.top = '-999px';
    }});
  }});
}})();
</script>"""
