"""
画像内テキストの検出・置換（OCR + オーバーレイ）
PaddleOCR（優先）または EasyOCR でテキストを検出。PaddleOCR は日本語精度が高い。
"""
import difflib
import logging
import os
import tempfile
import cv2

logger = logging.getLogger(__name__)
import numpy as np
from pathlib import Path


_paddle_ocr = None
_easyocr_reader = None

# フォント候補（太字・バナー向け、優先順）
FONT_PATHS = [
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",  # macOS 太字
    "/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc",
    "/System/Library/Fonts/ヒラギノ丸ゴ ProN W6.ttc",  # macOS 丸ゴ太字
    "/System/Library/Fonts/ヒラギノ丸ゴ ProN W4.ttc",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def _preprocess_for_ocr(img, min_side_target=800):
    """
    小さい画像を拡大して OCR 精度を向上。拡大のみ（CLAHE は使わない）。
    最短辺が小さいほど拡大を強めに（文字が大きく見えるほど認識精度が上がる）。
    Returns: (preprocessed_img, scale_factor)
    """
    h, w = img.shape[:2]
    min_side = min(h, w)
    if min_side < 400:
        min_side_target = max(min_side_target, 1600)  # 非常に小さい画像は強めに拡大
    elif min_side < 800:
        min_side_target = max(min_side_target, 1200)  # 小さい画像も拡大
    if min_side >= min_side_target:
        return img, 1.0
    scale = min_side_target / min_side
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return resized, scale


def _gentle_contrast_enhance(img):
    """
    装飾文字向けの軽いコントラスト補正。CLAHE は使わず、L チャンネルの正規化のみ。
    小さい（トリミング）画像で効果を発揮。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_min, l_max = l.min(), l.max()
    if l_max - l_min > 20:
        l = np.clip((l.astype(np.float32) - l_min) * 255.0 / (l_max - l_min), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _enhance_for_ocr(img):
    """
    装飾文字・バナー向けに軽くコントラストを強調。
    強すぎると逆効果になるため控えめに。
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _box_center_and_size(pts):
    """bbox の中心と幅・高さを返す"""
    pts = np.array(pts)
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    w, h = x_max - x_min, y_max - y_min
    return (cx, cy), w, h


def _merge_adjacent_boxes(results):
    """
    隣接するテキストボックスを結合（「初回」+「限定」→「初回限定」）。
    同一行で水平距離が近いものを結合する。
    """
    if len(results) < 2:
        return results
    # 上から下、左から右でソート
    def sort_key(item):
        (pts, _, _) = item
        pts = np.array(pts)
        cy = (pts[:, 1].min() + pts[:, 1].max()) / 2
        cx = (pts[:, 0].min() + pts[:, 0].max()) / 2
        return (cy, cx)
    sorted_results = sorted(results, key=sort_key)
    merged = []
    i = 0
    while i < len(sorted_results):
        bbox, text, conf = sorted_results[i]
        pts = np.array(bbox)
        combined_pts = pts.copy()
        combined_text = str(text).strip()
        combined_conf = conf
        j = i + 1
        while j < len(sorted_results):
            nbbox, ntext, nconf = sorted_results[j]
            npts = np.array(nbbox)
            (cx1, cy1), w1, h1 = _box_center_and_size(pts)
            (cx2, cy2), w2, h2 = _box_center_and_size(npts)
            # 水平隣接：同一行で右端と次の左端が近い
            right = pts[:, 0].max()
            left_next = npts[:, 0].min()
            h_gap = left_next - right
            y_ok = abs(cy1 - cy2) < max(h1, h2) * 0.6
            x_ok = h_gap < max(w1, w2) * 0.8
            # 垂直隣接：縦書き（初回/限定など）で下のボックスが近い
            bottom = pts[:, 1].max()
            top_next = npts[:, 1].min()
            v_gap = top_next - bottom
            v_ok = abs(cx1 - cx2) < max(w1, w2) * 0.6
            v_adjacent = v_gap < max(h1, h2) * 0.8 and v_ok
            if (y_ok and x_ok) or v_adjacent:
                combined_pts = np.vstack([combined_pts, npts])
                combined_text += str(ntext).strip()
                combined_conf = (combined_conf + nconf) / 2
                pts = combined_pts
                j += 1
            else:
                break
        # 結合後の bbox は外接矩形
        x_min, y_min = combined_pts[:, 0].min(), combined_pts[:, 1].min()
        x_max, y_max = combined_pts[:, 0].max(), combined_pts[:, 1].max()
        merged_bbox = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        merged.append((merged_bbox, combined_text, combined_conf))
        i = j
    return merged


def _filter_noise(results, strict=True):
    """
    ノイズを除外：1文字かつ信頼度が低い検出を削除。
    strict=False（トリミング時）: 閾値を 0.80 に緩め、結合で補完できる文字を残す
    """
    thresh = 0.80 if not strict else 0.92
    filtered = []
    for bbox, text, conf in results:
        t = str(text).strip()
        if len(t) == 1 and conf < thresh:
            continue
        if len(t) == 0:
            continue
        filtered.append((bbox, text, conf))
    return filtered


def _postprocess_ocr_results(results, merge_adjacent=False):
    """
    OCR 結果の後処理。
    merge_adjacent=True（トリミング画像）: ノイズ閾値緩め＋隣接ボックス結合で「AmazonP」+「ay」→「AmazonPay」を補完
    merge_adjacent=False: ノイズ除外のみ（全体画像では誤結合を避ける）
    """
    filtered = _filter_noise(results, strict=not merge_adjacent)
    if merge_adjacent and len(filtered) > 1:
        return _merge_adjacent_boxes(filtered)
    return filtered


def _ocr_readtext(input_path, force_engine=None):
    """
    OCR でテキストを検出。PaddleOCR を優先（日本語精度が高い）、なければ EasyOCR。
    force_engine="easyocr" のときは EasyOCR のみ使用。
    Returns: [(bbox, text, conf), ...], engine_name
    """
    img = cv2.imread(str(input_path))
    is_small = img is not None and min(img.shape[:2]) < 400

    # EasyOCR のみ使用（範囲選択時の二重検出用）
    if force_engine == "easyocr":
        if img is None:
            reader = _get_easyocr_reader()
            return reader.readtext(str(input_path)), "easyocr"
        preprocessed, scale = _preprocess_for_ocr(img, min_side_target=800)
        reader = _get_easyocr_reader()
        results = reader.readtext(preprocessed, width_ths=0.7)
        if scale != 1.0:
            results = [
                (np.array(bbox) / scale, text, conf)
                for (bbox, text, conf) in results
            ]
        return _postprocess_ocr_results(results, merge_adjacent=is_small), "easyocr"

    # PaddleOCR を試行（小さい画像は拡大＋軽いコントラスト補正）
    try:
        ocr = _get_paddle_ocr()
        if ocr is not None:
            if img is not None:
                preprocessed, scale = _preprocess_for_ocr(img, min_side_target=800)
                # コントラスト補正は装飾バナーで逆効果になる場合があるため無効化
                # if is_small:
                #     preprocessed = _gentle_contrast_enhance(preprocessed)
                result = ocr.predict(preprocessed)
            else:
                result = ocr.predict(str(input_path))
                scale = 1.0
            if result is not None and len(result) > 0:
                res = result[0]
                # PaddleOCR 3.x: OCRResult は rec_texts, rec_scores, rec_polys を持つ
                try:
                    texts = res["rec_texts"]
                    scores = res["rec_scores"]
                    polys = res["rec_polys"]
                    if texts and len(texts) == len(scores) == len(polys):
                        lines = [
                            (np.array(poly) / scale, str(txt) if not isinstance(txt, tuple) else str(txt[0]), float(score))
                            for poly, txt, score in zip(polys, texts, scores)
                        ]
                        return _postprocess_ocr_results(lines, merge_adjacent=is_small), "paddleocr"
                except (KeyError, TypeError):
                    pass
                # PaddleOCR 2.x 互換: [[[bbox], (text, conf)], ...]
                if result[0]:
                    lines = [(np.array(line[0]) / scale, line[1][0], float(line[1][1])) for line in result[0]]
                    return _postprocess_ocr_results(lines, merge_adjacent=is_small), "paddleocr"
    except Exception as e:
        logger.warning("PaddleOCR が使用できません（EasyOCR にフォールバック）: %s", e)

    # EasyOCR にフォールバック（拡大＋小さい画像は軽いコントラスト補正）
    if img is None:
        reader = _get_easyocr_reader()
        return reader.readtext(str(input_path)), "easyocr"
    preprocessed, scale = _preprocess_for_ocr(img, min_side_target=800)
    # if is_small:
    #     preprocessed = _gentle_contrast_enhance(preprocessed)
    reader = _get_easyocr_reader()
    results = reader.readtext(preprocessed, width_ths=0.7)
    # bbox を元画像の座標系に戻す
    if scale != 1.0:
        results = [
            (np.array(bbox) / scale, text, conf)
            for (bbox, text, conf) in results
        ]
    return _postprocess_ocr_results(results, merge_adjacent=is_small), "easyocr"


def _get_paddle_ocr():
    """PaddleOCR を遅延初期化。mobile モデルで高速化"""
    global _paddle_ocr
    if _paddle_ocr is not None:
        return _paddle_ocr
    try:
        from paddleocr import PaddleOCR
        # mobile モデル: 検出・認識ともに軽量で高速（server より数倍速い）
        _paddle_ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            device="cpu",
        )
        return _paddle_ocr
    except ImportError:
        return None
    except Exception as e:
        logger.warning("PaddleOCR の初期化に失敗: %s", e)
        return None


def _get_easyocr_reader():
    """EasyOCR Reader を遅延初期化"""
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader
    try:
        import easyocr
        _easyocr_reader = easyocr.Reader(["ja", "en"], gpu=False, verbose=False)
        return _easyocr_reader
    except ImportError:
        raise ImportError("OCR が必要です: pip install paddleocr または pip install easyocr")


def _crop_image(img, x_pct, y_pct, w_pct, h_pct):
    """割合指定で画像を切り抜く。x_pct,y_pct: 左上(%), w_pct,h_pct: 幅・高さ(%)"""
    h, w = img.shape[:2]
    x = int(w * x_pct / 100)
    y = int(h * y_pct / 100)
    cw = int(w * w_pct / 100)
    ch = int(h * h_pct / 100)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    cw = max(1, min(cw, w - x))
    ch = max(1, min(ch, h - y))
    return img[y : y + ch, x : x + cw]


def _merge_ocr_results(texts_a, texts_b):
    """
    2つの OCR 結果を統合。一方が他方の部分文字列の場合は長い方を採用。
    「AmazonP」と「クレジットカード・AmazonPayで」→ 長い方のみ返す。
    """
    # (text, conf) のリスト。重複は conf の高い方を保持
    seen = {}
    for t, c in texts_a + texts_b:
        t = str(t).strip()
        if not t:
            continue
        if t not in seen or seen[t] < c:
            seen[t] = c
    # 部分文字列を除外: A が B の部分文字列なら A を削除
    keys = sorted(seen.keys(), key=len, reverse=True)
    kept = []
    for k in keys:
        is_sub = any(k != other and k in other for other in kept)
        if not is_sub:
            kept.append(k)
    return [{"text": k, "confidence": seen[k]} for k in kept]


def detect_text(input_path, crop=None):
    """
    画像内のテキストを検出する（置換せず検出のみ）

    Args:
        input_path: 入力画像パス
        crop: 省略時は全体。指定時は (x_pct, y_pct, w_pct, h_pct) でトリミングしてから検出

    Returns:
        tuple: (texts, engine) - texts: [{"text": str, "confidence": float}, ...], engine: "paddleocr"|"easyocr"|"paddleocr+easyocr"
    """
    path_to_use = input_path
    tmp_path = None
    if crop and len(crop) >= 4:
        img = cv2.imread(str(input_path))
        if img is not None:
            cropped = _crop_image(img, float(crop[0]), float(crop[1]), float(crop[2]), float(crop[3]))
            fd, tmp_path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            cv2.imwrite(tmp_path, cropped)
            path_to_use = tmp_path
    try:
        results, engine = _ocr_readtext(path_to_use)
        texts = [{"text": str(text).strip(), "confidence": float(conf)} for (_, text, conf) in results]

        # 範囲選択時: PaddleOCR と EasyOCR の両方で検出し、長い方を優先して統合
        if crop and len(crop) >= 4:
            try:
                results_easy, _ = _ocr_readtext(path_to_use, force_engine="easyocr")
                texts_easy = [(str(t).strip(), float(c)) for (_, t, c) in results_easy if str(t).strip()]
                texts_paddle = [(r["text"], r["confidence"]) for r in texts]
                merged = _merge_ocr_results(texts_paddle, texts_easy)
                if merged:
                    texts = merged
                    engine = "paddleocr+easyocr"
            except Exception as e:
                logger.debug("範囲選択時の EasyOCR 補完をスキップ: %s", e)

        return texts, engine
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _text_color_in_region(img_bgr, pts):
    """
    テキスト領域から文字色を取得。背景色を推定し、背景と最も異なる色＝文字色とする。
    白文字が薄緑になるのを防ぐため、最も純粋な文字色のみを採用。BGR → RGB
    """
    h, w = img_bgr.shape[:2]
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    pad = max(2, int(min(x_max - x_min, y_max - y_min) * 0.15))

    # 背景: bbox の外側（パディング領域）からサンプル
    y_top = max(0, y_min - pad)
    y_bot = min(h, y_max + pad)
    x_left = max(0, x_min - pad)
    x_right = min(w, x_max + pad)
    border_pixels = []
    if y_top < y_min:
        border_pixels.append(img_bgr[y_top:y_min, x_left:x_right].reshape(-1, 3))
    if y_max < y_bot:
        border_pixels.append(img_bgr[y_max:y_bot, x_left:x_right].reshape(-1, 3))
    if x_left < x_min:
        border_pixels.append(img_bgr[y_min:y_max, x_left:x_min].reshape(-1, 3))
    if x_max < x_right:
        border_pixels.append(img_bgr[y_min:y_max, x_max:x_right].reshape(-1, 3))
    bg_pixels = np.vstack([p for p in border_pixels if len(p) > 0]).astype(np.float32) if border_pixels else np.array([])
    if len(bg_pixels) == 0:
        bg_pixels = img_bgr[y_min:y_max, x_min:x_max].reshape(-1, 3).astype(np.float32)
    bg_color = np.median(bg_pixels, axis=0)

    # テキスト領域内のピクセルで、背景から最も離れているもの＝文字色
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    text_region = img_bgr[mask > 0].astype(np.float32)
    if len(text_region) == 0:
        return (255, 255, 255)
    dist_from_bg = np.sqrt(np.sum((text_region - bg_color) ** 2, axis=1))

    # 上位5〜10%のみ採用（アンチエイリアスの薄緑を除外し、純粋な白・金などを取得）
    n = max(1, min(len(text_region) // 10, int(len(text_region) * 0.08)))
    idx = np.argsort(dist_from_bg)[-n:]
    text_color = np.median(text_region[idx], axis=0).astype(np.uint8)
    r, g, b = int(text_color[2]), int(text_color[1]), int(text_color[0])

    # 緑背景で抽出色が緑寄り（GがR,Bより明らかに大きい）→ 白文字の誤検出。最明部を使用
    bg_r, bg_g, bg_b = int(bg_color[2]), int(bg_color[1]), int(bg_color[0])
    is_green_bg = bg_g > bg_r + 20 and bg_g > bg_b + 20
    is_greenish_text = g > r + 15 and g > b + 15
    if is_green_bg and is_greenish_text:
        brightness = 0.299 * text_region[:, 2] + 0.587 * text_region[:, 1] + 0.114 * text_region[:, 0]
        bright_n = max(1, int(len(text_region) * 0.1))
        bright_idx = np.argsort(brightness)[-bright_n:]
        bright_median = np.median(text_region[bright_idx], axis=0).astype(np.uint8)
        r, g, b = int(bright_median[2]), int(bright_median[1]), int(bright_median[0])

    # 抽出色が背景に近すぎる場合（距離<80）→ 白文字の可能性が高い
    dist = np.sqrt((r - bg_r) ** 2 + (g - bg_g) ** 2 + (b - bg_b) ** 2)
    if dist < 80 and (r + g + b) > 400:
        r, g, b = 255, 255, 255

    return (r, g, b)


def _bg_color_in_region(img_bgr, pts):
    """テキスト領域の外側から背景色を推定。BGR の tuple を返す"""
    h, w = img_bgr.shape[:2]
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    pad = max(2, int(min(x_max - x_min, y_max - y_min) * 0.15))
    y_top = max(0, y_min - pad)
    y_bot = min(h, y_max + pad)
    x_left = max(0, x_min - pad)
    x_right = min(w, x_max + pad)
    border_pixels = []
    if y_top < y_min:
        border_pixels.append(img_bgr[y_top:y_min, x_left:x_right].reshape(-1, 3))
    if y_max < y_bot:
        border_pixels.append(img_bgr[y_max:y_bot, x_left:x_right].reshape(-1, 3))
    if x_left < x_min:
        border_pixels.append(img_bgr[y_min:y_max, x_left:x_min].reshape(-1, 3))
    if x_max < x_right:
        border_pixels.append(img_bgr[y_min:y_max, x_max:x_right].reshape(-1, 3))
    bg_pixels = np.vstack([p for p in border_pixels if len(p) > 0]).astype(np.float32) if border_pixels else np.array([])
    if len(bg_pixels) == 0:
        bg_pixels = img_bgr[y_min:y_max, x_min:x_max].reshape(-1, 3).astype(np.float32)
    bg = np.median(bg_pixels, axis=0).astype(np.uint8)
    return (int(bg[0]), int(bg[1]), int(bg[2]))


def _outline_color(rgb):
    """
    アウトライン用の色。
    白・明るい文字: 縁も白（柔らかい見た目）
    暗い文字: 縁を明るくして視認性確保
    金・オレンジ系: 暗い縁（茶系）で立体感
    """
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    r, g, b = rgb
    # 白に近い → 縁も白
    if lum > 220 and min(rgb) > 200:
        return rgb
    # 明るい文字（白系）→ 縁も同色
    if lum > 180:
        return rgb
    # 金・オレンジ系（R,G が高く B が低い）→ 暗い茶色の縁
    if r > 180 and g > 150 and b < 120:
        return (max(0, r - 80), max(0, g - 60), max(0, b - 40))
    # 暗い文字 → 縁を明るく
    return tuple(max(0, min(255, int(255 - (255 - c) * 0.5))) for c in rgb)


def _outline_width(font_size, text_color_rgb):
    """フォントサイズと文字色に応じた縁の太さ（1〜3px）"""
    lum = 0.299 * text_color_rgb[0] + 0.587 * text_color_rgb[1] + 0.114 * text_color_rgb[2]
    if lum > 180:
        return 0  # 白・明るい文字は縁なし
    # 暗い文字: フォントサイズの 5〜7% を縁の太さに（最小1、最大3）
    w = max(1, min(3, int(font_size * 0.06)))
    return w


def replace_text(input_path, output_path, old_text, new_text, crop=None, progress_callback=None, use_dual_ocr=False,
                 font_size_override=None, text_color_override=None, bg_color_override=None, position_offset=None,
                 outline_color_override=None, outline_width_override=None):
    """
    画像内のテキストを置換する

    Args:
        input_path: 入力画像パス
        output_path: 出力画像パス
        old_text: 検索・置換する元のテキスト（部分一致で検索）
        new_text: 置換後のテキスト
        crop: 省略時は全体。指定時は (x_pct, y_pct, w_pct, h_pct) でトリミングしてから処理し、結果を貼り戻す
        progress_callback: (step, message) のコールバック（任意）
        use_dual_ocr: True のとき PaddleOCR と EasyOCR の両方で検出し、bbox を統合（範囲選択時の置換精度向上）
        font_size_override: フォントサイズ（px）。None で自動
        text_color_override: (r, g, b) の tuple。None で自動
        bg_color_override: (b, g, r) の tuple（OpenCV BGR）。None で自動
        position_offset: (dx, dy) の tuple。描画位置のオフセット（px）。None で中央
        outline_color_override: (r, g, b) の tuple。None で自動（白文字＋黒縁など手動指定可能）
        outline_width_override: 縁の太さ（px）。None で自動

    Raises:
        ValueError: old_text が見つからない場合
    """
    if crop and len(crop) >= 4:
        img_full = cv2.imread(str(input_path))
        if img_full is None:
            raise ValueError(f"画像を読み込めませんでした: {input_path}")
        cropped = _crop_image(
            img_full, float(crop[0]), float(crop[1]), float(crop[2]), float(crop[3])
        )
        h, w = img_full.shape[:2]
        x = int(w * float(crop[0]) / 100)
        y = int(h * float(crop[1]) / 100)
        cw = int(w * float(crop[2]) / 100)
        ch = int(h * float(crop[3]) / 100)
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        cw = max(1, min(cw, w - x))
        ch = max(1, min(ch, h - y))

        fd_in, tmp_in = tempfile.mkstemp(suffix=".png")
        os.close(fd_in)
        fd_out, tmp_out = tempfile.mkstemp(suffix=".png")
        os.close(fd_out)
        try:
            cv2.imwrite(tmp_in, cropped)
            replace_text(tmp_in, tmp_out, old_text, new_text, crop=None, progress_callback=progress_callback, use_dual_ocr=True,
                        font_size_override=font_size_override, text_color_override=text_color_override, bg_color_override=bg_color_override, position_offset=position_offset,
                        outline_color_override=outline_color_override, outline_width_override=outline_width_override)
            result_region = cv2.imread(tmp_out)
            if result_region is not None:
                rh, rw = result_region.shape[:2]
                if (rh, rw) == (ch, cw):
                    img_full[y : y + ch, x : x + cw] = result_region
                else:
                    resized = cv2.resize(result_region, (cw, ch), interpolation=cv2.INTER_LINEAR)
                    img_full[y : y + ch, x : x + cw] = resized
            ext = Path(output_path).suffix.lower()
            if ext in (".jpg", ".jpeg"):
                cv2.imwrite(str(output_path), img_full, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif ext == ".webp":
                cv2.imwrite(str(output_path), img_full, [cv2.IMWRITE_WEBP_QUALITY, 95])
            else:
                cv2.imwrite(str(output_path), img_full)
            print(f"      保存しました: {output_path}")
        finally:
            for p in (tmp_in, tmp_out):
                if p and os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
        return

    if progress_callback:
        progress_callback(0, "画像を読み込み中...")

    img = cv2.imread(str(input_path))
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {input_path}")

    if progress_callback:
        progress_callback(1, "テキストを検出中...")

    results, _ = _ocr_readtext(input_path)
    if use_dual_ocr:
        try:
            results_easy, _ = _ocr_readtext(input_path, force_engine="easyocr")
            # 短い方が長い方の部分文字列かつ bbox が重なれば短い方を除外（長い方を優先）
            combined = results + results_easy
            combined = [(np.array(pts), str(txt).strip(), conf) for (pts, txt, conf) in combined if str(txt).strip()]

            def _is_redundant(short_pts, short_txt, long_pts, long_txt):
                if short_txt == long_txt or short_txt not in long_txt:
                    return False
                s, l = np.array(short_pts), np.array(long_pts)
                sx1, sy1, sx2, sy2 = s[:, 0].min(), s[:, 1].min(), s[:, 0].max(), s[:, 1].max()
                lx1, ly1, lx2, ly2 = l[:, 0].min(), l[:, 1].min(), l[:, 0].max(), l[:, 1].max()
                overlap = max(0, min(sx2, lx2) - max(sx1, lx1)) * max(0, min(sy2, ly2) - max(sy1, ly1))
                area_s = (sx2 - sx1) * (sy2 - sy1)
                return area_s > 0 and overlap / area_s > 0.5

            # 長いテキストから処理して、短い重複を除外
            combined.sort(key=lambda r: -len(str(r[1])))
            kept = []
            for pts, txt, conf in combined:
                redundant = any(_is_redundant(pts, txt, o[0], str(o[1])) for o in kept if str(o[1]) != txt)
                if not redundant:
                    kept.append((pts, txt, conf))
            results = kept
        except Exception as e:
            logger.debug("置換時の EasyOCR 補完をスキップ: %s", e)

    # old_text に一致する領域を探す（部分一致・大文字小文字無視）
    # 編集後のテキストは OCR 結果と完全一致しないことがあるため、類似度でも検索
    old_lower = old_text.strip().lower()
    old_stripped = old_text.strip()
    matches = []
    for (bbox, text, conf) in results:
        text_clean = str(text).strip()
        if old_lower in text_clean.lower():
            matches.append((bbox, text_clean, conf))

    # 完全一致がなければ類似度で最も近いものを採用（編集ミスや OCR 差を吸収）
    if not matches and old_stripped:
        best_ratio, best_item = 0.0, None
        for (bbox, text, conf) in results:
            text_clean = str(text).strip()
            if not text_clean:
                continue
            ratio = difflib.SequenceMatcher(None, old_lower, text_clean.lower()).ratio()
            if ratio > best_ratio and ratio >= 0.75:
                best_ratio, best_item = ratio, (bbox, text_clean, conf)
        if best_item:
            matches = [best_item]

    if not matches:
        # 編集範囲指定時: OCR がマッチしなくても、選択範囲全体を置換するフォールバック
        if use_dual_ocr and old_stripped:
            h_img, w_img = img.shape[:2]
            full_bbox = np.array([[0, 0], [w_img, 0], [w_img, h_img], [0, h_img]])
            matches = [(full_bbox, old_stripped, 0.0)]
            logger.info("OCR でマッチせず、編集範囲全体を置換します（置換元: %s）", old_stripped[:30])
        else:
            raise ValueError(f"「{old_text}」が見つかりませんでした。OCRで検出されたテキストを確認してください。")

    if progress_callback:
        progress_callback(2, f"{len(matches)}件のテキストを置換中...")

    # BGR → RGB（PIL用）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise ImportError("Pillow が必要です: pip install Pillow")

    default_font = ImageFont.load_default()
    _tmp_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))  # textbbox 用の仮 draw

    h_img, w_img = img.shape[:2]
    for bbox, detected_text, _ in matches:
        pts = np.array(bbox, dtype=np.int32)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        box_w = x_max - x_min
        box_h = y_max - y_min

        # フォールバック（範囲全体置換）かどうか: bbox が画像全体
        is_full_region_fallback = (
            x_min <= 2 and y_min <= 2 and
            x_max >= w_img - 2 and y_max >= h_img - 2
        )

        # 元テキストの文字色を取得（inpaint 前に。背景と最も異なる色＝文字色）
        text_color = text_color_override if text_color_override else _text_color_in_region(img, pts)
        outline_color = outline_color_override if outline_color_override else _outline_color(text_color)

        # フォント選択: ユーザー指定があればそれを使用、なければ自動
        use_font = default_font
        if font_size_override is not None and font_size_override > 0:
            font_size = int(font_size_override)
        elif is_full_region_fallback:
            font_size = min(int(box_h * 0.35), max(18, 80 // max(1, len(new_text))))
        else:
            font_size = max(12, int(box_h * 0.70))
        fit_ratio = 0.70
        if font_size_override is not None and font_size_override > 0:
            for fp in FONT_PATHS:
                if Path(fp).exists():
                    try:
                        use_font = ImageFont.truetype(fp, int(font_size_override))
                        font_size = int(font_size_override)
                        break
                    except Exception:
                        pass
        if use_font == default_font:
            for fp in FONT_PATHS:
                if not Path(fp).exists():
                    continue
                for fs in range(min(font_size, int(box_h * fit_ratio)), 8, -1):
                    try:
                        trial_font = ImageFont.truetype(fp, fs)
                        bbox_draw = _tmp_draw.textbbox((0, 0), new_text, font=trial_font)
                        tw = bbox_draw[2] - bbox_draw[0]
                        th = bbox_draw[3] - bbox_draw[1]
                        if tw <= box_w * fit_ratio and th <= box_h * fit_ratio:
                            use_font = trial_font
                            font_size = fs
                            break
                    except Exception:
                        continue
                if use_font != default_font:
                    break

        if use_font == default_font:
            font_size = max(12, int(box_h * 0.65))
            for fp in FONT_PATHS:
                if Path(fp).exists():
                    try:
                        use_font = ImageFont.truetype(fp, font_size)
                        break
                    except Exception:
                        pass

        if outline_width_override is not None and outline_width_override >= 0:
            outline_w = int(outline_width_override)
        elif outline_color_override is not None:
            # 縁色を手動指定した場合は縁を描画（フォントサイズの約6%、最小1）
            outline_w = max(1, min(6, int(font_size * 0.06)))
        else:
            outline_w = _outline_width(font_size, text_color)

        # 塗りつぶし範囲: フォールバック時はテキスト幅＋余白のみ（範囲全体は塗らない＝他要素を残す）
        bbox_draw_pre = _tmp_draw.textbbox((0, 0), new_text, font=use_font)
        th_actual = bbox_draw_pre[3] - bbox_draw_pre[1]
        tw_actual = bbox_draw_pre[2] - bbox_draw_pre[0]
        fill_h = th_actual + 2 * outline_w + 12
        fill_w = tw_actual + 2 * outline_w + 24
        if is_full_region_fallback:
            # 左寄せで塗りつぶし幅を置換元テキスト長に応じて調整（「クレジットカード」→「クレカ」で「・AmazonPayで」を残す）
            old_len = max(1, len(old_stripped))
            fill_w = int(fill_w * max(1.0, old_len / max(1, len(new_text))))
            fill_w = min(fill_w, w_img - 16)
            cy = (y_min + y_max) // 2
            fill_top = max(0, cy - fill_h // 2)
            fill_bot = min(h_img, fill_top + fill_h)
            fill_left = max(0, 8)
            fill_right = min(w_img, fill_left + fill_w)
            fill_top, fill_left = max(0, fill_top), max(0, fill_left)
            pts_exp = np.array([[fill_left, fill_top], [fill_right, fill_top], [fill_right, fill_bot], [fill_left, fill_bot]], dtype=np.int32)
            y_max = fill_bot
            x_min, y_min, x_max, y_max = fill_left, fill_top, fill_right, fill_bot
        else:
            pad_top = 15 if y_min < 80 else 0
            fill_top = max(0, y_min - pad_top)
            y_max_fill = min(y_max, y_min + fill_h)
            edge_pad = 5
            x1 = max(0, x_min - edge_pad)
            y1 = max(0, fill_top - edge_pad)
            x2 = min(w_img, x_max + edge_pad)
            y2 = min(h_img, y_max_fill + edge_pad)
            pts_exp = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            y_max = y_max_fill

        bg_bgr = _bg_color_in_region(img, pts)
        if bg_color_override and len(bg_color_override) >= 3:
            bg_bgr = (int(bg_color_override[2]), int(bg_color_override[1]), int(bg_color_override[0]))
        else:
            lum = 0.299 * bg_bgr[2] + 0.587 * bg_bgr[1] + 0.114 * bg_bgr[0]
            if lum < 80:
                bg_bgr = tuple(min(255, int(c * 1.5 + 128)) for c in bg_bgr)
        img_filled = img.copy()
        cv2.fillPoly(img_filled, [pts_exp], bg_bgr)
        img_rgb = cv2.cvtColor(img_filled, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        draw = ImageDraw.Draw(pil_img)

        # テキストの描画位置（中央寄せ。アセンダー・縁取りで上部が切れないよう ty を調整）
        bbox_draw = draw.textbbox((0, 0), new_text, font=use_font)
        tw = bbox_draw[2] - bbox_draw[0]
        th = bbox_draw[3] - bbox_draw[1]
        tx = x_min + (x_max - x_min - tw) // 2
        ty = y_min + (y_max - y_min - th) // 2
        if position_offset and len(position_offset) >= 2:
            tx += int(position_offset[0])
            ty += int(position_offset[1])
        # 上: アセンダー・縁取りで切れないよう ty を下げる（テキスト小さいので余白多めに）
        top_extent = ty + bbox_draw[1] - outline_w
        top_pad = 8
        if top_extent < fill_top + top_pad:
            ty += fill_top - top_extent + top_pad
        # 下: y_max をはみ出さない（下のテキストを覆わない）
        bottom_extent = ty + bbox_draw[3] + outline_w
        if bottom_extent > y_max:
            ty -= bottom_extent - y_max

        # 縁取り: PIL の stroke_width/stroke_fill で高品質描画（Pillow 8.0+）
        stroke_fill = outline_color
        try:
            draw.text(
                (tx, ty), new_text, font=use_font, fill=text_color,
                stroke_width=outline_w, stroke_fill=stroke_fill,
            )
        except TypeError:
            # 古い Pillow は stroke 非対応 → 従来の複数描画でフォールバック
            if outline_w > 0:
                for dx in range(-outline_w, outline_w + 1):
                    for dy in range(-outline_w, outline_w + 1):
                        if dx != 0 or dy != 0:
                            draw.text((tx + dx, ty + dy), new_text, font=use_font, fill=stroke_fill)
            draw.text((tx, ty), new_text, font=use_font, fill=text_color)

        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    ext = Path(output_path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ext == ".webp":
        cv2.imwrite(str(output_path), img, [cv2.IMWRITE_WEBP_QUALITY, 95])
    else:
        cv2.imwrite(str(output_path), img)
    print(f"      保存しました: {output_path}")
