"""
ImgCraft — 画像処理 Web アプリ
- アップスケール / 形式変換 / 軽量化 / パイプライン
"""
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from io import BytesIO
from pathlib import Path

from flask import Flask, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename

from upscale import convert, crop, optimize, optimize_video, remove_background, upscale

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB（動画対応）
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
VIDEO_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}
ALLOWED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

ROOT_DIR = Path(__file__).parent


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in VIDEO_EXTENSIONS


def process_one(input_path, output_path, mode, opts):
    """1ファイルの画像または動画を処理。opts は form または dict（.get() を持つ）"""
    if mode == "upscale":
        scale = int(opts.get("scale", 4))
        upscale_mode = opts.get("upscale_mode", "photo")
        upscale(str(input_path), str(output_path), mode=upscale_mode, scale=scale)
        return None
    elif mode == "convert":
        quality = int(opts.get("quality", 95))
        convert(str(input_path), str(output_path), quality=quality)
        return None
    elif mode == "optimize":
        max_w, max_h = opts.get("max_width"), opts.get("max_height")
        max_width = int(max_w) if max_w else None
        max_height = int(max_h) if max_h else None
        input_path = Path(input_path)
        if input_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            return optimize_video(str(input_path), str(output_path), max_width, max_height, crf=18)
        quality = int(opts.get("quality", 85))
        return optimize(str(input_path), str(output_path), max_width, max_height, quality, auto=True)
    elif mode == "removebg":
        remove_background(str(input_path), str(output_path))
        return None
    elif mode == "crop":
        x_pct = float(opts.get("crop_x_pct", 0))
        y_pct = float(opts.get("crop_y_pct", 0))
        w_pct = float(opts.get("crop_w_pct", 100))
        h_pct = float(opts.get("crop_h_pct", 100))
        quality = int(opts.get("quality", 95))
        crop(str(input_path), str(output_path), x_pct, y_pct, w_pct, h_pct, quality)
        return None
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    files = request.files.getlist("file")
    files = [f for f in files if f and f.filename and allowed_file(f.filename)]

    if not files:
        return jsonify({"error": "画像または動画ファイルが選択されていません"}), 400

    mode = request.form.get("mode", "convert")

    # パイプライン処理
    if mode == "pipeline":
        pipeline_steps = request.form.get("pipeline_steps")
        if not pipeline_steps:
            return jsonify({"error": "パイプラインのステップが設定されていません"}), 400
        try:
            steps = json.loads(pipeline_steps)
        except json.JSONDecodeError:
            return jsonify({"error": "パイプラインの形式が不正です"}), 400
        if not steps or not isinstance(steps, list):
            return jsonify({"error": "パイプラインに1つ以上のステップを追加してください"}), 400

        files = [f for f in files if not is_video(f.filename)]
        if not files:
            return jsonify({"error": "パイプラインは画像のみ対応です"}), 400

        ext = request.form.get("output_format", "png")
        if any(s.get("mode") == "removebg" for s in steps):
            ext = "png"  # 背景削除を含む場合は PNG

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            processed = []
            optimize_results = []
            try:
                used_names = set()
                for i, file in enumerate(files):
                    stem = Path(secure_filename(file.filename)).stem
                    input_path = tmpdir / secure_filename(file.filename)
                    file.save(input_path)

                    current_path = input_path
                    for si, step in enumerate(steps):
                        step_mode = step.get("mode")
                        if not step_mode:
                            continue
                        out_ext = ".png" if step_mode == "removebg" else f".{ext}"
                        step_out = tmpdir / f"_step_{i}_{si}{out_ext}"
                        process_one(current_path, step_out, step_mode, step)
                        current_path = step_out

                    # 最終出力形式に変換（必要なら）
                    out_ext = f".{ext}"
                    if current_path.suffix.lower() != out_ext.lower():
                        convert_path = tmpdir / f"_final_{i}{out_ext}"
                        process_one(current_path, convert_path, "convert", {"quality": 95})
                        current_path = convert_path
                    if len(files) > 1:
                        base = f"{stem}{out_ext}"
                        output_name = base
                        j = 1
                        while output_name in used_names:
                            output_name = f"{stem}_{j}{out_ext}"
                            j += 1
                        used_names.add(output_name)
                    else:
                        output_name = f"{stem}{out_ext}"
                    output_path = tmpdir / output_name
                    shutil.copy(current_path, output_path)
                    processed.append((output_name, output_path))

                if len(processed) == 1:
                    out_name, out_path = processed[0]
                    with open(out_path, "rb") as f:
                        data = BytesIO(f.read())
                    download_name = out_name
                    count = 1
                else:
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        for name, path in processed:
                            zf.write(path, name)
                    data = BytesIO(zip_buffer.getvalue())
                    download_name = f"images_{uuid.uuid4().hex[:8]}.zip"
                    count = len(processed)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        response = send_file(data, as_attachment=True, download_name=download_name, mimetype="application/octet-stream")
        response.headers["X-Count"] = str(count)
        return response

    if mode in ("convert", "upscale", "removebg", "crop"):
        files = [f for f in files if not is_video(f.filename)]
        if not files:
            return jsonify({"error": "形式変換・アップスケール・背景削除・トリミングは画像のみ対応です。動画は軽量化モードでどうぞ。"}), 400
    ext = request.form.get("output_format", "png")
    if mode == "removebg":
        ext = "png"  # 背景削除は透過 PNG 固定

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        processed = []
        optimize_results = []

        try:
            used_names = set()
            for i, file in enumerate(files):
                stem = Path(secure_filename(file.filename)).stem
                input_path = tmpdir / secure_filename(file.filename)
                file.save(input_path)

                file_ext = Path(secure_filename(file.filename)).suffix.lower()
                out_ext = ".mp4" if file_ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"} else f".{ext}"
                if len(files) > 1:
                    base = f"{stem}{out_ext}"
                    output_name = base
                    j = 1
                    while output_name in used_names:
                        output_name = f"{stem}_{j}{out_ext}"
                        j += 1
                    used_names.add(output_name)
                else:
                    output_name = f"{stem}{out_ext}"
                output_path = tmpdir / output_name

                result = process_one(input_path, output_path, mode, request.form)
                if result:
                    optimize_results.append(result)

                processed.append((output_name, output_path))

            if len(processed) == 1:
                # 1枚: そのまま返す
                out_name, out_path = processed[0]
                with open(out_path, "rb") as f:
                    data = BytesIO(f.read())
                download_name = out_name
                is_zip = False
                count = 1
            else:
                # 複数枚: ZIPで返す
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name, path in processed:
                        zf.write(path, name)
                data = BytesIO(zip_buffer.getvalue())
                download_name = f"images_{uuid.uuid4().hex[:8]}.zip"
                is_zip = True
                count = len(processed)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    response = send_file(data, as_attachment=True, download_name=download_name, mimetype="application/octet-stream")
    response.headers["X-Count"] = str(count)

    if mode == "optimize" and optimize_results:
        if len(optimize_results) == 1:
            opt = optimize_results[0]
            response.headers["X-Reduction-Percent"] = str(opt["reduction_percent"])
            response.headers["X-Input-Size"] = str(opt["input_size"])
            response.headers["X-Output-Size"] = str(opt["output_size"])
        else:
            total_in = sum(r["input_size"] for r in optimize_results)
            total_out = sum(r["output_size"] for r in optimize_results)
            reduction = round((1 - total_out / total_in) * 100, 1) if total_in else 0
            response.headers["X-Reduction-Percent"] = str(reduction)
            response.headers["X-Input-Size"] = str(total_in)
            response.headers["X-Output-Size"] = str(total_out)

    return response


if __name__ == "__main__":
    os.chdir(ROOT_DIR)  # weights/ の相対パス用
    app.run(debug=True, host="0.0.0.0", port=5001)
