"""
ImgCraft — 画像処理 Web アプリ
- アップスケール / 形式変換 / 軽量化 / パイプライン
"""
import json
import os
import shutil
import tempfile
import threading
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from queue import Queue, Empty

from flask import Flask, Response, render_template, request, send_file, jsonify
from werkzeug.utils import secure_filename

from upscale import convert, crop, optimize, optimize_video, remove_background, upscale

STEP_LABELS = {
    "removebg": "背景削除",
    "crop": "トリミング",
    "upscale": "アップスケール",
    "optimize": "軽量化",
    "convert": "形式変換",
}

# パイプラインジョブ用（job_id -> {queue, result_path, filename, error, tmpdir}）
jobs = {}
jobs_lock = threading.Lock()

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


def process_one(input_path, output_path, mode, opts, progress_callback=None):
    """1ファイルの画像または動画を処理。opts は form または dict（.get() を持つ）"""
    if mode == "upscale":
        scale = int(opts.get("scale", 4))
        upscale_mode = opts.get("upscale_mode", "photo")
        tw = opts.get("upscale_target_width")
        th = opts.get("upscale_target_height")
        target_width = int(tw) if tw else None
        target_height = int(th) if th else None
        upscale(
            str(input_path),
            str(output_path),
            mode=upscale_mode,
            scale=scale,
            target_width=target_width,
            target_height=target_height,
            progress_callback=progress_callback,
        )
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

    # パイプライン処理（進捗ストリーミング）
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

        job_id = uuid.uuid4().hex[:12]
        progress_queue = Queue()
        tmpdir = Path(tempfile.mkdtemp())

        with jobs_lock:
            jobs[job_id] = {
                "queue": progress_queue,
                "result_path": None,
                "filename": None,
                "count": 0,
                "error": None,
                "tmpdir": tmpdir,
            }

        # メインスレッドでファイルを保存（リクエストコンテキストが必要）
        for i, file in enumerate(files):
            input_path = tmpdir / secure_filename(file.filename)
            file.save(input_path)

        def run_pipeline():
            try:
                processed = []
                used_names = set()
                total_files = len(files)
                total_steps = len([s for s in steps if s.get("mode")])

                for i, file in enumerate(files):
                    stem = Path(secure_filename(file.filename)).stem
                    input_path = tmpdir / secure_filename(file.filename)

                    current_path = input_path
                    for si, step in enumerate(steps):
                        step_mode = step.get("mode")
                        if not step_mode:
                            continue
                        step_label = STEP_LABELS.get(step_mode, step_mode)
                        progress_queue.put({
                            "file_index": i,
                            "file_total": total_files,
                            "step_index": si,
                            "step_total": total_steps,
                            "step_name": step_label,
                            "status": "processing",
                        })
                        out_ext = ".png" if step_mode == "removebg" else f".{ext}"
                        step_out = tmpdir / f"_step_{i}_{si}{out_ext}"
                        process_one(current_path, step_out, step_mode, step)
                        current_path = step_out
                        progress_queue.put({
                            "file_index": i,
                            "step_index": si,
                            "step_name": step_label,
                            "status": "done",
                        })

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
                    result_path = tmpdir / "_result"
                    shutil.copy(out_path, result_path)
                    filename = out_name
                    count = 1
                else:
                    zip_path = tmpdir / "_result.zip"
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        for name, path in processed:
                            zf.write(path, name)
                    result_path = zip_path
                    filename = f"images_{uuid.uuid4().hex[:8]}.zip"
                    count = len(processed)

                with jobs_lock:
                    jobs[job_id]["result_path"] = result_path
                    jobs[job_id]["filename"] = filename
                    jobs[job_id]["count"] = count
                progress_queue.put({"done": True})
            except Exception as e:
                with jobs_lock:
                    jobs[job_id]["error"] = str(e)
                progress_queue.put({"done": True, "error": str(e)})

        thread = threading.Thread(target=run_pipeline)
        thread.start()
        return jsonify({"job_id": job_id}), 202

    if mode in ("convert", "upscale", "removebg", "crop"):
        files = [f for f in files if not is_video(f.filename)]
        if not files:
            return jsonify({"error": "形式変換・アップスケール・背景削除・トリミングは画像のみ対応です。動画は軽量化モードでどうぞ。"}), 400
    ext = request.form.get("output_format", "png")
    if mode == "removebg":
        ext = "png"  # 背景削除は透過 PNG 固定

    # アップスケールは進捗ストリーミング対応（ジョブベース）
    if mode == "upscale":
        job_id = uuid.uuid4().hex[:12]
        progress_queue = Queue()
        tmpdir = Path(tempfile.mkdtemp())

        with jobs_lock:
            jobs[job_id] = {
                "queue": progress_queue,
                "result_path": None,
                "filename": None,
                "count": 0,
                "error": None,
                "tmpdir": tmpdir,
            }

        def put_progress(step, msg, extra=None):
            progress_queue.put({"step": step, "msg": msg, "extra": dict(extra or {}, **{"file_index": put_progress._file_index, "file_total": len(files)})})

        put_progress._file_index = 0

        upscale_opts = {
            "scale": int(request.form.get("scale", 4)),
            "upscale_mode": request.form.get("upscale_mode", "photo"),
            "upscale_target_width": request.form.get("upscale_target_width"),
            "upscale_target_height": request.form.get("upscale_target_height"),
        }

        for i, file in enumerate(files):
            file.save(tmpdir / secure_filename(file.filename))

        def run_upscale():
            try:
                used_names = set()
                processed = []
                for i, file in enumerate(files):
                    put_progress._file_index = i
                    stem = Path(secure_filename(file.filename)).stem
                    input_path = tmpdir / secure_filename(file.filename)
                    output_path = tmpdir / f"{stem}.{ext}"
                    process_one(input_path, output_path, "upscale", upscale_opts, progress_callback=put_progress)
                    processed.append((f"{stem}.{ext}", output_path))

                if len(processed) == 1:
                    out_name, out_path = processed[0]
                    result_path = tmpdir / "_result"
                    shutil.copy(out_path, result_path)
                    filename = out_name
                    count = 1
                else:
                    zip_path = tmpdir / "_result.zip"
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        for name, path in processed:
                            zf.write(path, name)
                    result_path = zip_path
                    filename = f"images_{uuid.uuid4().hex[:8]}.zip"
                    count = len(processed)

                with jobs_lock:
                    jobs[job_id]["result_path"] = result_path
                    jobs[job_id]["filename"] = filename
                    jobs[job_id]["count"] = count
                progress_queue.put({"done": True})
            except Exception as e:
                with jobs_lock:
                    jobs[job_id]["error"] = str(e)
                progress_queue.put({"done": True, "error": str(e)})

        thread = threading.Thread(target=run_upscale)
        thread.start()
        return jsonify({"job_id": job_id}), 202

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

    response = send_file(
        data, as_attachment=True, download_name=download_name, mimetype="application/octet-stream"
    )
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


@app.route("/jobs/<job_id>/stream")
def job_stream(job_id):
    """パイプライン進捗をSSEでストリーミング"""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "ジョブが見つかりません"}), 404

    def generate():
        queue = job["queue"]
        while True:
            try:
                event = queue.get(timeout=60)
            except Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
                continue
            if event.get("done"):
                yield f"data: {json.dumps(event)}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/jobs/<job_id>/result")
def job_result(job_id):
    """パイプライン処理結果をダウンロード"""
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "ジョブが見つかりません"}), 404
    if job.get("error"):
        return jsonify({"error": job["error"]}), 500
    result_path = job.get("result_path")
    filename = job.get("filename")
    if not result_path or not Path(result_path).exists():
        return jsonify({"error": "結果がまだ準備できていません"}), 404

    tmpdir = job.get("tmpdir")

    def cleanup():
        with jobs_lock:
            jobs.pop(job_id, None)
        if tmpdir and Path(tmpdir).exists():
            try:
                shutil.rmtree(tmpdir)
            except OSError:
                pass

    resp = send_file(
        result_path,
        as_attachment=True,
        download_name=filename,
        mimetype="application/octet-stream",
    )
    resp.headers["X-Count"] = str(job.get("count", 1))
    resp.call_on_close(cleanup)
    return resp


if __name__ == "__main__":
    os.chdir(ROOT_DIR)  # weights/ の相対パス用
    app.run(debug=True, host="0.0.0.0", port=5001)
