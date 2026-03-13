# 画像処理ツール

Real-ESRGAN ベースの画像・動画処理ツール（アップスケール・形式変換・軽量化）

## セットアップ

```bash
python3 -m venv upscale-env
source upscale-env/bin/activate  # Windows: upscale-env\Scripts\activate
pip install realesrgan basicsr facexlib gfpgan
pip install flask
```

### 動画軽量化を使う場合
```bash
brew install ffmpeg  # macOS
```

## Web アプリの起動

```bash
source upscale-env/bin/activate
python app.py
```

ブラウザで http://localhost:5001 を開く

## コマンドライン

```bash
# アップスケール（デフォルト）
python upscale.py input.jpg output.png [--mode photo|anime] [--scale 4]

# 形式変換
python upscale.py convert input.jpg output.png [-q 95]

# 軽量化（画像・動画）
python upscale.py optimize input.png output.jpg [--max-width 1200] [-q 85]
python upscale.py optimize input.mp4 output.mp4 [--crf 18]
```
