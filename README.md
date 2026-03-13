# 画像処理ツール

Real-ESRGAN ベースの画像・動画処理ツール（アップスケール・形式変換・軽量化・背景削除）

## セットアップ

```bash
python3 -m venv upscale-env
source upscale-env/bin/activate  # Windows: upscale-env\Scripts\activate
pip install realesrgan basicsr facexlib gfpgan
pip install flask
pip install rembg onnxruntime  # 背景削除用
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

### 主な機能
- **形式変換**: PNG / JPEG / WebP への変換
- **軽量化**: 画像・動画の圧縮（長辺1920px以下、動画は H.264）
- **アップスケール**: Real-ESRGAN で 2倍/4倍（写真・アニメモード）
- **背景削除**: rembg で人物・商品などの背景を自動削除（透過 PNG）
- **プレビュー**: 1枚処理時は完了後に結果をプレビュー表示（透過はチェッカーボード背景で表示）

## コマンドライン

```bash
# アップスケール（デフォルト）
python upscale.py input.jpg output.png [--mode photo|anime] [--scale 4]

# 形式変換
python upscale.py convert input.jpg output.png [-q 95]

# 軽量化（画像・動画）
python upscale.py optimize input.png output.jpg [--max-width 1200] [-q 85]
python upscale.py optimize input.mp4 output.mp4 [--crf 18]

# 背景削除
python upscale.py removebg input.jpg output.png
```
