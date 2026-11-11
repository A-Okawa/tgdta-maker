# TG-DTA Maker

TG-DTA の測定ファイル（CSV / TXT）から、論文用の TG・DTG・DTA の図を作る Streamlit アプリです。
XRD Maker・Raman Maker・UV-Vis-NIR Maker・グラフ描画app（figapp）と同じ体裁（Arial 7 pt、目盛り 6.5 pt、
線 0.9 pt、軸 0.5 pt、外向きの目盛り、88 × 70 mm、同じ色の組み合わせ）で描くので、同じ論文の図と並べてもそろいます。
画面は 日本語・English・Tiếng Việt・ไทย で使えます。

## できること
- **読み込み**：装置の CSV（`Time(min),TEMP（℃）,DTA(μV),TG(mg),`、Shift_JIS）をそのまま。UTF-8・タブ区切り・
  前置きの行があるファイルも読みます。列は名前から見分け、見分けられなければ試料パネルの「列の対応」で選びます。
- **並べ方**：TG と DTA（または DTG）を左右の軸で 1 枚に／TG・DTG・DTA を上下のパネルに。複数の試料は重ね描き。
- **描く区間**：昇温部だけ（自動。最高温度での保持を除くので、温度軸で縦線が残らない）／すべて／時間で指定。
- **TG**：質量変化 Δm/m₀ (%)・残存質量 (%)・Δm (mg)。基準の温度（例：120 °C）で 0 にそろえることも。
- **DTG**：TG を Savitzky–Golay で平滑化してから微分（% min⁻¹ または % K⁻¹）。
- **DTA**：発熱を上／下、装置の符号（発熱が＋／−）、質量で割る (μV mg⁻¹)、始点を 0 に、試料ごとの縦の位置。
- **注記**：Exo. の矢印、DTA のピーク温度、温度範囲ごとの質量変化（例：−5.8%）、縦線（位置 | 文字）。
- **数値の一覧**：昇温速度・全体の質量変化・発熱/吸熱ピーク・範囲ごとの質量変化。描いた値は CSV で保存できます。
- **保存**：PDF・SVG（文字を編集可能）・TIFF・PNG（300 / 600 dpi）、作業ファイル（.json）、作図スタイル（.json。XRD Maker のスタイルも開けます）。

## 共通ファイル
`figure_style.py`・`palettes.py`・`rich_text.py`・`colors.csv`・`fonts/` は XRD Maker と同じファイルです。
片方を直したら、XRD Maker・Raman Maker・UV-Vis-NIR Maker・このアプリのすべてにコピーしてください
（`tests/test_tgdta.py` が XRD Maker 版と同じかを確かめます）。

## 動かし方
- 手元：`pip install -r requirements.txt` → `streamlit run app.py`
- PC 版（ターミナル不要）：サイドバー下の「PC 版をダウンロード」の ZIP を展開し、`setup_*` → `launch_*` を開く。
- Streamlit Community Cloud：Main file に `app.py`。Secrets に `password = "…"` を書くと最初にパスワードを聞きます。
- テスト：`python -m pytest tests`

## ファイル
| ファイル | 中身 |
|---|---|
| `app.py` | 画面（サイドバー・試料パネル・プレビュー） |
| `tgdta_io.py` | 読み込み・昇温部の判定・TG/DTG/DTA の計算・ピーク検出 |
| `tgdta_plot.py` | 論文用の図（matplotlib）とインタラクティブ表示（Plotly） |
| `project_io.py`・`style_templates.py` | 作業ファイル・作図スタイルの保存と読み込み（値の検査つき） |
| `i18n.py` | ベトナム語・タイ語の画面の文字 |
| `desktop.py`・`setup_local.py`・`build_bundle.py`・`setup_*`・`launch_*` | PC 版 |
