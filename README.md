# eeg-analysis-visualisation-app

EEG Analysis and Visualisation Application - EEG-MRI加齢解析パイプライン

## Overview

EEGデータの前処理、コネクティビティ解析、ネットワーク解析を行うパイプラインです。

### Features

- **EEG前処理**: BrainVisionデータの読み込み、フィルタリング、ICAによるアーティファクト除去
- **コネクティビティ解析**: PSI, PLI, wPLI, 虚数コヒーレンスの計算
- **周波数帯域間結合**: n:m位相結合解析（Delta, Theta, Alpha, Beta, Gamma）
- **ネットワーク解析**: グラフ理論に基づくネットワークメトリクス
- **可視化**: ヒートマップ、コネクティビティサークルプロット

## Project Structure

```
├── config/                  # 設定ファイル (YAML)
│   ├── eeg.yaml            # EEG前処理パラメータ
│   ├── settings.yaml       # パイプライン全体設定
│   └── load_config.py      # 設定読み込みユーティリティ
├── scripts/                 # 処理スクリプト
│   └── eeg/
│       ├── preprocess.py       # EEG前処理関数
│       ├── preprocess_run.py   # バッチ前処理実行
│       ├── connectivity.py     # コネクティビティ解析
│       ├── connectivity_plot.py # 可視化関数
│       └── psd.py              # パワースペクトル密度解析
├── utils/                   # ユーティリティ
│   └── parser.py           # コマンドライン引数パーサー
├── metadata/                # メタデータ
│   ├── EEG64chElectrode.txt    # 64ch電極名
│   └── Participants_MPILMBB_LEMON.csv  # 被験者情報
├── jupyter_notebook/        # 探索用Jupyterノートブック
├── main.py                  # メインエントリーポイント (run_psi_2)
├── main2.py                 # 代替エントリーポイント (run_psi)
└── pipeline.py              # パイプラインスクリプト
```

## Setup

```bash
pip install mne numpy scipy pandas matplotlib seaborn networkx mne-connectivity
```

## Usage

```bash
# 動的コネクティビティ解析 (Version 2)
python main.py --raw_path /path/to/raw --preprocess_path /path/to/preprocessed

# 基本PSI解析 (Version 1)
python main2.py --raw_path /path/to/raw --preprocess_path /path/to/preprocessed
```

## Configuration

`config/eeg.yaml` でEEG前処理パラメータを設定:
- フィルタ: 1-40 Hz バンドパス + 50 Hz ノッチ
- ICA: FastICA / Infomax
- リファレンス: 平均リファレンス

## Data

LEMON (Leipzig Study for Mind-Body-Emotion Interactions) データセットを使用:
- 若年群 (20-30歳) と高齢群 (65-80歳)
- 64チャンネルEEG
- 開眼/閉眼条件

