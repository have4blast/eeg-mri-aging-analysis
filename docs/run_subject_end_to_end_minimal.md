# 被験者1件を end-to-end で実行する最小スクリプト

概要
- このドキュメントは、1人の被験者を前処理→ネットワーク解析まで一通り実行するための最小限の手順とサンプルスクリプトを示します。
- 目的：現行パイプラインの最小実行経路を確認し、後で自動化／並列化できるようにする。

前提
- データは `data/` または `preprocessed/` に配置されていること。
- 既存の前処理スクリプト（例：`preprocess_run.py`）およびネットワーク解析スクリプト（例：`connectivity.py`）があることを想定する。
- Python 環境に必要なパッケージがインストール済み（MNE, numpy, scipy, joblib など）。

推奨依存（例）
- requirements.txtに記載する最小項目
```
mne
numpy
scipy
joblib
pyyaml
pandas
matplotlib
```

簡単な設定ファイル例（`config/minimal_subject.yaml`）
```yaml
subject: sub-010002
data_root: ./data
preprocessed_dir: ./preprocessed
output_dir: ./results/minimal
n_surrogates: 100
n_jobs: 2
chunk_size: 50
bands:
  - name: alpha
    lfreq: 8
    hfreq: 12
```

最小スクリプト例: `scripts/run_subject_minimal.py`
- 役割：設定ファイルを読み、前処理（存在すれば呼び出し）、続けてネットワーク解析を行う。
- できるだけ既存のスクリプト／関数を呼び出す設計にする。

```python
#!/usr/bin/env python3
"""scripts/run_subject_minimal.py
使い方: python scripts/run_subject_minimal.py config/minimal_subject.yaml
"""
import sys
import os
import yaml
import subprocess
from pathlib import Path

cfg_path = Path(sys.argv[1])
with cfg_path.open() as f:
    cfg = yaml.safe_load(f)

subject = cfg['subject']
data_root = Path(cfg['data_root'])
pre_dir = Path(cfg['preprocessed_dir'])
out_dir = Path(cfg['output_dir'])
out_dir.mkdir(parents=True, exist_ok=True)

# 1) 前処理（存在する場合は呼び出す）
preprocess_script = Path('preprocess_run.py')
if preprocess_script.exists():
    cmd = [sys.executable, str(preprocess_script), '--subject', subject, '--data-root', str(data_root), '--out', str(pre_dir)]
    subprocess.run(cmd, check=True)
else:
    print('preprocess_run.py が見つかりません。既に前処理済みであることを想定して続行します。')

# 2) ネットワーク解析
connectivity_script = Path('connectivity.py')
if connectivity_script.exists():
    cmd = [sys.executable, str(connectivity_script), '--subject', subject, '--pre-dir', str(pre_dir), '--out', str(out_dir), '--n-surrogates', str(cfg.get('n_surrogates', 100)), '--n-jobs', str(cfg.get('n_jobs', 2))]
    subprocess.run(cmd, check=True)
else:
    # 最低限：preprocessed の .fif を読み込み、簡単な測度（例：バンドパワー）を出力するプレースホルダ
    import mne
    pre_fif = pre_dir / subject / f"{subject}_preprocessed.fif"
    if pre_fif.exists():
        raw = mne.io.read_raw_fif(pre_fif, preload=True)
        # 簡易：alpha帯域のパワーを計算して CSV に保存
        raw.filter(8., 12., fir_design='firwin')
        psds, freqs = mne.time_frequency.psd_welch(raw, fmin=8., fmax=12.)
        import numpy as np
        band_power = psds.mean(axis=(0,2)) if psds.ndim==3 else psds.mean(axis=1)
        import pandas as pd
        df = pd.DataFrame({'ch': raw.ch_names, 'alpha_power': band_power})
        df.to_csv(out_dir / f"{subject}_alpha_power.csv", index=False)
        print('簡易解析を完了しました。')
    else:
        raise FileNotFoundError(f'preprocessed file not found: {pre_fif}')

print('完了: 被験者', subject)
```

実行方法（PowerShell）
- 環境有効化後に実行
```
# 仮想環境を有効化した例
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
python scripts/run_subject_minimal.py config/minimal_subject.yaml
```

出力（例）
- `results/minimal/<subject>/` にネットワーク解析結果（CSV, npy, PNG など）を配置する想定
- サロゲート集計は `*_surrogate_counts.npy` のような簡易形式

注意点
- 本スクリプトは "最小実行経路" のテンプレートです。実際には `preprocess_run.py` や `connectivity.py` の CLI 引数に合わせて引数名を調整してください。
- HPC で多数被験者を並列に実行する場合は、このスクリプトを被験者単位でジョブ配列から呼び出すことを想定しています（内部で joblib 並列を使う場合は `n_jobs` を被験者ごとの割り当てに合わせて設定）。
- 再現性のため、乱数シードを設定しておくこと（サロゲート、ICA 等）。

次のステップ
- このテンプレートをリポジトリの実際の関数呼び出しに合わせて差し替える（必要なら私が具体的に `preprocess_run.py` と `connectivity.py` の引数を読んで修正案を作ります）。
