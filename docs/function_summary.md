# 関数サマリ（詳細）

このドキュメントは、指定されたスクリプト群に含まれる主な関数とその役割、引数・返り値、および内部で行われる処理手順を詳述したものです。関数の内部実装はソースを参照してくださいが、ここでは処理フローをステップごとにまとめています。

---

## scripts/eeg/batch_surrogate_networks.py

### subject_list_from_range(start_id, end_id)
- 目的: 指定した ID 範囲から被験者 ID のリストを生成する。
- 入力: start_id (例: 'sub-010002'), end_id (例: 'sub-010036')
- 出力: ['sub-010002', ..., 'sub-010036']
- 手順:
  1. 文字列から数値部分を抽出する（'sub-010002' -> 10002）。
  2. 開始・終了の数値を int に変換。
  3. range を使って包括的な番号列を生成。
  4. 各番号をゼロ埋めして 'sub-{i:06d}' 形式にフォーマットして返す。

### load_pair_pickle(pair_path)
- 目的: ペアワイズ解析結果の pickle を安全に読み込む。
- 入力: pair_path (Path または文字列)
- 出力: 読み込んだオブジェクト（dict や ndarray）または None
- 手順:
  1. 指定パスが存在するか確認。
  2. open(..., 'rb') でファイルを開く。
  3. pickle.load を実行してオブジェクトを取得。
  4. 読み込み失敗（破損、型不一致等）の場合は例外をキャッチして None を返す。

### run_for_band(subjects, band_pair, args)
- 目的: 各被験者・バンドについて観測行列の読み込み、必要ならサロゲート検定、閾値処理、ネットワーク解析を実行して結果を保存する。
- 入力: subjects (list), band_pair (tuple e.g. ('alpha','alpha')), args (Namespace/設定)
- 出力: 被験者毎の集約 DataFrame
- 手順:
  1. 各被験者について出力ディレクトリを作成。
  2. まず既存の pickle / npy を探し読み込む（標準パターンと古いパターンの両方）。
  3. 読み込めたオブジェクトから observed_mean（psi_mean）と p_matrix, sig_mask を取得する。
  4. 観測行列が無ければ保存済みの observed.npy を探す。
  5. 観測行列や p 行列がなければ surrogate_psi_test_for_pair を呼び出して計算する（時間がかかる）。
  6. p_matrix があって sig_mask が無ければ benjamini_hochberg を適用して sig_mask を作る。
  7. sig_mask を使って有意辺のみを残した隣接行列 adj_sig を作成、対角成分は False にする。
  8. analyze_network_from_psi によりネットワーク指標を計算（密度・中心性・効率など）。
  9. get_subject_metadata でメタデータ（性別・グループ等）を取得し、結果行に付与。
  10. per-subject CSV と adjacency / siglist を保存。
  11. すべての被験者処理後に DataFrame を作り返す。

### group_compare_and_save(df, args, band_label)
- 目的: 被験者指標をグループ別に保存・比較する。
- 入力: df (集計 DataFrame)、args、band_label
- 出力: グループ別 CSV、グループ比較結果 CSV
- 手順:
  1. 出力ディレクトリを作成。
  2. 全被験者分の CSV を保存。
  3. df を 'group' 列で groupby し、各グループごとに CSV を保存。
  4. ユニークなグループが2つあれば、各指標について 2 群の t 検定を実行。
  5. 検定結果（平均差、t 値、p 値等）を CSV に保存。

### main()
- 目的: スクリプトのエントリーポイント。コマンドライン引数を解釈して一連の処理を呼び出す。
- 手順:
  1. argparse で引数を取得（被験者範囲、バンド、サロゲート数等）。
  2. subject_list_from_range で被験者リストを作る。
  3. band ペアを構築して run_for_band を呼ぶ。
  4. 最後に group_compare_and_save で集計と比較を行う。

---

## scripts/eeg/connectivity_plot.py

これらは可視化と行列保存に特化したユーティリティです。

### plot_save_psi_matrix(psi_matrix, ch_names, id, condition, output_path)
- 目的: PSI 行列をヒートマップとして保存する。
- 手順:
  1. matplotlib / seaborn で Figure を作成。
  2. sns.heatmap で行列をプロット（チャネル名を軸ラベルに使用）。
  3. 出力先ディレクトリを作成し、PNG を保存。ファイル名は {id}_{condition}.png。

### plot_save_connectivity_circle(con_matrix, ch_names, id, condition, output_path)
- 目的: 接続円図を描画して保存する（mne の plot_connectivity_circle を想定）。
- 手順:
  1. 非対話モードにして Figure を作成。
  2. plot_connectivity_circle に行列とチャネル名を渡して描画。
  3. 出力先ディレクトリを作り PNG を保存。

### save_psi_matrix(psi_matrix, id, condition, output_path)
- 目的: PSI 行列を .npy として保存する。
- 手順:
  1. 出力ディレクトリを作成。
  2. numpy.save で {id}_{condition}_psi_matrix.npy を保存。

---

## scripts/eeg/connectivity.py

このファイルには信号処理・同期指標計算・ネットワーク解析の中核関数が含まれます。ここでは主要な関数の入力・出力・手順を説明します。

### create_epochs(raw, epoch_duration)
- 目的: 固定長イベントから Epochs を作る。
- 手順:
  1. mne.make_fixed_length_events でイベントを生成。
  2. mne.Epochs を作成し preload=True で返す。

### get_phase(epochs, method='wavelet', f=(0.5,4))
- 目的: 波形から位相を抽出する（各チャネル・サンプルごとの瞬時位相）。
- 入力: epochs（n_epochs, n_channels, n_times）
- 出力: phase 配列
- 手順（一般）:
  1. epochs.get_data() でデータ配列と sampling rate を取得。
  2. method に応じて処理:
     - 'hilbert': バンドパス（指定周波数帯）を適用後、解析信号を得るため hilbert を適用し、np.angle で位相を取得。
     - 'wavelet': tfr_array_morlet 等を使って複素スペクトルを計算し、その位相を抽出。
  3. 必要に応じてエポックを連結してチャネル×時間形状に整形して返す。

### proportional_thresholding(psi_matrix, proportion)
- 目的: 行列の全エッジのうち上位 proportion を残す。
- 手順:
  1. 上三角の非対角要素の絶対値を flatten。
  2. quantile を計算してカットオフを決定。
  3. 閾値以上の要素のみ残す新しい隣接行列を返す。

### apply_thresholding(...)
- 目的: 指定方式で閾値処理（proportional または 絶対閾値）を行い、binary/weighted の選択を適用する。
- 手順:
  1. threshold_type により proportion を使うか threshold を使うか分岐。
  2. network_type によって値を 0/1 に変換するか、重みを保持するか選択。

### analyze_network_from_psi(adj, mode="weighted")
- 目的: 隣接行列からグラフ指標を計算する。
- 入力: adj (numpy array, n×n)
- 出力: results dict
- 手順:
  1. networkx.from_numpy_array でグラフを作る（weighted に対応）。
  2. 密度を計算（非ゼロ要素比）。
  3. degree_centrality, その他 centrality（weighted の場合は度の重み対応）を計算。
  4. グローバル効率 / ローカル効率を計算。
  5. 結果を辞書化して返す。

### benjamini_hochberg(pvals, alpha=0.05)
- 目的: 多重比較を制御する BH 手法で有意マスクを作る。
- 入力: pvals マトリクスまたは 1D 配列
- 出力: boolean マスク（同サイズ）
- 手順:
  1. p 値を一次元に展開しソート。
  2. 各 p_i に対して閾値 i/m * alpha を比較し、最大の有意インデックスを見つける。
  3. 元の形に戻して True/False マスクを返す。

### _phase_randomize_channel(sig, random_state=None)
- 目的: 単一チャネル信号の位相をランダム化してサロゲート信号を作る。
- 手順:
  1. FFT を取り振幅と位相に分離。
  2. 位相をランダムな一様分布で置換（対称性を保つため共役を考慮）。
  3. 逆 FFT で時間領域に戻す。

### phase_randomize_signals(data, random_state=None)
- 目的: 全チャネルについて位相ランダム化を行い、サロゲートデータを生成する。
- 手順:
  1. 各チャネルについて _phase_randomize_channel を呼ぶ。
  2. サロゲートセット（n_surrogates × チャネル × 時間）を返す（メモリに注意）。

### surrogate_psi_test_for_pair(...)
- 目的: 指定バンドで PSI を計算し、サロゲート法で p 値と有意性を評価する主要関数。
- 入力（抜粋）: data (channels × time)、sfreq、pair_key（バンドペア）、band_list、n_surrogates、ウィンドウ設定、random_state、alpha
- 出力: observed_mean, p_matrix, sig_mask, surrogates (可能な場合)
- 手順（概要）:
  1. band_list の各バンドに対して中心周波数・ウィンドウ長を決定し、データをバンド通過フィルタする。
  2. データをスライディングウィンドウで切り、各ウィンドウで位相を抽出（get_phase）。
  3. 各ウィンドウでチャネル間 PSI を計算し、ウィンドウ平均して observed_mean を得る。
  4. サロゲート数分ループ（もしくは並列化）:
     - phase_randomize_signals によってサロゲートデータを生成。
     - サロゲートから同様にウィンドウ単位で PSI を計算し、分布を得る。
  5. 観測値とサロゲート分布を比較して 2 尾あるいは片側の p 値を計算し p_matrix を作成。
  6. benjamini_hochberg を適用して sig_mask を作る（alpha に従う）。
  7. 必要なら surrogates を返す、また途中結果をファイルに保存するオプションがある。

### run_surrogate_and_plot_for_subject(...)
- 目的: 被験者単位で surrogate_psi_test_for_pair を呼び、結果の保存と可視化を行う。
- 手順:
  1. surrogate_psi_test_for_pair を呼んで observed_mean, p_matrix, sig_mask を得る。
  2. sig_mask があれば有意ペア一覧を作成してテキスト保存。
  3. 有意接続のみの隣接行列で connectivity circle を描画して PNG 保存。

---

## scripts/eeg/preprocess_run.py

### get_subject_ids(root_path)
- 目的: 前処理対象の被験者ディレクトリを列挙する。
- 手順:
  1. os.listdir で root_path の直下を走査。
  2. 名前が 'sub-' で始まるディレクトリのみを残してソートして返す。

### process_subject(args, subject_id)
- 目的: 1 被験者分の前処理ワークフローを実行する。
- 手順:
  1. preprocess.load_rawdata を呼んで Raw を読み込む。
  2. チャネルタイプ設定（VEOG を 'eog' にする）。
  3. preprocess.set_montage でモンタージュを適用。
  4. preprocess.set_eeg_reference で平均参照を適用。
  5. preprocess.divide_conditions で EO/EC を分割。
  6. preprocess.filter_data を呼んで各条件をフィルタ。
  7. preprocess.apply_ica で ICA を適用（EOG 関連成分を検出し除去）。
  8. preprocess.save_raw で 前処理済みファイルを保存。
  9. 例外が発生した場合は logger.error で記録し、最終的にメモリを解放する。

### preprocess_all_subjects(args)
- 目的: 全被験者について process_subject を順次実行する。進捗表示あり。

---

## scripts/eeg/preprocess.py

### load_rawdata(args, id)
- 目的: .vhdr/.edf 等の生ファイルを読み込んで mne.io.Raw を返す。
- 手順:
  1. raw ディレクトリ構造からファイルパスを組み立て。
  2. ファイル存在確認。
  3. mne.io.read_raw_* を使って読み込む（必要に応じて preload=True）。
  4. 失敗時は例外を投げる。

### divide_conditions(raw)
- 目的: Raw の注釈から EO/EC ブロックを抽出・連結してそれぞれの Raw を返す。
- 手順:
  1. mne.events_from_annotations で events と event_id を取得。
  2. 'Stimulus/S200'（EO）と 'Stimulus/S210'（EC）のコードを取得。
  3. events を巡って隣り合うイベントの間をブロックとして抽出し、条件ごとにリストに追加。
  4. 各リストを mne.concatenate_raws で連結して raw_eo, raw_ec を返す（存在しない場合は None）。

### apply_ica(args, raw, condition, id, random_state=97)
- 目的: ICA によるアーチファクト除去と結果の可視化。
- 手順:
  1. mne.preprocessing.ICA を初期化して fit(raw) を実行。
  2. ica.find_bads_eog などで EOG 関連成分を検出し ica.exclude に記録。
  3. save_ica_plots で成分図とスコア図を保存。
  4. ica.apply を呼び、除去済みの Raw を返す。

### save_ica_plots(ica, eog_scores, condition_name, id, output_dir)
- 目的: ICA の図をファイルに保存する。出力ディレクトリが無ければ作成。

### save_raw(raw, output_path, id, condition)
- 目的: 前処理済み Raw を .fif 等で保存する。
- 手順:
  1. 出力パスを作成。
  2. raw.save(path, overwrite=True) 等で保存。

### compute_csd(raw)
- 目的: Raw からクロススペクトル密度を計算する（mne.time_frequency の関数を使用）。

---

## scripts/eeg/psd.py

### run_psd(args)
- 目的: 前処理済みデータから各チャネルの PSD を計算して保存するバッチ。
- 手順:
  1. preprocess_run.get_subject_ids で被験者一覧を取得。
  2. 各被験者・条件について前処理済みファイルを読み込む。
  3. compute_psd_multichannel を呼んで周波数軸と PSD を得る。
  4. 必要なら save_psd_metrics_csv で指標を CSV に保存。

### compute_psd_multichannel(raw, fmin=0.5, fmax=45)
- 目的: mne の psd_welch を使いチャネルごとの PSD を計算する。
- 出力: freqs, psd_array (n_channels × n_freqs)

### save_psd_metrics_csv(metrics_dict, subject_id, sex, group, condition, output_path, file_name)
- 目的: 指標辞書を CSV に追記または新規作成する。
- 手順:
  1. 行データを DataFrame にして、既存ファイルがあればヘッダなしで追記、無ければ新規作成。

---

## scripts/eeg/run_single_surrogate_test.py

このファイルは特定被験者のサロゲート PSI テストを素早く実行するためのスクリプトです。主な処理フロー:
1. ワークスペースルートを sys.path に追加してパッケージを import 可能にする。
2. 前処理済み raw (.fif) を読み込み、チャネル選択（eeg）とデータ抽出を行う。
3. surrogate_psi_test_for_pair を呼んで observed_mean, p_matrix, sig_mask, surrogates を得る。
4. 結果を .npy として保存し、有意ペア一覧をテキストに保存する。
5. 有意接続が存在すれば接続円図を描画して PNG に保存。

---

## その他
- scripts/eeg/__init__.py: 空ファイル。パッケージ初期化用。

---

補足: 上記は関数のアルゴリズム手順と呼び出し関係を明確にするための要約です。実際の例外処理や並列化（joblib 等）、I/O の細かな振る舞いは各ソースファイルの実装を参照してください。より細かい引数仕様（各引数の型や既定値）や、各関数の戻り値の具体的な配列形状などを追加したい場合は、対象ファイルを指定してください。
