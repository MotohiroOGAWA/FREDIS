# 段階学習・統合の実データ検証

実装後の CLI と Workbench バックエンドをローカルの MoNA 実測スペクトルで検証した。予測精度の評価ではなく、データ作成・学習・特徴量保持・モデル統合・再読込の動作検証。

## 入力とモデル

- 入力: `raw_data/MoNA/flavonoids/mona_pos_cf_flavonoids.msds`（元ファイルは変更していない）。
- 学習: Flavanone の record 4305 / 4358（30 / 10 eV）。
- 検証: Flavone の record 3765 / 3762（30 / 50 eV）。学習と検証の分子は異なる。
- 実験ピークを保持し、数値の CE 列を追加。学習データ作成では相対強度 0.1 以上を採用。
- nonring single bond 6 patterns → carbonyl / ether motif をそれぞれ追加した 7-pattern 分岐 → 8-pattern 統合。追加二種類は動作検証用の定義であり、検証済みの fragmentation rule ではない。
- CUDA 学習、1 epoch、少数の小型モデル。逐次 fine-tuning の resume は 2 epoch まで実行。

## CLI 実行結果

| 工程 | 結果 | 時間（秒） |
| --- | --- | ---: |
| `prepare_base` | 成功 | 2.83 |
| `prepare_carbonyl` | 成功 | 2.32 |
| `prepare_ether` | 成功 | 2.6 |
| `prepare_union` | 成功 | 3.45 |
| `pretrain` | 成功 | 4.36 |
| `branch_carbonyl` | 成功 | 3.95 |
| `branch_ether` | 成功 | 3.93 |
| `sequential` | 成功 | 3.95 |
| `sequential_resume` | 成功 | 3.75 |
| `integrate` | 成功 | 2.78 |
| `integrated_heads` | 成功 | 3.77 |
| `predict_base` | 成功 | 2.25 |
| `predict_fine_tuned` | 成功 | 2.62 |
| `predict_integrated` | 成功 | 2.48 |
| `predict_heads` | 成功 | 2.67 |

## 特徴量と統合

- 既存 pattern の Graphormer・rule/product embedding・最終 readout と共有入力処理の全保存重みがビット単位で一致。
- 基本モデルと分岐・逐次学習・統合後の反応特徴量を同じ入力で比較。float32 の最大差は 7.15255737e-07、rtol=1e-5 / atol=1e-6 内。
- 統合前後で保存した専用エンコーダーを化学定義に基づいて照合。数値 ID の再割当後も全重みを保持。
- 統合後の予測器だけの学習では、同じ入力に対する最終反応特徴量がビット単位で一致。
- 推論で checkpoint を strict に再読込し、各出力 MSDS に 1 spectrum と非空のピークがあることを確認。
- 共通重み・重複 pattern 重み・共有特徴設定が競合する場合の拒否を unit test で検証。重みを平均しない。
- Workbench の実バックエンドからも `workbench_integrated.pt` と対応する Fragmenter 設定を作成。ブラウザー側では入力、要求メッセージ、成功通知を jsdom で検証。実際の VS Code GUI の手動操作までは行っていない。

## その他の検証

- `python -m pytest tests/ml tests/cli -q`: 101 tests passed（統合用追加テストの前）。
- 追加の統合・特徴量保持・初期化の関連テスト: 19 tests passed。head-only の誤った pattern 追加を拒否する追加テストを含む統合テスト: 5 tests passed。
- `npm run check`: Workbench の全既定チェックが成功。
- `check-training-ui.js` / `check-projects.js`: 更新後の training/fine-tuning/integration UI、設定保存、プロジェクトチェックが成功。

## 成果物

- 実行コマンド・ログ・時間: `data/workflow_verification_final/report.json`。
- 基本モデル: `data/workflow_verification_final/pretrain/last.pt`。
- 分岐モデル: `branch_carbonyl/last.pt`, `branch_ether/last.pt`。
- 逐次 fine-tuning・resume: `sequential/last.pt`。
- 統合: `integrated.pt` と `integrated.fragmenter.pft`。
- 統合後の予測器学習: `integrated_heads/last.pt`。
- 各学習ディレクトリの `trainable_parameters.json` に実際の更新対象を記録。

再実行方法と CLI / Workbench の操作は [training.md](training.md) を参照。検証スクリプトは `tests/workflows/verify_pattern_workflow.py`。


## New model の組み立て方式の検証

`PYTHONPATH=. python tests/workflows/verify_model_assembly.py --output data/model_assembly_verification_final` で、上記の実測スペクトルから作成した少量の prepared data を再利用し、以下を確認しました。

- 単体 `.reaction-model.pt` と予測モデルから選択した複数の反応モデルを合わせて読み込む。
- 追加の carbonyl モデルだけを幅 12・graph_dim 19・3 heads とし、TPSA / MolLogP を個別入力にする。他の反応モデルは幅 8 を保持する。
- New model の学習 → Resume → 別パターンの追加学習 → 分岐の統合 → 統合モデルの予測部の学習 → strict load / スペクトル予測。
- 各段階で読み込み済みの完全な反応特徴計算の重みを同一に保持する。統合モデルの予測部を学習しても、実際の最終反応特徴は完全一致する（差 0）。

コマンド・アーキテクチャ・検証結果は `data/model_assembly_verification_final/report.json` に保存しています。PC / PE などの予測精度や化学反応の妥当性を評価するものではありません。


## Training Data が反応モデルの指定を所有する構成

`tests/workflows/verify_preparation_model_selection.py` で、実測 4 スペクトルを用いて、学習済み nonring モデルと新規 carbonyl パターン（幅 12、3 層、TPSA / MolLogP）を **データ作成時**に指定しました。反応定義の取り込み・prepared data の生成・設定の自動継承・学習・Resume・strict prediction が成功し、学習済み特徴の重みを保持しました。

Training で反応モデルを再指定する引数は使っていません。検証結果は `data/preparation_model_selection_final/report.json` に保存しています。


## 生の化学特徴・外側の構造伝搬・位置 one-hot の検証

`tests/workflows/verify_preparation_model_selection.py` を `data/raw_neighborhood_workflow_verification` に実行しました。データ作成では反応定義と学習済みモデルだけを指定し、Training の architecture JSON で追加パターンの Graphormer hidden_dim=12、3 layers、周辺 hidden_dim=7、3 hops を指定しました。

データ作成 → 学習 → Resume → 既存分岐との統合 → 統合モデルを指定したデータ作成 → 予測部の学習 → strict prediction が成功しました。読み込み済みの重みを保持し、予測部の学習前後の反応特徴の差は 0 でした。記録: `data/raw_neighborhood_workflow_verification/report.json`。

単体テストでは 3 hop 境界からの勾配・4 hop の遮断・他の SMARTS 原子を通った混入の遮断・接続のない原子の厳密なゼロ・固定 one-hot 列・原子化学特徴の入力幅・パターン別の異なる hidden_dim を確認しています。


## データセットに学習済みモデルを同梱する検証

`data/self_contained_dataset_verification/report.json` は、実測 4 スペクトルを用いた検証です。指定した元の学習済みモデルをデータ作成後に削除し、データセットを別ディレクトリへ移動してから学習・Resume・統合・予測を実行しました。統合モデルを参照して作成した次のデータセットについても、元の統合モデルを削除して予測部を学習できました。固定反応特徴の差は 0 です。

モデルのコピーは `pretrained_models/` に保存され、root/split の設定から相対パスで参照します。単体反応モデルと予測モデルの両形式、複数パターンからの参照の重複排除、元ファイルとのバイト一致、出力内のモデルを使った再作成時の保持もテストで確認しています。
