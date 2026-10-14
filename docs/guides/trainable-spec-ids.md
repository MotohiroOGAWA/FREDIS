# 学習可能な SpecID の抽出

FREDIS と FIORA・ICEBERG・MassFormer などで同じスペクトルを学習するため、各ツールが独立に判定した SpecID リストを照合します。FREDIS は実際のトレーニングデータ作成処理、モデルの順伝播、損失、逆伝播を実行し、有限の損失・勾配が得られるスペクトルを採用します。optimizer による重み更新はしません。

## CLI

```bash
fredis train trainable-ids \
  --input source.msds \
  --params patterns.fragmenter.pft \
  --output-dir /path/to/fredis-eligible \
  --workers 2 --batch-size 8
```

入力には一意で空でない `SpecID` が必要です。列名は `--spec-id-column` などで変更できます。反応パターン、アダクト、ピーク除外、探索制限によって判定が変わるため、本番のデータ作成と同じ設定を使ってください。`--params` 省略時は nonring single bond のプリセットを使います。

`--model-architecture architecture.json` で本番用のモデル設定を指定できます（設定そのもの、または `model_config` を含む JSON）。指定しない場合は既定の学習アーキテクチャです。モデルのサイズ・batch sizeに依存する判定なので、結果の `stats.json` に実効設定を記録します。これは任意の将来のモデルや全学習 epoch が成功する保証ではありません。assignment score の追加閾値による絞り込みはこのコマンドでは行いません。

出力ディレクトリは新規のものを指定します。

| ファイル | 内容 |
|---|---|
| `trainable_spec_ids.txt` | 学習可能な SpecID、1行1件、入力順 |
| `records.tsv` | 全件の採否、処理段階、除外理由、loss、勾配確認結果 |
| `stats.json` / `stats.txt` | 件数、除外理由、SMILES・アダクト・装置別集計、入力 SHA-256 |
| `run/` | 再開用 manifest、グループ別結果、worker ログ |

成功した batch の loss はその batch 全体の値です。batch の処理が失敗した場合は個別に確認し直します。worker の異常終了も理由付きで除外されます。中間データを残すには `--keep-export` を指定します。

同じ引数に `--resume` を追加すると、保存済みグループを再利用できます。入力、実装、アーキテクチャ、import した重み、処理設定が変わった場合は再開を拒否します。完了済み結果も SpecID ファイルの SHA-256 を照合します。

抽出した `trainable_spec_ids.txt` は他ツールと同じ1行1 SpecID の形式です。共通集合の計算は外部の共通ツールで行ってください。入力ファイルの照合に使える SHA-256 は `stats.json` に保存します。

検証は CPU で行います。`--workers` は並列 worker プロセス数です。device の選択や処理時間による timeout はありません。

## Workbench

Data → **Trainable Spectra** を開きます。

1. Training Data で反応パターン・アダクト・前処理を設定します。
2. 左側で入力 `.msds` と新規出力ディレクトリを選び、**Verify and export SpecIDs** を実行します。必要なら現在の Training architecture を使用できます。
3. 出力された `trainable_spec_ids.txt` を外部の共通集合ツールに渡します。

進捗、処理ログ、結果の件数をページ内で確認できます。Stop で中断した抽出は、同じ設定と出力先に Resume を指定して再開できます。

**Copy command** で現在の設定の CLI コマンドをクリップボードにコピーできます。コピーでは処理を開始しません。
