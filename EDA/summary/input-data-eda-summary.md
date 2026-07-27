# input データ EDA サマリー

## 行数検証
- conversation_train: 15,199 行、期待値 15,199、一致=True
- conversation_dev_test: 1,000 行、期待値 1,000、一致=True
- conversation_blind_a: 80 行、期待値 80、一致=True
- track_metadata_all_tracks: 47,071 行、期待値 47,071、一致=True
- track_metadata_test_tracks: 7,405 行、期待値 7,405、一致=True
- user_metadata_all_users: 8,772 行、期待値 8,772、一致=True

## 会話データの構造
- blind_a: 80 sessions、58 users、session あたり平均 rows=8.88、平均 music 件数=2.62。
- dev_test: 1,000 sessions、500 users、session あたり平均 rows=24.00、平均 music 件数=8.00。
- train: 15,199 sessions、8,591 users、session あたり平均 rows=24.00、平均 music 件数=8.00。

## ID coverage
- conversation_music_track_ids_in_all_tracks:train: missing=0、coverage=1.0000
- conversation_music_track_ids_in_all_tracks:dev_test: missing=0、coverage=1.0000
- conversation_music_track_ids_in_all_tracks:blind_a: missing=0、coverage=1.0000
- test_tracks は all_tracks に包含: missing=0

## Metadata の観察
- all_tracks は 47,071 tracks、ユニーク artist 名 10,637、tag 数の中央値 17.0。
- popularity 中央値=38.00、duration 中央値=3.78 分。

## Embedding の観察
- track audio-laion_clap: dim=512、nulls=0、zero_vectors=492、norm_median=1.0000
- track image-siglip2: dim=768、nulls=0、zero_vectors=586、norm_median=10.0657
- track cf-bpr: dim=128、nulls=0、zero_vectors=616、norm_median=0.0276
- track attributes-qwen3_embedding_0.6b: dim=1024、nulls=0、zero_vectors=492、norm_median=90.5467
- track lyrics-qwen3_embedding_0.6b: dim=1024、nulls=0、zero_vectors=492、norm_median=108.3203
- track metadata-qwen3_embedding_0.6b: dim=1024、nulls=0、zero_vectors=492、norm_median=97.7559
- user test_cold cf-bpr: dim=128、rows=129、nulls=0、norm_median=0.0000
- user test_warm cf-bpr: dim=128、rows=371、nulls=0、norm_median=0.0651
- user train cf-bpr: dim=128、rows=8591、nulls=0、norm_median=0.0375

## 次の実験候補
- BM25 corpus を比較する: title/artist/album baseline に対して tag_list と release_date を追加する。
- dev で diagnostic candidate recall@20/50/100 を測る。retrieval candidate と target turn の music-role track id を突き合わせる。
- turn number 別・goal category 別の人気 prior を reranker feature として試す。最終推薦対象は all_tracks を維持する。
- 広めの BM25/BERT candidate generation の後段で、metadata-qwen3 と cf-bpr embeddings による reranking を試す。

## 生成物
- tables: data_inventory.csv、conversation_stats.csv、id_coverage.csv、top_tracks_artists_tags.csv、および補助集計テーブル。
- figures: conversation_distributions.png、track_metadata_distributions.png、user_profile_distributions.png、embedding_norms.png。
