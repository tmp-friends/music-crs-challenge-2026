# Music-CRS 2026: Team Komekami Final Submission Code

Code submission for **RecSys Challenge 2026: Conversational Music Recommendation (Music-CRS)** by team **Komekami**, and the code / released tables for the accompanying workshop paper *"Auditing Alignment among Item Relevance, Goal Progress, and LLM-Judged Responses in Music-CRS"*.

- Challenge website: <https://nlp4musa.github.io/music-crs-challenge/>
- RecSys Challenge 2026: <https://www.recsyschallenge.com/2026/>
- Final results: <https://nlp4musa.github.io/music-crs-challenge/results.html>

## Final result (Blind B, hidden leaderboard)

Selected submission: **`submission_exp116_lgbm6_nocs_B.zip`** (LightGBM 6-seed LambdaRank, 12 candidate sources, cross_session source fully dropped).

| Metric | Score |
|---|---:|
| Composite | **0.4998** |
| nDCG@20 | 0.3528 |
| Catalog Diversity | 0.0275 |
| Lexical Diversity | 0.7315 |
| LLM-judge | 4.30 |
| Rank | 16 / 40 teams overall (Industry Track: 8th) |

All three Blind B submissions (max 3 per team) and their artifacts are tracked in [exp/inference/blindset_B/](exp/inference/blindset_B/):

| Submission | Configuration | Codabench score |
|---|---|---:|
| `submission_exp116_lxct6_B.zip` | LightGBM+XGBoost+CatBoost+TabM, 4 families × 6 seeds, with cross_session | (primary) |
| `submission_exp116_lgbm6_B.zip` | LightGBM 6-seed, with cross_session | 0.47 |
| **`submission_exp116_lgbm6_nocs_B.zip`** | **LightGBM 6-seed, cross_session dropped (12 sources)** | **0.50** |

## System overview

Three-stage pipeline. The recommendation path (stages 1–2) is CPU-only sparse retrieval + GBDT: no dense/neural retriever, no fine-tuned model, no external data.

```text
dialogue history + (if available) user_profile
   │
   ├─ [1] BM25 anchor (exp015 profile B):
   │      4 BM25 views + 4 exact-match fields → RRF union top-500
   │      → stage-1 LightGBM (binary logloss) → anchor ranking
   │
   ├─ [2] 12 auxiliary candidate sources (top-100 each):
   │      continuity ×4 / query-neighbor memory ×3 / entity ×4 / transition ×1
   │
   ├─ [3] union + within-session de-duplication → 73 rank-based features
   │      → LightGBM LambdaRank, 6-seed average → top-20 ranking
   │
   └─ [4] response generation: frozen Claude (claude-opus-4-8, no fine-tuning)
          grounded on the final top-1 track metadata; all 80 rows freshly generated
```

The complete command lines, reports, and validation outputs of the final runs are in [mcrs/experiments/exp116_gbdt_family_ensemble/README.md](mcrs/experiments/exp116_gbdt_family_ensemble/README.md).

## Repository layout

```text
.
├── run_inference_devset.py / run_inference_blindset.py   # official baseline runners (reused by stage 1)
├── mcrs/
│   ├── crs_baseline.py, db_item/, db_user/, retrieval_modules/, lm_modules/, system_prompts/
│   │                                # official baseline core (BM25/BERT retrieval, catalog/profile DB, LLM)
│   ├── candidates/, ltr/, rerank_modules/                # shared candidate/LTR/reranker modules
│   └── experiments/                                      # only the folders on the final-submission path:
│       ├── exp015_candidate_rules_ablation/              #   stage-1 BM25 anchor (build/train/apply + configs)
│       ├── exp021_candidate_fusion/                      #   continuity + query-neighbor-memory sources
│       ├── exp023_entity_candidates/                     #   entity sources
│       ├── exp024_transition_memory/                     #   transition source
│       ├── exp090_cross_session_source/                  #   cross_session source + within-session exclude
│       ├── exp022/ exp027/ exp058/                       #   feature builder + seed-ensemble modules (imported)
│       ├── exp063_response_quality/                      #   response generation (generate_responses_v2.py)
│       ├── exp116_gbdt_family_ensemble/                  #   final reranker (ensemble_rerank.py) + run scripts
│       └── results_ledger.jsonl                          #   append-only ledger of all scored runs
├── exp/
│   ├── inference/blindset_B/                             # final submission JSONs + ZIPs (tracked)
│   └── smoke_blindB/                                     # Blind B robustness smoke: sources, logs, REPORT.md
├── weights/                                              # trained GBDT weights of the final submission
├── EDA/                                                  # audit code, annotation, released tables (paper)
├── docs/paper/                                           # audit paper mirrors, annotation guideline, figures
└── scripts/                                              # download_data.py, validate_submission.py, audits
```

This repository is a curated snapshot of our working repository, restricted to the final-submission pipeline and the published analyses (fresh git history for a clean release).

## Setup

Python 3.10, [uv](https://docs.astral.sh/uv/):

```bash
uv venv .venv --python=3.10
source .venv/bin/activate
uv pip install -e ".[eda]"
uv pip install xgboost catboost claude-agent-sdk
```

Versions used for the final runs: `lightgbm 4.6.0`, `xgboost 3.2.0`, `catboost 1.2.10`, `bm25s 0.3.9`, `datasets 4.8.5`, `transformers 5.9.0`, `torch 2.6.0+cu124`, `numpy 2.2.6`, `pandas 2.3.3`, `pyarrow 24.0.0`, `omegaconf 2.3.0`, `claude-agent-sdk 0.2.106`. `xgboost`/`catboost` (and a CUDA GPU) are only needed for the `lxct6` variant; the selected final submission uses LightGBM only (CPU). `flash-attn` is optional (configs default to `sdpa`).

**Data.** Only the official challenge datasets are used, loaded directly from Hugging Face (`talkpl-ai/TalkPlayData-Challenge-*`); no manual download step is required (optionally `python scripts/download_data.py` prefetches everything). No external data of any kind is used anywhere (retrieval, features, training, inference, response generation); in particular, no LFM-2B / listening-history data.

**Response-generation credentials.** Stage 4 calls the frozen `claude-opus-4-8` model through `claude-agent-sdk`, which reuses a local [Claude Code](https://claude.com/claude-code) CLI login (`claude` installed and logged in) or an `ANTHROPIC_API_KEY` environment variable. Rankings (stages 1–3) are fully deterministic given the fixed seeds; the LLM response text naturally varies between runs (the model and prompts are fixed).

## Reproducing the final submission (Blind B → prediction.json)

**Trained weights.** The trained GBDT models of the final submission are published in [weights/](weights/): the stage-1 anchor LightGBM and the 6 stage-2 LambdaRank boosters, with a model card ([weights/README.md](weights/README.md)) describing training data, parameters, and verification. The same files are mirrored on Hugging Face: [komekami/music-crs-challenge-2026](https://huggingface.co/komekami/music-crs-challenge-2026). They allow direct inspection and verification without retraining; both stages are also fully retrainable from the official data by the commands below (stage-2 training happens inside `ensemble_rerank.py`, is deterministic, and reproduces the published boosters and the submitted Blind B ranking exactly). The response model is an unmodified commercial LLM (`claude-opus-4-8`) with no weights to distribute. Blind-side inputs are already tracked in the repo, so steps 1–3 regenerate the dev/train-side inputs only.

Tracked inputs you do **not** need to regenerate:

- Blind B anchor ranking: [exp/inference/blindset_B/exp015_B_ll_t500_B.json](exp/inference/blindset_B/exp015_B_ll_t500_B.json)
- Blind B candidate sources + within-session exclude list: [exp/smoke_blindB/sources/](exp/smoke_blindB/sources/)
- Final submission artifacts for byte-level format comparison: [exp/inference/blindset_B/](exp/inference/blindset_B/)

### 1. Stage-1 anchor: train + dev predictions (heaviest step)

```bash
ONLY_RUNS=B TOPKS=500 bash mcrs/experiments/exp015_candidate_rules_ablation/run_retrain.sh
```

Builds the top-500 candidate parquets over Train/Dev, trains the stage-1 LightGBM (binary logloss), and writes
`mcrs/experiments/exp015_candidate_rules_ablation/results/top500/B_exact_split/lgbm_B_ll_top500_dev_predictions.json` (the `--primary` input of step 4) and `.../lgbm_B_ll_top500.txt` (the model used by the Blind B anchor config). Single-process this takes several hours; `build_dataset.py` supports `--task_shard_index/--task_shard_count` for parallel builds (~12× on a 28-core machine, ~3 GB RAM per shard). For a quick smoke run use `TRAIN_LIMIT`/`DEV_LIMIT`.

To regenerate the (already tracked) Blind B anchor JSON from the trained model:

```bash
python run_inference_blindset.py --tid exp015_B_ll_t500_B --eval_dataset blindset_B --batch_size 4
```

(config: [mcrs/experiments/exp015_candidate_rules_ablation/configs/exp015_B_ll_t500_B.yaml](mcrs/experiments/exp015_candidate_rules_ablation/configs/exp015_B_ll_t500_B.yaml); `track_split_types: [all_tracks]`, i.e. the full 47,071-track catalog).

### 2. Dev-side candidate sources (12 sources, top-100 each)

```bash
python mcrs/experiments/exp021_candidate_fusion/continuity_candidates.py --topk 100 \
  --output_dir mcrs/experiments/exp021_candidate_fusion/results/continuity_candidates
python mcrs/experiments/exp021_candidate_fusion/query_neighbor_memory.py --topk 100 \
  --variants "current,recent_profile,history_music" \
  --output_dir mcrs/experiments/exp021_candidate_fusion/results/query_neighbor_memory
python mcrs/experiments/exp023_entity_candidates/entity_candidates.py --topk 100 \
  --variants "current,recent2,recent4,current_tight" \
  --output_dir mcrs/experiments/exp023_entity_candidates/results/entity_candidates_top100
python mcrs/experiments/exp024_transition_memory/transition_memory.py --topk 100 \
  --variants "recent1" \
  --output_dir mcrs/experiments/exp024_transition_memory/results/transition_memory_top100
```

### 3. Blind-side candidate sources (already tracked; to regenerate)

```bash
bash exp/smoke_blindB/run_blindB_smoke_sources.sh   # continuity / qnm / entity / cross_session for Blind B
python mcrs/experiments/exp024_transition_memory/transition_memory.py \
  --blind_dataset_name talkpl-ai/TalkPlayData-Challenge-Blind-B \
  --variants "recent1" --topk 100 --output_dir exp/smoke_blindB/sources/transition
python mcrs/experiments/exp090_cross_session_source/generate_exclude_json.py \
  --blind_dataset_name talkpl-ai/TalkPlayData-Challenge-Blind-B \
  --output exp/smoke_blindB/sources/within_session_exclude_blindB.json
```

See [exp/smoke_blindB/REPORT.md](exp/smoke_blindB/REPORT.md) for the cold-start robustness smoke test that motivated dropping the cross_session source on Blind B.

### 4. Stage-2 reranker → Blind B top-20 ranking

Run the exact final command (LightGBM 6-seed, 12 sources, cross_session excluded) documented in [exp116 README, section "LightGBM 6-seed / cross_session 完全 drop Blind B hedge"](mcrs/experiments/exp116_gbdt_family_ensemble/README.md). Abbreviated:

```bash
python mcrs/experiments/exp116_gbdt_family_ensemble/ensemble_rerank.py \
  --families "lgbm" \
  --seeds "20260616,20260617,20260618,20260619,20260620,20260621" \
  --skip_oof \
  --primary "exp015B=mcrs/experiments/exp015_candidate_rules_ablation/results/top500/B_exact_split/lgbm_B_ll_top500_dev_predictions.json" \
  --blind_primary "exp015B=exp/inference/blindset_B/exp015_B_ll_t500_B.json" \
  --source  ... (12 dev-side source JSONs from step 2) \
  --blind_source ... (12 Blind-side source JSONs from step 3) \
  --exclude_json exp/smoke_blindB/sources/within_session_exclude_blindB.json \
  --allow_short_sources --protect_top 0 \
  --num_boost_round 450 --num_leaves 31 --min_data_in_leaf 80 --lambda_l2 8.0 --num_threads 8 \
  --output_report  mcrs/experiments/exp116_gbdt_family_ensemble/results/ranker/report_ens_blindB_lgbm6_nocs.json \
  --output_dev_full mcrs/experiments/exp116_gbdt_family_ensemble/results/ranker/dev_full_ens_blindB_lgbm6_nocs.json \
  --output_blind_tracks mcrs/experiments/exp116_gbdt_family_ensemble/results/ranker/blindB_tracks_ens_blindB_lgbm6_nocs.json
```

Expected report values (deterministic): `candidate_rows = 2,949,717`, `candidate_positive_count = 4,333`, dev full-fit blend nDCG `0.2587`.

### 5. Response generation (all 80 rows, freshly generated)

```bash
python mcrs/experiments/exp063_response_quality/generate_responses_v2.py \
  --backend claude --model claude-opus-4-8 \
  --tracks_json mcrs/experiments/exp116_gbdt_family_ensemble/results/ranker/blindB_tracks_ens_blindB_lgbm6_nocs.json \
  --output exp/inference/blindset_B/exp116_lgbm6_nocs_B.json \
  --test_dataset_name talkpl-ai/TalkPlayData-Challenge-Blind-B \
  --batch_size 6
```

Responses are leak-safe by construction: the prompt contains only the dialogue up to the current turn, the input-side user profile (when present), and the catalog metadata of the final top-1 track. Every submission regenerates all 80 responses against its own final ranking (no reuse or splicing of past responses).

### 6. Package and validate

```bash
python - <<'PY'
import json, zipfile
data = json.load(open("exp/inference/blindset_B/exp116_lgbm6_nocs_B.json", encoding="utf-8"))
json.dump(data, open("/tmp/prediction.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
with zipfile.ZipFile("exp/inference/blindset_B/submission_exp116_lgbm6_nocs_B.zip", "w", zipfile.ZIP_DEFLATED) as z:
    z.write("/tmp/prediction.json", "prediction.json")
PY
python scripts/validate_submission.py exp/inference/blindset_B/submission_exp116_lgbm6_nocs_B.zip
```

The validator checks the submission constraints mechanically: 80 records, exactly 20 ranked unique `predicted_track_ids` per row, non-empty responses, ZIP root = `prediction.json`, file name ≤ 63 chars; with `--catalog` it additionally verifies every track id exists in the official catalog.

## Evaluation alignment audit (workshop paper)

Code, inputs, and result tables for the paper *"Auditing Alignment among Item Relevance, Goal Progress, and LLM-Judged Responses in Music-CRS"*:

- Main audit script: [EDA/20260720_evaluation_alignment_audit.py](EDA/20260720_evaluation_alignment_audit.py). Its inputs are the official HF datasets plus the tracked run ledger [mcrs/experiments/results_ledger.jsonl](mcrs/experiments/results_ledger.jsonl); it outputs the released tables `EDA/tables/evaluation_alignment_*.csv` (tracked) and the summary [EDA/summary/20260720_evaluation_alignment_audit.md](EDA/summary/20260720_evaluation_alignment_audit.md). Reproduce with `python EDA/20260720_evaluation_alignment_audit.py` (seed 20260720, 5,000 bootstrap replicates by default). The response-corpus section additionally reads our local Blind A submission artifacts referenced from the ledger; rows whose artifacts are unavailable are listed in `EDA/tables/evaluation_alignment_response_exclusions.csv`.
- Pivot-intent detector manual audit: annotation guideline [docs/paper/pivot_intent_annotation_guideline.md](docs/paper/pivot_intent_annotation_guideline.md), sampling/metrics script [EDA/20260723_pivot_intent_annotation.py](EDA/20260723_pivot_intent_annotation.py), blind labels [EDA/annotation/](EDA/annotation/), metrics `EDA/tables/pivot_intent_manual_metrics.csv`, summary [EDA/summary/20260723_pivot_intent_manual_audit.md](EDA/summary/20260723_pivot_intent_manual_audit.md).
- Paper manuscript (markdown mirror): [docs/paper/paper_evaluation_alignment_audit.md](docs/paper/paper_evaluation_alignment_audit.md) ([日本語版](docs/paper/paper_evaluation_alignment_audit_ja.md)).


## Rule compliance

- Only the official `talkpl-ai/TalkPlayData-Challenge-*` datasets are used; no external data, no LFM-2B / listening-history data, in any stage.
- Candidates are always retrieved from the full track catalog (`track_split_types: [all_tracks]`); every submission row has exactly 20 ranked catalog track ids.
- No future-turn or ground-truth information is used in features, prompts, fallbacks, or validation; Blind splits are used for inference only.
- Response generation sees only current-turn-and-earlier dialogue, the input-side user profile, and catalog metadata.

## Acknowledgements

Built on the official [music-crs-baselines](https://github.com/nlp4musa/music-crs-baselines) provided by the challenge organizers (baseline runners, BM25/BERT retrieval modules, catalog/profile DBs are theirs). Datasets by [talkpl.ai](https://huggingface.co/talkpl-ai) / the Music-CRS organizers.

## Contact

Team Komekami (GitHub org `tmp-friends`).
