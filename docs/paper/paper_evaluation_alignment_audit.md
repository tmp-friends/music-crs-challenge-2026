# Auditing Alignment among Item Relevance, Goal Progress, and LLM-Judged Responses in Music-CRS

**Team:** Komekami

**Code:** <https://github.com/tmp-friends/music-crs-challenge-2026>

> **Submission note (not part of the manuscript):** Evaluation-alignment proposal; draft status: 2026-07-24 (language simplified and condensed to fit the page limit). The official RecSys Challenge 2026 Timeline currently shows that the paper deadline was extended from **20 July 2026 to 24 July 2026**, and the notification date from 3 August to 5 August. The Paper Submission Guidelines block on the same page still displays the old dates, so the authors must confirm the effective deadline in EasyChair before submission. Source: <https://www.recsyschallenge.com/2026/>.
>
> This draft targets the ACM RecSys Challenge 2026 workshop format: 4 pages of body text plus 1 page of references, in `acmart` `sigconf` double-column format.

## Abstract

The official evaluation of the RecSys Challenge 2026 Music-CRS task can disagree with itself. The task scores a ranked list of 20 tracks and a natural-language response per turn with a single composite of nDCG@20, catalog and lexical diversity, and an LLM judge. Auditing the released training data and the interim leaderboard, we find two internal conflicts. First, the accuracy target contradicts the dataset's own goal labels. When a listener explicitly asks for a different artist, the next ground-truth track still stays with an artist the session has already played in 76.0% of cases, 12.9 points more often than on other turns. The pattern sharpens when the dataset's own goal-progress label marks the preceding same-artist recommendation as unsuccessful. The next track then repeats that very artist in 85.8% of cases. A ranker that follows the user's request is therefore penalized by nDCG. Second, the response terms of the composite move independently of the recommendations. We froze one top-20 ranking on the blind leaderboard and rewrote only the responses. The composite rose by 0.0843, which equals the contribution of a +0.17 nDCG@20 gain. The composite cannot tell better recommendations from better-worded explanations. We recommend reporting goal adherence and ranking--response consistency alongside it. Team Komekami ranked 8th in the Industry Track and 16th among 40 teams.

**CCS Concepts:** • Information systems → Recommender systems; Evaluation of retrieval results.

**Keywords:** RecSys Challenge, Conversational Recommendation, Music Recommendation, Evaluation Metrics, LLM-as-a-Judge, Synthetic Data

## 1 Introduction

When a listener in the RecSys Challenge 2026 Music-CRS training data explicitly asks for a different artist, the next ground-truth track usually stays with an artist the session has already played, so a ranker that faithfully follows the request loses nDCG exactly where the user asked for change. This paper measures that tension, and a second one like it, inside the challenge's official evaluation.

Music-CRS scores, for each evaluation turn, a ranked list of 20 catalog tracks and a natural-language response with a single composite: nDCG@20, catalog and lexical diversity, and a Gemini-family LLM judge [1]. Our central claim is that item relevance, goal adherence, and judged response quality are related but different quantities that can pull a system in different directions.

Prior work has criticized conversational-recommendation evaluations that reduce success to matching one held-out item or utterance [3]; LLM judges scale well but exhibit verbosity bias [4] and favor their own generations [5]; and an earlier RecSys Challenge ranked submissions on accuracy alone, giving participants no incentive to pursue its stated diversity goals [6]. What has not been checked, to our knowledge, is whether a challenge composite disagrees *with itself*: whether its accuracy target contradicts the dataset's own goal labels, and how far its response terms move under a fixed ranking, both measurable on Music-CRS.

We ask two questions:

- **RQ1:** When a listener explicitly asks for a different artist, does the next ground-truth track actually change artists, and does the dataset's own goal-progress label register the conflict?
- **RQ2:** With the full ranked list frozen, how much do the official response metrics move on the leaderboard when only the way the response is written changes?

Our contribution is an evaluation-alignment audit:

- a full-training-split analysis of item--goal alignment with session-level confidence intervals and a manually validated pivot detector;
- a hash-verified comparison of five response styles under one fixed ranking, scored by the official Blind-A evaluator;
- reporting recommendations for conversational evaluation.

## 2 Task, Data, and Metrics

Given the dialogue up to the current user request and, when available, the user profile and the released metadata and embeddings, a system must return a ranked list of 20 track IDs drawn from the full catalog, together with a natural-language response that explains the recommendation. On the development split every recommendation turn is a target (8,000 turns). The blind splits score one turn per session.

### 2.1 TalkPlayData-Challenge

Music-CRS builds on TalkPlayData 2, a synthetic conversational music dataset generated by a pipeline of LLM agents [2]. A Listener agent, which knows the session goal and the user profile, converses with a Recsys agent, which sees the profile but not the goal. Each session is generated in the order `music[t-1]` → `progress/message[t]` → `music[t]`. The Listener labels whether the previous recommendation makes progress toward the goal and writes the next request before the Recsys agent picks the current track [2]. The challenge release contains 15,199 training and 1,000 development sessions, each with eight recommendation turns. The catalog holds 47,071 tracks, and turns 2--8 carry 106,393 progress labels.

Blind A and Blind B each contain 80 evaluation turns (one per session) with hidden relevant tracks. Blind A backed an interim leaderboard that returned official scores during the challenge. Blind B decided the final ranking. Blind B also removes the progress and reasoning fields and withholds the user identifier and profile for 40 cold-start cases, a distribution shift we revisit in Section 6.3.

### 2.2 Official metrics and audit scope

The composite has four parts [1]:

- **nDCG@20 (weight 0.50):** rewards ranking high the single *relevant track* attached to each turn (`music[t]`).
- **Catalog diversity (weight 0.10):** unique recommended tracks across the whole prediction file, as a fraction of the catalog.
- **Lexical diversity (weight 0.10):** corpus-level Distinct-2 over the responses.
- **LLM judge (weight 0.30):** personalization and explanation quality on a 1--5 scale, mapped linearly to [0,1] before aggregation.

The judge's model family is disclosed; its prompt and per-example scores are not. RQ1 uses only the released training split and track metadata, and treats the progress labels `MOVES_TOWARD_GOAL` and `DOES_NOT_MOVE_TOWARD_GOAL`, the Listener's private annotations, as synthetic artifacts of data generation, not as human satisfaction judgments. RQ2 uses official scores of Blind-A submissions from the open interim phase.

## 3 Submitted System

Our system separates track selection from response generation (Figure 1). The ranking path is a two-stage recommender cascade in the tradition of [11]: candidate generation with a learned first-stage ranker, then a reranker over the merged pool. It uses only official challenge data and always searches the full 47,071-track catalog. A final stage writes a response for the fixed recommendation.

![Overview of the submitted Music-CRS pipeline.](figures/system_pipeline.svg)

*Figure 1: The submitted pipeline.*

### 3.1 Candidate generation

For each target turn, the system sees the dialogue up to the current request and, when available, the user profile. An anchor stage fuses eight lexical signals into a 500-candidate pool by reciprocal-rank fusion: four BM25S [7] views over catalog text indexes and four exact-match rules for explicitly mentioned tracks, artists, albums, and tags. A first-stage LightGBM classifier trained on the 121,592 training turns then reorders the pool, and its top 20 anchor the candidate union. Twelve auxiliary sources each add up to 100 candidates from four families: session-local artist/album/tag continuity, BM25 neighbors from official training tasks, catalog entities mentioned in recent user text, and entity-to-track transitions learned from training sessions. After merging, deduplication, and removal of tracks already recommended in the session, the development pool averages 372 candidates per target and contains the relevant track for 54.16% of the 8,000 targets.

### 3.2 Learning to rank

Each candidate is described by 73 rank-only features: its rank and top-k membership in the anchor and in each source, cross-source agreement, and turn position. Content, embedding, and train-statistics features were deliberately excluded after their development gains repeatedly inverted on Blind A. As a side effect, the ranker never reads the user identifier. Because the neighbor and transition sources are built from training-split ground truth, this fusion stage is trained on the 8,000 development turns instead, avoiding self-leakage. Six LightGBM models with the LambdaRank objective [8, 9], identical except for the random seed, treat each session-turn as a ranking group with its single relevant track as the positive item. Averaging their scores damps the roughly ±0.02 retraining fluctuation of 80-turn blind scores. On Blind A, the official BM25 baseline scores nDCG@20 0.1938, the anchor alone 0.3869, and the full pipeline 0.4355.

### 3.3 Grounded response generation

Once the ranking is fixed, a hosted `claude-opus-4-8` model (no fine-tuning) receives only the dialogue so far, the available profile, and metadata of the top-ranked track. The prompt requires the explanation to stay within these inputs and forbids invented track attributes. Its style is the best-scoring bundle of Table 2. The generator cannot change the ranking, and all 80 responses are regenerated for every submission so that the text always matches its final top track. The ranking path is deterministic given trained models and indexes. Response wording depends on the hosted model.

On Blind B, the selected submission scored composite 0.4998 (nDCG@20 0.3528, LLM judge 4.30), ranking 8th in the Industry Track and 16th among 40 teams [10].

## 4 Audit Method

### 4.1 RQ1: does the ground truth follow artist-change requests?

By the generation order of Section 2.1, the progress label at turn `t` rates the previous recommendation, and the music row at `t` is the next relevant track, chosen after the current request. We join music rows to catalog metadata, comparing tracks on their sets of lowercased artist names (a multi-artist track matches on any shared name), and call the relevant track at `t` a *session-artist repeat* if it shares an artist with any earlier music turn in the session. For `t>=3`, we apply the same definition to the previous recommendation.

We detect explicit change requests (*pivots*) with three simple, case-insensitive lexical rules over the user message:

- **artist-targeted**: change phrases naming an artist, band, singer, or other person (“a different artist”, “a band I haven't heard”);
- **broad**: also accepts generic phrases (“something new”, “discover something different”);
- **core**: centered on “different”, “another”, and “someone else” (“another artist”, “by someone else”).

The three rules are alternative definitions rather than nested filters. Because the manual audit below gives the artist-targeted rule the highest precision, it is our primary rule, with the other two as sensitivity checks.

For each rule we report, within its pivot turns, the quantities of Table 1, including a *high-conflict* slice (pivot message, preceding same-artist recommendation, `DOES_NOT_MOVE_TOWARD_GOAL` label). Pivots concentrate in later turns, where artist repeats are also more common, so we compare the two groups at each turn and reweight both to the pooled turn distribution. Confidence intervals come from resampling whole sessions 5,000 times (seed 20260720). A logistic model with turn fixed effects and session-clustered errors serves as a secondary check. Code, public inputs, and result tables are released in the repository above.

To validate the rules, one annotator blindly labeled 200 stratified turns (50 broad-rule positives from the high-conflict slice, 50 other broad-rule positives, 100 broad-rule negatives) as `ARTIST_PIVOT`, `NOT_ARTIST_PIVOT`, or `UNCERTAIN`, seeing only the request and the preceding dialogue. Strata follow the widest (broad) rule, so reweighting each stratum to the 106,393-turn population yields precision, recall, and F1 for all three rules (session bootstrap, seed 20260723). We exclude `UNCERTAIN` cases from the binary metrics and report them as coverage.

### 4.2 RQ2: response metrics under a frozen ranking

We take one Blind-A submission (nDCG@20 0.4355, catalog diversity 0.0280), freeze all 20 predicted track IDs per turn, and swap in five response sets, regenerating all 80 responses each time (Table 2):

- **Initial**: the pipeline's original Qwen3.5-4B prompt;
- **Qwen 4B, revised**: the same generator, but suppressing apologies, forbidding invented attributes, and rotating opening styles;
- **Claude, short** (~75 words): generation moved to a hosted Claude model;
- **Claude, grounded long** (~120 words): adding metadata grounding, enthusiasm, and question endings;
- **Gemini Pro, grounded**: the grounded prompt applied to Gemini with extended reasoning.

We verify that the rankings are identical by hashing every ordered top-20 list, and split each composite change into its lexical and judge parts.

## 5 Results

### 5.1 RQ1: negative feedback does not redirect the next relevant track

Table 1 summarizes the audit. If the ground truth followed artist-change requests, repeats would be rarer on pivot turns; the opposite holds. Under the primary rule, the next relevant track is a session-artist repeat on 76.3% of the 15,126 pivot turns (`Next same` in Table 1). After the turn reweighting of Section 4.1, this rate is 76.0%, against 63.1% on non-pivot turns. The table reports the gap as the +12.9-point turn-adjusted difference. The gap is positive at every turn taken separately (9.0--17.1 points), and the secondary logistic check gives an odds ratio of 1.90 (95% CI: 1.81--2.01).

The progress labels do register the conflict. On pivot turns at `t>=3`, the label rating the previous turn's recommendation reads `MOVES_TOWARD_GOAL` in 56.3% of cases when that recommendation was itself a session-artist repeat, against 78.5% when it introduced a new artist. The relevant track chosen right after this judgment ignores it: in the high-conflict slice of Table 1 (pivot message, preceding same-artist recommendation, `DOES_NOT_MOVE_TOWARD_GOAL` label), 4,529 of 5,278 relevant tracks (85.8%; 95% CI: 84.7--86.8) repeat the immediately preceding artist.

**Table 1: Pivot audit under three lexical rules. `Next same`: share of pivot turns whose next relevant track repeats a session artist. `Turn-adj. diff`: pivot minus non-pivot gap after reweighting to the pooled turn distribution (pp; 95% session-bootstrap CI). `Prior MOVES same/diff`: share of preceding recommendations labeled `MOVES_TOWARD_GOAL`, by whether they repeated a session artist (`t>=3`). `Conflict repeats previous`: share of high-conflict turns repeating the immediately preceding artist (n in parentheses).**

| Pivot rule | Pivot turns | Next same | Turn-adj. diff | Prior `MOVES` same/diff | Conflict repeats previous |
|---|---:|---:|---:|---:|---:|
| **Artist-targeted (primary)** | 15,126 | 76.3% | +12.9 pp [11.8, 14.0] | 56.3% / 78.5% | 85.8% (n=5,278) |
| Broad discovery/change | 10,982 | 77.0% | +12.6 pp [11.5, 13.7] | 43.2% / 69.1% | 85.9% (n=4,899) |
| Core change | 7,147 | 80.1% | +16.0 pp [14.7, 17.2] | 46.1% / 71.7% | 86.2% (n=3,227) |

The manual audit yields 194 classifiable cases out of the 200 annotated turns, covering 95.4% of the turn population after stratum reweighting. Against these labels, the primary rule has precision 0.992 (95% CI: 0.970--1.000), recall 0.322 (0.225--0.435), and F1 0.486; the broad rule reaches 0.904 and 0.218, the core rule 0.983 and 0.160. All three rules are high-precision but low-recall: false positives are too rare to explain the repeat rates above, and low recall limits coverage, not precision (Section 6.3).

### 5.2 RQ2: large composite changes under an identical ranking

Table 2 reports official Blind-A scores for five response sets over one identical ranking, so all movement comes from lexical diversity and the LLM judge. The best bundle (Claude, grounded long) improves the composite by 0.0843 over the initial responses: 0.0638 from the normalized judge term (75.6%) and 0.0206 from lexical diversity (24.4%; components rounded). Under the official weights, this equals the composite effect of a +0.1687 nDCG@20 improvement. The point is not that the two changes are equally valuable to users, but that the composite cannot tell them apart.

**Table 2: Blind-A response variants with all 20 ranked tracks fixed (bundles: Section 4.2).**

| Response variant | Lexical diversity | LLM judge | Composite |
|---|---:|---:|---:|
| Initial response | 0.5925 | 4.05 | 0.5086 |
| Qwen 4B, revised | 0.6605 | 4.20 | 0.5266 |
| Claude, short | **0.8133** | 4.30 | 0.5494 |
| Claude, grounded long | 0.7983 | **4.90** | **0.5929** |
| Gemini Pro, grounded | 0.7204 | 4.30 | 0.5401 |

## 6 Discussion

### 6.1 The metrics measure different things

Two gaps emerge. The first separates item accuracy from goal adherence. A ranker that faithfully follows the pivot misses the positive (Table 1). The same holds in training. Our learning-to-rank positives (Section 3.2) are these nDCG targets, so any system optimized for the composite inherits the repetition pattern by construction. The data pipeline suggests why: each next track is drawn from a pool derived from one listening session [2], and such pools are internally coherent, so the pool-level target can contradict the text-level request. The second gap separates response quality from recommendation correctness. Response scores moved substantially under an identical ranking (Table 2). Neither gap makes any single metric invalid. The risk is reading their weighted sum as one measure of conversational quality. Blind artist diversification is no fix either. In our development experiments, demoting same-artist candidates on pivot turns lowered nDCG, and the promoted alternatives usually violated the user's remaining constraints.

### 6.2 Recommendations for conversational evaluation

We suggest three complementary reporting practices to challenge organizers and benchmark builders.

1. **Report goal adherence separately.** For explicit pivots, check whether the recommendation respects the requested change (artist, genre, era, tempo, novelty) instead of inferring success only from one held-out item. Our released pivot detector already implements the artist case.
2. **Score ranking--response consistency.** Give the judge the recommended track's metadata and have it score whether the explanation matches that track, separately from prose quality.
3. **Calibrate the judge and publish slices.** Validate the judge on a stratified human-annotated subset (short/long responses, cold/warm users, matched/mismatched recommendations), and break scores out for pivot turns and prior-artist targets.

These additions complement rather than replace nDCG and LLM judging, echoing earlier calls to make beyond-accuracy goals operational [6].

### 6.3 Limitations

Our audit is observational and the data are synthetic, so we estimate neither human satisfaction nor causal effects. Turn adjustment does not remove unmeasured session or intent differences. The manual intent audit has a single annotator, and the English-only lexical rules have low recall, so they characterize high-precision slices, not how common artist pivots are. On RQ2, the five bundles change several factors at once and share one frozen ranking. Blind A has only 80 targets, and the Blind-B distribution shift further cautions against treating Blind-A differences as stable effects. Human annotation of response grounding is the natural follow-up.

## 7 Conclusion

Item relevance, goal adherence, and judged response quality in Music-CRS are related but not interchangeable: the ground-truth track repeats the just-rejected artist in 85.8% of the sharpest conflict cases, and rewriting the responses alone under a frozen ranking moved the composite by 0.0843. Reporting goal adherence and ranking--response consistency alongside it would make progress easier to interpret.

## References

[1] RecSys Challenge 2026. 2026. “Conversational Music Recommendation.” <https://www.recsyschallenge.com/2026/>.

[2] Keunwoo Choi, Seungheon Doh, and Juhan Nam. 2025. “TalkPlayData 2: An Agentic Synthetic Data Pipeline for Multimodal Conversational Music Recommendation.” arXiv:2509.09685. <https://arxiv.org/abs/2509.09685>.

[3] Xiaolei Wang, Xinyu Tang, Wayne Xin Zhao, Jingyuan Wang, and Ji-Rong Wen. 2023. “Rethinking the Evaluation for Conversational Recommendation in the Era of Large Language Models.” In *Proceedings of EMNLP 2023*, 10052--10065. <https://arxiv.org/abs/2305.13112>.

[4] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. 2023. “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.” In *Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track*. <https://arxiv.org/abs/2306.05685>.

[5] Arjun Panickssery, Samuel R. Bowman, and Shi Feng. 2024. “LLM Evaluators Recognize and Favor Their Own Generations.” In *Advances in Neural Information Processing Systems 37*. <https://arxiv.org/abs/2404.13076>.

[6] Lucien Heitz, Oana Inel, and Sanne Vrijenhoek. 2024. “Recommendations for the Recommenders: Reflections on Prioritizing Diversity in the RecSys Challenge.” In *Proceedings of the ACM RecSys Challenge 2024*, 22--26. <https://doi.org/10.1145/3687151.3687155>.

[7] Xing Han Lù. 2024. “BM25S: Orders of Magnitude Faster Lexical Search via Eager Sparse Scoring.” arXiv:2407.03618. <https://arxiv.org/abs/2407.03618>.

[8] Guolin Ke, Qi Meng, Thomas Finley, et al. 2017. “LightGBM: A Highly Efficient Gradient Boosting Decision Tree.” In *Advances in Neural Information Processing Systems 30*. <https://dl.acm.org/doi/10.5555/3294996.3295074>.

[9] Christopher J. C. Burges. 2010. “From RankNet to LambdaRank to LambdaMART: An Overview.” Microsoft Research Technical Report MSR-TR-2010-82. <https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/>.

[10] Music-CRS Challenge 2026. 2026. “Final Results: Blind-Dataset-B.” <https://nlp4musa.github.io/music-crs-challenge/results.html>.

[11] Maksims Volkovs, Himanshu Rai, Zhaoyue Cheng, Ga Wu, Yichao Lu, and Scott Sanner. 2018. “Two-stage Model for Automatic Playlist Continuation at Scale.” In *Proceedings of the ACM Recommender Systems Challenge 2018*. <https://doi.org/10.1145/3267471.3267480>.
