# Music-CRS におけるアイテム関連性・目標進捗・LLM評価応答の整合性監査

**チーム:** Komekami

**コード:** <https://github.com/tmp-friends/music-crs-challenge-2026> [TODO: リポジトリが公開され、提出システムと一致することを確認]

> **投稿メモ（論文本文には含めない）:** 評価整合性分析案、ドラフト基準日: 2026-07-24（ページ制限に収めるため表現を平易化・圧縮）。RecSys Challenge 2026 公式サイトの Timeline では、論文投稿締切が **2026年7月20日から7月24日へ延長**され、採否通知も8月3日から8月5日へ変更されている。一方、同ページの Paper Submission Guidelines 欄には旧日付が残っているため、投稿前に EasyChair 上の有効な締切を必ず確認すること。出典: <https://www.recsyschallenge.com/2026/>。
>
> 本稿は ACM RecSys Challenge 2026 Workshop の書式、すなわち本文4ページ＋参考文献1ページ、`acmart` の `sigconf` による二段組を想定した日本語確認用ドラフトである。正式投稿には英語版を用いる。角括弧内の TODO は投稿前にすべて解消すること。

## 要旨

RecSys Challenge 2026 Music-CRS の公式評価は、それ自身と食い違い得る。このタスクは、各ターンの推薦20曲のランキングと自然言語応答を、nDCG@20・catalog/lexical diversity・LLM judge からなる単一の composite で採点する。公開された訓練データと interim leaderboard を監査し、二つの内部矛盾を見出した。第一に、精度の正解はデータセット自身の goal ラベルと矛盾する。リスナーが明示的に別の artist を求めても、次の正解 track は76.0%のケースでセッション内で既に流れた artist にとどまる。これは他のターンより12.9ポイント高い。このパターンは、直前の same-artist 推薦がデータセット自身の goal-progress ラベルで不成功とされた場合に、いっそう強まる。その場合、次の track は85.8%のケースでまさにその artist を繰り返す。したがってユーザーの要求に従う ranker は nDCG で罰される。第二に、composite の応答項は推薦とは独立に動く。blind leaderboard 上で top-20 ランキングを凍結し、応答だけを書き直した。composite は0.0843上昇した。これは nDCG@20 +0.17 改善の寄与に相当する。composite は、より良い推薦と、より上手く書かれた説明を区別できない。そこで composite と併せて、goal への追従とランキング--応答整合性の報告を推奨する。チーム Komekami は Industry Track 8位、全40チーム中16位であった。

**CCS Concepts:** • Information systems → Recommender systems; Evaluation of retrieval results.

**キーワード:** RecSys Challenge、対話型推薦、音楽推薦、評価指標、LLM-as-a-Judge、合成データ

## 1 はじめに

RecSys Challenge 2026 Music-CRS の訓練データでは、リスナーが明示的に別の artist を求めても、次の正解 track はたいていセッション内で既に流れた artist にとどまる。したがって要求に忠実に従う ranker は、ユーザーが変化を求めたまさにそのターンで nDCG を失う。本稿は、この緊張関係と、それに似たもう一つの緊張関係を、Challenge の公式評価の内側で測定する。

Music-CRS は各評価ターンについて、カタログ楽曲20件のランキングと自然言語応答を単一の composite で採点する。内訳は nDCG@20、catalog・lexical diversity、Gemini 系 LLM judge である [1]。本稿の中心的主張は、アイテム関連性、goal への追従、judge による応答品質は、関連はするが別々の量であり、システムを異なる方向へ引っ張り得るというものである。

先行研究は、held-out のアイテムや発話との一致だけで成功を測る対話型推薦の評価を批判し [3]、LLM judge が verbosity bias を示すこと [4]、自己の生成文を優遇すること [5] を報告し、過去の RecSys Challenge が accuracy 単独で順位付けを行い、掲げた多様性目標を追う incentive を参加者に与えなかったことを示してきた [6]。しかし我々の知る限り、Challenge の composite が*それ自身と*食い違うか（すなわち精度の正解がデータセット自身の goal ラベルと矛盾するか、そしてランキング固定下で応答項がどこまで動くか）は検証されておらず、Music-CRS ではその両方を一つのタスク上で測定できる。

本研究の問いは二つである。

- **RQ1:** リスナーが明示的に別の artist を求めたとき、次の正解 track は実際に artist を変えるのか。また、データセット自身の goal-progress ラベルはこの衝突を捉えているか。
- **RQ2:** ランキング全体を固定したとき、応答の書き方だけを変えると公式の応答指標は leaderboard 上でどの程度動くか。

貢献は評価整合性監査であり、次の三つからなる。

- session 単位の信頼区間と人手検証済み pivot 検出器による、訓練データ全体のアイテム--goal 整合性分析
- 一つのランキングをハッシュ検証付きで固定し、公式 Blind-A 評価で採点した5種類の応答スタイルの比較
- 対話型評価への報告方法の提言

## 2 タスク、データ、指標

現在のユーザー要求までの対話と、利用可能な場合は user profile および公開 metadata・embedding を入力として、システムは全カタログから選んだ track ID の ranked list 20件と、推薦を説明する自然言語応答を返す。development split では全推薦ターン（8,000件）が評価対象であり、blind split では各セッション1ターンが採点される。

### 2.1 TalkPlayData-Challenge

Music-CRS は、LLM エージェントのパイプラインで生成された合成対話音楽データセット TalkPlayData 2 に基づく [2]。session goal と user profile を知る Listener エージェントが、profile は見られるが goal は知らされない Recsys エージェントと会話する。各セッションは `music[t-1]` → `progress/message[t]` → `music[t]` の順で生成される。すなわち Listener が直前の推薦が goal へ前進しているかをラベル付けして次の要求を書き、その後に Recsys エージェントが現在の track を選ぶ [2]。Challenge のリリースは、各8回の推薦ターンを持つ訓練15,199セッションと development 1,000セッションを含む。カタログは47,071曲で、turn 2--8 には106,393件の progress ラベルが付く。

Blind A と Blind B はそれぞれ80評価ターン（1セッションにつき1件）を含み、relevant track は非公開である。Blind A は challenge 期間中に公式スコアを返す interim leaderboard を担い、Blind B が最終順位を決めた。Blind B ではさらに progress・reasoning フィールドが削除され、40件の cold-start ケースでは user identifier と profile も提供されない。この分布シフトは6.3節で再訪する。

### 2.2 公式指標と監査範囲

composite は4指標からなる [1]。

- **nDCG@20（重み0.50）:** 各ターンに1つ付く *relevant track*（`music[t]`）を上位に置くほど高い。
- **Catalog diversity（重み0.10）:** 提出ファイル全体のユニーク推薦曲数をカタログ総数で割った値。
- **Lexical diversity（重み0.10）:** 応答コーパス全体の Distinct-2。
- **LLM judge（重み0.30）:** personalization と explanation quality を1--5で採点し、線形に [0,1] へ写像して統合。

judge のモデル系列は公開されているが、prompt とサンプル別スコアは非公開である。RQ1 は公開済みの訓練 split と track metadata のみを用い、progress ラベル `MOVES_TOWARD_GOAL` / `DOES_NOT_MOVE_TOWARD_GOAL`、すなわち Listener の内部注釈は、データ生成時の合成 artifact として扱い、人間の満足度評価とはみなさない。RQ2 は公開 interim phase で返された Blind-A 提出の公式スコアを用いる。

## 3 提出システム

本システムは楽曲選択と応答生成を分離する（図1）。ランキング経路は two-stage 推薦システムのカスケード [11] であり、学習済みの第一段 ranker を含む候補生成と、統合プールに対する reranker からなる。Challenge 公式データのみを使い、常に47,071曲の全カタログを検索する。最後の段が、確定した推薦に対する応答を生成する。

![提出した Music-CRS pipeline の概要。](figures/system_pipeline.svg)

*図1: 提出した pipeline。*

### 3.1 候補生成

各対象ターンで、システムは現在の要求までの対話と、利用可能な場合は user profile を見る。anchor 段は8個の語彙シグナルを reciprocal-rank fusion で500候補のプールに統合する。その内訳は、catalog テキスト index に対する BM25S [7] の4 view と、明示的に言及された track・artist・album・tag の完全一致ルール4種である。訓練121,592ターンで学習した第一段 LightGBM 分類器がこのプールを並べ替え、その top 20 が候補 union の anchor になる。補助 source は4系統12個で、各 source が最大100候補を加える。すなわち、セッション内の artist/album/tag 継続、公式訓練タスクの BM25 近傍、直近のユーザー発話に言及された catalog entity、訓練セッションから学習した entity→track 遷移である。統合・重複除去し、セッション内で推薦済みの track を除くと、development の候補プールは平均372件で、8,000ターンの54.16%に relevant track を含む。

### 3.2 Learning to rank

各候補は rank のみの73特徴で記述する。すなわち、anchor と各 source での順位・top-k への包含、source 間の一致度、ターン位置である。content・embedding・訓練統計の特徴量は、development での改善が Blind A で繰り返し反転したため意図的に除外した。副作用として、ranker は user identifier を一切読まない。近傍・遷移 source は訓練 split の正解から構築されているため、この統合段は代わりに development 8,000ターンで学習し、自己リークを避ける。random seed のみが異なる6個の LightGBM モデルが LambdaRank 目的関数 [8, 9] を最適化する。各 session-turn を ranking group、その唯一の relevant track を正例とし、スコアを平均することで、80ターンの blind スコアが再学習だけで約±0.02揺れる変動を抑える。Blind A では、公式 BM25 baseline の nDCG@20 は0.1938、anchor 単体は0.3869、full pipeline は0.4355である。

### 3.3 推薦に基づく応答生成

ランキング確定後、fine-tuning していない hosted `claude-opus-4-8` に、現在までの対話、利用可能な profile、top-ranked track の metadata のみを与える。prompt はこれらの入力の範囲内で説明することを求め、根拠のない track 属性の記述を禁じる。その文体は表2で最高スコアの bundle である。生成器はランキングを変更できない。また、submission ごとに80件の応答をすべて再生成し、文章が常に最終的な top track と一致するようにする。ランキング経路は学習済みモデルと index を固定すれば決定的だが、応答の文言は hosted モデルに依存する。

採用した Blind-B 提出は composite 0.4998（nDCG@20 0.3528、LLM judge 4.30）で、Industry Track 8位、全40チーム中16位であった [10]。

## 4 監査方法

### 4.1 RQ1: 正解は artist 変更要求に従うのか

2.1節の生成順序より、turn `t` の progress ラベルは直前の推薦を評価し、turn `t` の music 行は現在の要求の後に選ばれた次の relevant track である。music 行を catalog metadata に結合し、track 同士は小文字化した artist 名の集合で比較して（複数 artist の track はいずれかの名前が一致すればマッチ）、turn `t` の relevant track がセッション内のそれ以前の music turn と artist を共有する場合を *session-artist repeat* と呼ぶ。`t>=3` では直前の推薦にも同じ定義を適用する。

明示的な変更要求（*pivot*）は、ユーザー発話に対する単純な（大文字小文字を区別しない）三つの語彙ルールで検出する。

- **artist-targeted**: artist・band・singer など人物を指す変更表現（“a different artist”、“a band I haven't heard”）
- **broad**: “something new”、“discover something different” のような一般的表現も許容
- **core**: “different”・“another”・“someone else” が中心（“another artist”、“by someone else”）

三つのルールは包含関係にある絞り込みではなく独立した定義であり、後述の人手監査で precision が最も高かった artist-targeted ルールを主ルール、他の二つを感度チェックとする。

各ルールについて、その pivot ターン内で表1の各量を集計する。これには *high-conflict* slice、すなわち pivot 発話・直前の same-artist 推薦・`DOES_NOT_MOVE_TOWARD_GOAL` ラベルが重なるケースを含む。pivot は後半ターンに集中し、artist の繰り返しも後半ほど多いため、両群を各ターンで比較し、両群合算（pooled）のターン分布に重み付けし直す。信頼区間はセッション全体の再標本化5,000回（seed 20260720）で求め、turn 固定効果とセッションクラスタ頑健誤差を持つロジスティックモデルを二次チェックとする。コード・公開入力・結果テーブルは上記リポジトリで公開する。

ルールの妥当性確認として、1名のアノテータが層化抽出した200ターン（high-conflict slice の broad 陽性50、その他の broad 陽性50、broad 陰性100）を、要求とそれ以前の対話のみを見て `ARTIST_PIVOT`・`NOT_ARTIST_PIVOT`・`UNCERTAIN` にブラインド判定した。層は最も広い broad ルールに基づいて定義するため、各層を106,393ターンの母集団へ重み戻しすれば、三つのルールすべての precision・recall・F1 が推定できる（セッション bootstrap、seed 20260723）。`UNCERTAIN` は二値指標から除外し、coverage として報告する。

### 4.2 RQ2: ランキングを凍結した応答指標

一つの Blind-A 提出（nDCG@20 0.4355、catalog diversity 0.0280）について、各ターンの predicted track ID 20件をすべて凍結し、5種類の応答セットに差し替える。差し替えごとに80応答をすべて再生成する（表2）。

- **Initial**: pipeline 本来の Qwen3.5-4B prompt
- **Qwen 4B, revised**: 同じ生成モデルのまま、謝罪を抑制し、属性の捏造を禁止し、書き出しをローテーション
- **Claude, short**（約75語）: 生成を hosted Claude モデルへ移行
- **Claude, grounded long**（約120語）: さらに metadata grounding・熱量・疑問形の締めを追加
- **Gemini Pro, grounded**: grounded prompt を extended reasoning 付きの Gemini に適用

全 top-20 リストのハッシュでランキングの同一性を検証し、composite の変化を lexical 項と judge 項に分解する。

## 5 結果

### 5.1 RQ1: 負のフィードバックは次の正解を転換させない

表1に監査の要約を示す。正解が artist 変更要求に従うなら、pivot ターンでは repeat はむしろ稀になるはずだが、実際は逆である。主ルールでは、pivot ターン15,126件の76.3%で次の relevant track が session-artist repeat だった（表1の `Next same`）。§4.1のターン再重み付け後、この割合は76.0%で、非 pivot ターンの63.1%を上回る。表1はこの差を +12.9ポイントの turn-adjusted difference として報告する。差はターン別に見ても全ターンで正であり（9.0--17.1ポイント）、二次チェックのロジスティックモデルは odds ratio 1.90（95% CI: 1.81--2.01）を与える。

progress ラベル自体はこの衝突を捉えている。pivot ターン（`t>=3`）では、直前ターンの推薦を評価するラベルが `MOVES_TOWARD_GOAL` となる割合は、その推薦自身が session-artist repeat だった場合56.3%で、新規 artist を提示した場合の78.5%より低い。しかしこの判定の直後に選ばれる relevant track はそれを無視する。表1の high-conflict slice（pivot メッセージ・直前の same-artist 推薦・`DOES_NOT_MOVE_TOWARD_GOAL` ラベル）では、5,278件中4,529件（85.8%、95% CI: 84.7--86.8）が直前の artist をそのまま繰り返す。

**表1: 三つの語彙ルールによる pivot 監査。`Next same`: pivot ターンで次の relevant track が session artist を繰り返す割合。`Turn-adj. diff`: 合算（pooled）のターン分布へ重み付けし直した後の pivot と非 pivot の差（pp、95% session-bootstrap CI）。`Prior MOVES same/diff`: 直前の推薦が session artist を繰り返した／新規だった場合に `MOVES_TOWARD_GOAL` とされた割合（`t>=3`）。`Conflict repeats previous`: high-conflict slice で直前の artist を繰り返す割合（括弧内は n）。**

| Pivot ルール | Pivot turns | Next same | Turn-adj. diff | Prior `MOVES` same/diff | Conflict repeats previous |
|---|---:|---:|---:|---:|---:|
| **Artist-targeted（主）** | 15,126 | 76.3% | +12.9 pp [11.8, 14.0] | 56.3% / 78.5% | 85.8% (n=5,278) |
| Broad discovery/change | 10,982 | 77.0% | +12.6 pp [11.5, 13.7] | 43.2% / 69.1% | 85.9% (n=4,899) |
| Core change | 7,147 | 80.1% | +16.0 pp [14.7, 17.2] | 46.1% / 71.7% | 86.2% (n=3,227) |

人手監査では、アノテーションした200ターン中194件が二値判定可能で、層別再重み付け後の母集団の95.4%をカバーする。このラベルに対し、主ルールは precision 0.992（95% CI: 0.970--1.000）、recall 0.322（0.225--0.435）、F1 0.486。broad ルールは0.904・0.218、core ルールは0.983・0.160である。三ルールとも high-precision／low-recall であり、false positive は上記の repeat 率を説明できないほど稀で、低 recall が制限するのは coverage であって precision ではない（§6.3）。

### 5.2 RQ2: ランキングが同一でも composite は大きく動く

表2に、同一ランキング上の5種類の応答セットに対する公式 Blind-A スコアを示す。変動はすべて lexical diversity と LLM judge に由来する。最良の bundle（Claude, grounded long）は initial response より composite を0.0843改善し、内訳は正規化後の judge 項が0.0638（75.6%）、lexical 項が0.0206（24.4%。内訳は丸め値）である。公式の重みの下で、これは nDCG@20 を +0.1687 改善した場合と同じ composite 効果に相当する。要点は、両者がユーザーにとって等価だということではなく、composite がこの二つを区別できないことである。

**表2: 20件の ranked track をすべて固定した Blind-A 応答 variant（bundle の定義は4.2節）。**

| Response variant | Lexical diversity | LLM judge | Composite |
|---|---:|---:|---:|
| Initial response | 0.5925 | 4.05 | 0.5086 |
| Qwen 4B, revised | 0.6605 | 4.20 | 0.5266 |
| Claude, short | **0.8133** | 4.30 | 0.5494 |
| Claude, grounded long | 0.7983 | **4.90** | **0.5929** |
| Gemini Pro, grounded | 0.7204 | 4.30 | 0.5401 |

## 6 議論

### 6.1 各指標は別々のものを測っている

二つのギャップが浮かび上がる。第一はアイテム精度と goal 追従の間である。pivot に忠実に従う ranker は正例を外す（表1）。これは学習にもそのまま組み込まれる。3.2節の learning-to-rank の正例はまさにこの nDCG 正解なので、composite に最適化されたシステムは構造的にこの繰り返しパターンを継承する。データ生成 pipeline はその理由を示唆する。次の track は一つのリスニングセッションに由来する pool から選ばれ [2]、pool は内部的に一貫しているため、pool レベルの正解はテキストレベルの要求と食い違い得る。第二は応答品質と推薦の正しさの間である。同一ランキングの下で応答スコアは大きく動いた（表2）。どちらのギャップも個々の指標を無効にはしないが、その加重和を対話品質の単一の尺度として読むことにはリスクがある。また artist の機械的な多様化は解決策にならない。我々の development 実験では、pivot ターンで same-artist 候補を降格させると nDCG が下がり、繰り上がった代替の多くはユーザーがなお保持している制約に反していた。

### 6.2 対話型評価への提言

challenge organizer と benchmark 構築者に向けて、補完的な三つの報告方法を提案する。

1. **goal 追従を別建てで報告する。** 明示的な pivot について、held-out アイテム一つから成功を推定せず、要求された変更（artist、genre、era、tempo、novelty）を推薦が守っているかを確認する。我々が公開した pivot detector は artist の場合を既に実装している。
2. **ランキングと応答の整合性を採点する。** judge に推薦 track の metadata を与え、説明がその track と一致しているかを、文章品質とは分けて採点させる。
3. **judge を較正し slice を公開する。** 層化した人手評価サブセット（短文／長文、cold／warm ユーザー、整合／不整合推薦）で judge を検証し、pivot ターンと既出 artist 正解の別にスコアを分けて報告する。

これらは nDCG や LLM judge の置き換えではなく補完であり、beyond-accuracy の目標を実効的にせよという過去の提言とも一致する [6]。

### 6.3 限界

本監査は観察的でデータは合成であるため、人間の満足も因果効果も推定せず、ターン調整は未観測のセッション差・意図差を除去しない。人手の意図監査はアノテータ1名であり、英語表現のみを対象とする語彙ルールは recall が低いため、特徴付けできるのは high-precision な slice であって artist pivot の頻度ではない。RQ2 では、5つの bundle が複数要因を同時に変え、すべて単一の凍結ランキングを共有する。また Blind A は80ターンのみであり、Blind-B の分布シフトも、Blind-A の差を安定した効果とみなすことを妨げる。応答 grounding の人手アノテーションが自然な次の課題である。

## 7 結論

Music-CRS のアイテム関連性・goal 追従・judge による応答品質は、関連はするが交換可能ではない。正解 track は最も衝突の強い slice の85.8%で直前に拒まれた artist を繰り返し、ランキングを凍結したまま応答を書き直すだけで composite は0.0843動いた。これと併せて goal 追従とランキング--応答整合性を報告すれば、進歩はより解釈しやすくなる。

## 参考文献

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

---

## 著者向け TODO（論文本文には含めない）

1. ✅ (2026-07-24) 著者情報は提出原稿側に記載済み（本ミラーには載せない）。team name Komekami は abstract と本文に記載済み。登録チーム名と正確に一致するか最終確認する。
2. <https://github.com/tmp-friends/music-crs-challenge-2026> が公開済みで、sanitized され、Blind B の `lgbm6_nocs_B` system から `prediction.json` までを再現できることを確認する。
3. **2026年7月24日**の締切を EasyChair で確認する。公式 Timeline は延長後の日付を示す一方、同じページの guidelines 欄には7月20日が残っている。
4. 残る過去 artifact 1件（`submission_exp014_ab_s06_t500fb_A.zip`）を、可能なら元の Codabench submission record または backup から取得して ledger のパスへ配置し、camera-ready 前に公開 audit script を再実行する。当時の採点済み artifact の代わりに、新規生成した response file を使用しない。
5. 表2の5 variant について、response grounding、personalization、verbosity を比較する小規模な人手監査を追加する。実施しない場合、judge score の上昇が人間向け response の改善を意味するとは主張しない。
6. ✅ (2026-07-24) `figures/system_pipeline.pdf` を SVG から cairosvg で生成済み。コンパイル後の PDF で文字・矢印とも正常に描画されることを確認した。camera-ready で font embedding を ACM 要件に厳密に合わせる必要があれば drawio からの再書き出しを検討する。
7. ✅ (2026-07-24) `latex/main.tex` として `\documentclass[sigconf]{acmart}` へ変換し、tectonic でコンパイル済み（`latex/main.pdf`）。**実測: 本文は4ページ目で完結、参考文献は5ページ目前半で終了 → 4P+1P を約3/4ページの余裕付きで充足**。著者情報を入れた後に再確認する。*(2026-07-25 更新: abstract 改稿で本文が約6行5ページ目へはみ出したため、図1を `\columnwidth` 0.92→0.78 に縮小して結論を4ページ目に戻した。余裕は最小限で図中ラベルも小さくなったため、camera-ready で可読性を再確認するか、本文を約6行削って図サイズを戻す。)* *(2026-07-25 追記: Introduction を全面改稿 — cold open、composite のインライン化、「composite がそれ自身と食い違う」というギャップ文、結果プレビュー段落、貢献 (i)–(iii) の列挙とチーム順位 — し、2.2節の指標箇条書きを散文化（同日、著者の好みにより箇条書きへ復元）、本文数文を圧縮。現時点のコンパイルでは結論の末尾約4行が5ページ目にはみ出している。著者判断により、全節のブラッシュアップ完了後に最終的なページ詰めを行う。)* *(2026-07-25 追記2: 著者承認済みの短縮 abstract（約165語、1文1情報）を LaTeX と md 両版に適用。空いたスペースで結論が4ページ目に完全に収まり、参考文献は4ページ目から始まり5ページ目で終了。図1も `\columnwidth` 0.92 に復元し可読性懸念を解消。4P+1P を余裕付きで再充足。)* *(2026-07-25 追記3: §2 をブラッシュアップ — 生成順序を§2.1へ移動し Listener/Recsys の情報非対称を具体化（§4.1は参照のみに）、ラベル数の由来（15,199×7）を明示、Blind A/B の役割（スコア返却ありの interim leaderboard vs 最終順位決定）を明記、judge 正規化を具体化（1–5を[0,1]へ線形写像）、catalog diversity に「提出ファイル全体」を明示、監査スコープ文を§2.2へ一本化。§2.2の箇条書きは著者の好みで維持。ページ状況: 著者依頼の§2.1冗長削減後、5ページ目に残るのは結論の末尾約3行のみ。全節ブラッシュアップ後の最終パスでページ詰めを行う。)* *(2026-07-25 追記4: §2見出し直下にタスク定義のリード段落を追加。本文増加で Table 1（`table*`）が参考文献より後のフロート専用5ページ目へ流れたため、フロートのソース位置を§4冒頭へ前倒しして修正——両表は§5の隣の4ページ目に配置。現在のレイアウトは §7 結論と参考文献が5ページ目を共有しており、最終パスで結論を4ページ目へ戻す詰めが必要。)* *(2026-07-25 追記5: §4–§6 をブラッシュアップ。§4.1: 公開 regex に基づくルール別の実例フレーズを追加、artist 照合を「小文字化した artist 名集合（複数 artist はいずれか一致）」と明示、「共通のターン分布」を pooled ターン分布に訂正（表1キャプションも）、人手監査の層化設計を明確化（全層を最も広い broad ルールで定義するため三ルールすべての precision/recall が推定可能）。§4.2: 現在形へ統一し、5 bundle を個別に列挙（initial = pipeline 本来の Qwen3.5-4B prompt / revised / Claude 約75語・約120語 / Gemini Pro grounded）——表2キャプションの参照先不履行を解消。§5.1: 記述的スコープの段落を圧縮。§5.2: 内訳の丸め注記を追加、安定性チェックを「表2外の提出物の再採点」と明示し bundle 効果0.0843と再採点差0.0075の桁差を対置、生成モデル系列相関の文は削除。§6.1: 繰り返しパターンの機構仮説（次 track はリスニングセッション由来 pool から選択、choi2025）を追加し、根拠のなかった「多様化では直せない」を development 実験の実測（same-artist 降格で nDCG 低下・制約違反の代替）に置換。§6.2: 提言の宛先を organizer / benchmark 構築者と明示、公開 pivot detector が artist ケースを実装済みと付記。§6.3: 語彙ルールの英語限定と RQ2 の単一凍結ランキングの限界を追加。ページ状況: 本文増加で4/5ページ境界が §6.2 の項目1に移動——本文は4ページ目を約30行超過し、参考文献は5ページ目内で開始・終了（計5ページ）。最終的な 4P 詰めは著者対応のまま保留。)* *(2026-07-25 追記6: §4–§6 の冗長削減パス（洗い出した候補を全適用）。削除: §5.1 の記述的スコープ段落（§1・§6.3 と重複）、§4.1 の (i)–(iii) 列挙（表1キャプションと重複。high-conflict の定義は本文に残置）、§4.2 の再提出予告句（§5.2 が自己完結で導入）、§5.1 の「表1はルール一致要求を記述」文と broad ルール防御文（スコープ主張は §6.3 が保持）、§5.2 の「推薦を一つも変えずに」、§6.3 の「指標整合性の診断と位置づけ」自己言及、§5.2 安定性チェックの「0.10の差が内容の変化を反映するとは限らない」句、§6.2 の「不一致を可視化」文、§6.3 の judge 感度説明句、§4.1 の 0.992（4度目の登場）、§6.1 の「別 artist ながら sound が違い」修飾。圧縮: §4.1 層化説明、§4.2 bundle 注意文、§6.1 の 2 つの gap 再要約（表参照へ）、「等価だと主張するのではない」の 1 文化。§5.1 は導出可能な 11,535 件と turn-adjusted CI（表1に保持）も削除し、「共通のターン分布」を pooled に統一。ページ状況: 4/5 ページ境界が §6.3 中盤へ戻り、本文超過は約30行→約12行に縮小。参考文献は5ページ目内で終了（計5ページ）。最終 4P 詰めは著者対応のまま保留。)* *(2026-07-25 追記7: 著者承認のページ詰めパス（Tier 1/2 リスト）。序論の結果先取り段落を削除（abstract・§5 と重複。*pivot* の定義は §4.1 に残存）、序論末尾をコンプライアンス宣言のみに縮約、表1キャプションを圧縮（「丸め前の値から計算」を削除し定義を短縮）、§3.3 の Blind-B スコア列挙を composite 0.4998（nDCG@20 0.3528、judge 4.30）へ短縮、§2.1 の metadata/embedding 列挙を削除（§2 リード文が既に列挙）、§4.2 の bundle 注意文を削除（§6.3 が保持）、§5.2 の安定性段落を 1 文に凝縮。著者指示により、セミコロンで 1 文に繋いでいた 2 箇所（序論コンプライアンス宣言・§6.2 項目1の detector 文）は 2 文に復元。ページ状況: 5 ページ目に残る本文は結論末尾の約 4 行のみで、参考文献は 5 ページ内で終了（計 5 ページ）。)* *(2026-07-25 追記8: 著者依頼により §4.1 の三つの語彙ルール定義を箇条書き（名前: 説明+マッチ例）へ変換し、§3.3 の Blind-B 13番目 source 除外の記述（0.47→0.50）を削除——監査の主張に非依存・他所からの参照なし・Blind-B 構成は提出 code リポジトリに記録。スコア文は「採用した Blind-B 提出は…」で開始。ページ状況は不変: 結論末尾の約 4 行が 5 ページ目（リスト化の増分と削除がほぼ相殺）。)* *(2026-07-25 追記9: 著者依頼により図1キャプションを「提出した3段階 pipeline。」へ短縮——削除した内容は図中の Stage 1/2/3 見出し・着色・「cannot change the ranking」注記が担う——、表2キャプションを「…（bundle の定義は4.2節）」へ圧縮し §4.2 冒頭と重複する数値を削除。表1キャプションは §4.1 本文の列挙削除時に定義を一本化した経緯があるため意図的に維持。**ページ状況: 4P+1P 達成**——結論が4ページ目で完結し、5ページ目は参考文献のみ（計5ページ）。)* *(2026-07-25 追記7: 著者依頼により §5.1 を可読性改稿。第1段落は期待方向を先頭に置き（「正解が artist 変更要求に従うなら pivot ターンで repeat は稀になるはず。実際は逆」）、生値76.3%を表1の `Next same` 列に紐付け、76.0%対63.1%を「+12.9 pp turn-adjusted difference の背後にあるターン再重み付け後の pivot／非 pivot 率」と明示（76.3/76.0 の紛らわしい併存を解消。再重み付け後の率は表に無い）、ターン別レンジとロジスティック OR をそれぞれターン別分析・二次チェックとして位置付けた。第2段落は「ラベルは直前ターンの推薦を評価する」を明示し、多義的だった "does not follow" を「この判定の直後に選ばれる relevant track はそれを無視する」に置換、「負の progress」を実ラベル名 `DOES_NOT_MOVE_TOWARD_GOAL` に統一。第3段落は 194/200 と母集団 coverage を展開し、含意（false positive は repeat 率を説明できないほど稀・低 recall が制限するのは coverage であって precision ではない→§6.3）を追加。ページ状況: §5.1 は3ページ目開始に移動し、5ページ目への本文超過は約12行→約20行（§6.3+結論）、参考文献は5ページ目内で終了（計5ページ）。最終 4P 詰めは著者対応のまま保留。)* *(2026-07-25 追記10: 著者依頼により序論末尾のコンプライアンス段落（提出システムは文脈提示のみ・非公開情報不使用・judge prompt 逆解析なし）を削除——チーム順位は abstract と §3.3 に重複、データ来歴は §2.2 の監査スコープ段落が記録。序論は貢献リストで終わる構成に。浮いた行で、著者指摘の LLM 的ケイデンス対策として句読点の均一リズム解消パスを実施: §2–§6 の独立節をコロン/セミコロンで結合していた 14 箇所（§6.1 の gap 段落の 5 箇所と §5.2 の em-dash 構文を含む）を文分割し、構造用途（リスト導入・キャプション/CI のコロン・§4.2 の列挙セミコロン・意図的な対句 2 箇所）は温存。散文密度はコロン 35→30、セミコロン 24→12。ページ状況: 参考文献が 4 ページ目から始まる余裕付きで 4P+1P 維持。)* *(2026-07-25 追記11: 著者依頼により §4.2 の5応答bundleを、セミコロン連結の1文から表2の各行に対応する箇条書き（Initial / Qwen 4B, revised / Claude, short / Claude, grounded long / Gemini Pro, grounded）へ変換し、間接的だった表への参照を「（表2）」と明示化。ページ状況: 箇条書き化の増分で参考文献が完全に5ページ目へ押し出され、結論はちょうど4ページ目末尾で終了（4P+1P、余裕なし）。)* *(2026-07-25 追記12: 著者判断により §5.2 の judge 安定性チェック段落を全削除。調査の結果、そこで使われていた「4.15と4.25」のペアは exp056/exp057——Table 2 の5 bundleとは無関係などころか ranking 品質も違う（nDCG 0.3030 vs Table 2 の0.4355）、response も exp053 からのコピー流用という、かなり古い post-hoc fusion 実験系列——であることが判明。Table 2 の実際の bundle で再実測するのは新規 Codabench submission のコストに見合わないと判断し、残す/再測定するのではなく削除を選択。ページ状況: 参考文献に再び余裕が戻り、4ページ目途中から開始（4P+1P、余裕あり）。)* *(2026-07-25 追記13: 著者依頼により abstract を可読性改稿。「二つの不整合を見出す」と冒頭で予告し、各発見を First/Second の主張文で開始してから根拠数値を置く構成に変更。precision 0.992 の検出器文は主張の後ろへ移動。「session artist を繰り返す」は「セッション内で既に流れた artist を繰り返す」へ、「On Blind A」は「interim blind leaderboard」へ開いた。RQ2 の冗長句「推薦を1曲も変えずに」は削除。著者の文体ルールに従い em-dash 構文は不使用。LaTeX と md 両版に適用（約165語→約185語）。ページ状況: 結論は4ページ目で完結、参考文献は4ページ目から始まり5ページ目で終了（4P+1P 維持）。)* *(2026-07-25 追記14: 提案した2案から著者が選んだ主張先行型へ abstract を全面差し替え。冒頭を「The official evaluation ... can disagree with itself」の主張文で開始（主催 workshop への tone 配慮で「disagrees」でなく「can disagree」）、§5.2 の解釈文「composite はより良い推薦とより上手い説明を区別できない」を abstract へ昇格、precision 0.992 の検出器詳細は削除（§4.1・§5.1 が保持）。First/Second の発見構造と他の数値は維持（約195語）。LaTeX と md 両版に適用。ページ状況: 結論は4ページ目で完結、参考文献は4ページ目から始まり5ページ目で終了（4P+1P 維持）。)* *(2026-07-25 追記15: 全引用の帰属主張を一次ソースとweb照合する citation audit を実施し3修正を適用。(1) §2.1: TalkPlayData 2 の Recsys エージェントは listener profile と recommendation pool を与えられて初期化され、隠されるのは conversation goal のみ→「それらを見られない」を「profile は見られるが goal は知らされない」へ修正。また progress は binary ラベルなので「goal に照らして評価し」を「goal へ前進しているかをラベル付けして」へ。(2) 序論の先行研究文: RecSys Challenge 2024 の leaderboard は AUC 単独で、heitz2024 の批判は「composite の誤誘導」でなく「diversity が評価に不在で incentive が無い」こと→「composite が遠ざけた」を「accuracy 単独で順位付けし incentive を与えなかった」へ修正。同文で verbosity bias=zheng2023 / 自己生成文の優遇=panickssery2024 と引用を分離（zheng2023 は self-enhancement を「決定できない」と明示的に限定するため）。(3) 一致確認済み: wang2023、panickssery2024、§6.2 の heitz2024、choi2025 の生成順序と 16-32 曲セッション由来 pool、composite 重みと judge 正規化（results.csv から Komekami の composite を 9 桁再現）、volkovs2018、lu2024、ke2017/burges2010。チーム数: results.csv は非 baseline 41 チームだが、公式サイト記載に合わせ著者判断で「40 teams」を維持。ページ状況: 結論は4ページ目で完結、参考文献は5ページ目に全て収まる（4P+1P 維持、余裕は減少）。)* *(2026-07-25 追記16: 著者依頼により abstract の発見1の数字提示を可読性改稿。「76.0% versus 63.1%」の並置を「76.0%、他のターンより12.9ポイント高い」に変え、差分を計算済みで提示（Table 1 の turn-adjusted +12.9 pp と一致）し、期待との逆転を「still」で明示。85.8% の文は曖昧な照応「The repeat rate is 85.8%」をやめ、実際の統計量（直前に不成功とされたまさにその artist の繰り返し=「the next track repeats that artist」）を明示。結論§7の表現とも整合。LaTeX と md 両版に適用。ページ状況: 不変（結論4ページ目完結・参考文献5ページ目、4P+1P 維持）。)* *(2026-07-25 追記17: abstract の85.8%文を、数字より先に方向を述べる形へ改稿。「Even when ... 85.8%」は76.0%に対して高いのか低いのかを読者の推測に委ねていたため、「The pattern sharpens when ...（不成功ラベル時にパターンはいっそう強まる）。The next track then repeats that very artist in 85.8% of cases.」へ変更。LaTeX と md 両版に適用。ページ状況不変（4P+1P 維持）。)*
8. `claude-opus-4-8` の正確な公開 product name と再現可能性を確認する。安定した公開名称がない場合は internal identifier を残し、response regeneration が非決定的であることを開示する。
9. organizers が challenge overview paper を公開した場合、camera-ready では challenge website reference をその論文へ置き換える。
