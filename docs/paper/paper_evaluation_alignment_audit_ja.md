# Music-CRS におけるアイテム関連性・目標進捗・LLM評価応答の整合性監査

**チーム:** Komekami

**コード:** <https://github.com/tmp-friends/music-crs-challenge-2026>

> **投稿メモ（論文本文には含めない）:** 評価整合性分析案、ドラフト基準日: 2026-07-24（ページ制限に収めるため表現を平易化・圧縮）。RecSys Challenge 2026 公式サイトの Timeline では、論文投稿締切が **2026年7月20日から7月24日へ延長**され、採否通知も8月3日から8月5日へ変更されている。一方、同ページの Paper Submission Guidelines 欄には旧日付が残っているため、投稿前に EasyChair 上の有効な締切を必ず確認すること。出典: <https://www.recsyschallenge.com/2026/>。
>
> 本稿は ACM RecSys Challenge 2026 Workshop の書式、すなわち本文4ページ＋参考文献1ページ、`acmart` の `sigconf` による二段組を想定した日本語確認用ドラフトである。正式投稿には英語版を用いる。

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
