"""track metadata に対する BM25 retrieval utility。

Hugging Face dataset から読み込んだ track metadata の指定フィールドを
BM25 index 化し、同じ設定の index は cache から再利用する。
"""
import os
import json
import bm25s
from datasets import load_dataset, concatenate_datasets


class BM25_MODEL:
    """track metadata を対象にした BM25 retriever。"""

    def __init__(self,
        dataset_name: str,
        split_types: list[str],
        corpus_types: list[str],
        cache_dir: str = "./cache",
        n_threads: int = 0,
        chunksize: int = 50,
    ) -> None:
        """BM25 retriever を初期化し、必要なら index を構築する。

        Args:
            dataset_name: track metadata を含む Hugging Face dataset 名。
            split_types: 読み込んで連結する split 名。
            corpus_types: index の文書文字列へ含める metadata フィールド。
            cache_dir: BM25 index と付随 artifact の cache ディレクトリ。
            n_threads: bm25s retrieval の thread 数。0 は bm25s の逐次既定値。
            chunksize: thread 使用時に bm25s retrieval へ渡す query chunksize。
        """
        self.dataset_name = dataset_name
        self.split_types = split_types
        self.corpus_types = corpus_types
        self.corpus_name = "_".join(corpus_types)
        self.cache_dir = cache_dir
        self.n_threads = int(n_threads)
        self.chunksize = int(chunksize)
        self.metadata_dict = self._load_corpus()
        if os.path.exists(f"{self.cache_dir}/bm25/{self.corpus_name}"):
            self.bm25_model, self.track_ids = self._load_bm25(self.corpus_name)
        else:
            self.build_index()
            self.bm25_model, self.track_ids = self._load_bm25(self.corpus_name)

    def _load_bm25(self, corpus_name: str) -> tuple[bm25s.BM25, list[str]]:
        """cache 済み BM25 index と track ID リストを読み込む。

        Args:
            corpus_name: cache 配下の corpus サブディレクトリ名。

        Returns:
            BM25 model と track ID リストの tuple。
        """
        bm25 = bm25s.BM25.load(f"{self.cache_dir}/bm25/{corpus_name}", load_corpus=True)
        track_ids = json.load(open(f"{self.cache_dir}/bm25/{corpus_name}/track_ids.json", "r"))
        return bm25, track_ids

    def _load_corpus(self) -> dict[str, dict]:
        """設定された split の metadata を読み込んで track_id keyed dict にする。

        Returns:
            track_id から metadata 辞書への mapping。
        """
        metadata_dataset = load_dataset(self.dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in self.split_types])
        metadata_dict = {item["track_id"]: item for item in metadata_concat_dataset}
        return metadata_dict

    def _stringify_metadata(self, metadata: dict[str, object]) -> str:
        """metadata 辞書を BM25 index 用の複数行文字列へ変換する。

        Args:
            metadata: ``self.corpus_types`` のフィールドを持つ track metadata。

        Returns:
            選択フィールドごとの ``field: value`` を改行連結した文字列。
        """
        metadata_str = ""
        for corpus_type in self.corpus_types:
            entity = metadata[corpus_type]
            if isinstance(entity, list):
                entity = ", ".join(entity)
            metadata_str += f"{corpus_type}: {entity}\n"
        return metadata_str

    def build_index(self) -> None:
        """読み込み済み corpus から BM25 index を構築して cache に保存する。"""
        track_ids = list(self.metadata_dict.keys())
        corpus = []
        for track_id in track_ids:
            metadata = self.metadata_dict[track_id]
            metadata_str = self._stringify_metadata(metadata)
            corpus.append(metadata_str)
        corpus_tokens = bm25s.tokenize(corpus)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        os.makedirs(os.path.join(self.cache_dir, "bm25", self.corpus_name), exist_ok=True)
        retriever.save(f"{self.cache_dir}/bm25/{self.corpus_name}", corpus=corpus)
        with open(os.path.join(self.cache_dir, "bm25", self.corpus_name, "track_ids.json"), "w") as f:
            json.dump(track_ids, f, indent=2)

    def text_to_item_retrieval(self, query: str, topk: int) -> list[str]:
        """自然言語 query に対する上位 track ID を返す。

        Args:
            query: metadata corpus と照合するユーザ query。
            topk: 返す track 数。

        Returns:
            BM25 score 降順の track ID リスト。
        """
        query_tokens = bm25s.tokenize([query.lower()])
        doc_scores = self.bm25_model.retrieve(
            query_tokens,
            k=topk,
            return_as="tuple",
            n_threads=self.n_threads,
            chunksize=self.chunksize,
        )
        bm25_results = doc_scores.documents[0]
        return [self.track_ids[item['id']] for item in bm25_results]

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int) -> list[list[str]]:
        """複数 query に対する上位 track ID を batch 取得する。

        Args:
            queries: metadata corpus と照合する query リスト。
            topk: query ごとに返す track 数。

        Returns:
            query ごとの BM25 score 降順 track ID リスト。
        """
        query_tokens = bm25s.tokenize([q.lower() for q in queries])
        doc_scores = self.bm25_model.retrieve(
            query_tokens,
            k=topk,
            return_as="tuple",
            n_threads=self.n_threads,
            chunksize=self.chunksize,
        )
        results = []
        for i in range(len(queries)):
            bm25_results = doc_scores.documents[i]
            results.append([self.track_ids[item['id']] for item in bm25_results])
        return results

    def batch_text_to_item_retrieval_with_scores(
        self, queries: list[str], topk: int
    ) -> list[list[tuple[str, float]]]:
        """複数 query に対する上位 track ID と BM25 score を batch 取得する。

        ``batch_text_to_item_retrieval`` と同じ retrieval を行うが、通常は破棄される
        ``doc_scores.scores`` を track ID と組にして返す。candidate source の confidence
        feature（source 内の相対スコア等）を後段の reranker へ渡すために使う。

        Args:
            queries: metadata corpus と照合する query リスト。
            topk: query ごとに返す track 数。

        Returns:
            query ごとの ``(track_id, bm25_score)`` リスト。順序は score 降順
            （``batch_text_to_item_retrieval`` と同一の並び）。
        """
        query_tokens = bm25s.tokenize([q.lower() for q in queries])
        doc_scores = self.bm25_model.retrieve(
            query_tokens,
            k=topk,
            return_as="tuple",
            n_threads=self.n_threads,
            chunksize=self.chunksize,
        )
        results: list[list[tuple[str, float]]] = []
        for i in range(len(queries)):
            # documents[i] と scores[i] は同じ並びで対応するため zip で組にする
            documents = doc_scores.documents[i]
            scores = doc_scores.scores[i]
            results.append(
                [
                    (self.track_ids[item["id"]], float(score))
                    for item, score in zip(documents, scores)
                ]
            )
        return results
