"""Claude Agent SDK（``claude-agent-sdk``）経由で response を生成する backend。

exp063 の手動フロー（``build_gen_context.py`` → Agent tool で subagent を手動 spawn → part JSON →
``assemble_claude_responses.py``）を、プログラムから再現可能な 1 経路に置き換えるための backend。
Claude Code の CLI ログイン（サブスク認証）をそのまま再利用するため API キーは不要で、GPU/disk も使わない
（手動 Claude subagent 経路が持っていた利点をそのまま保つ）。

``QwenBackend`` / ``GeminiBackend`` と同じ ``generate(sys_prompts, chat_histories, recommend_items,
max_new_tokens, batch_size)`` 署名を持つ drop-in backend として実装し、exp063 の
``generate_responses_v2.py`` から ``--backend claude`` で差し替えられるようにする。leak-safe context・
postclean・空 fallback・80 件保証・ranking 固定はすべて呼び出し側（generate_responses_v2）の既存ロジックを
そのまま使い、本 backend は「1 task → 1 plain-text 応答」の生成だけを担う。

リーク禁止: 本 backend は呼び出し側が組んだ leak-safe な system_prompt / chat_history / recommend_item
だけを受け取り、それ以外（future turn / ground truth / 評価アノテーション）には一切触れない。さらに
``setting_sources=[]`` で repo の CLAUDE.md / AGENTS.md / skill を子 ``claude`` プロセスへ読み込ませず、
生成器がコンペ作業文脈に汚染されないようにする。
"""

from __future__ import annotations

import asyncio

# Claude Agent SDK の許容応答モデル既定。V-Claude-long の品質は「現行最上位の Claude」由来なので、
# CLI の ambient default（環境次第で Sonnet 等になり得る）に委ねず明示的に Opus を既定にする。
DEFAULT_MODEL = "claude-opus-4-8"


def _render_user_prompt(chat_history, recommend_item):
    """正規化済み履歴 + 推薦 track から、query() に渡す 1 本の user prompt 文字列を組む。

    ``query(prompt=...)`` は文字列 1 本を取るため、Gemini backend が contents へ展開していた履歴を
    ここでは読みやすい transcript に平坦化する。chat_history は呼び出し側で current turn までに限定された
    leak-safe な履歴（末尾が現在の user 発話）なので、そのまま転記してよい。

    Args:
        chat_history: ``[{"role": "user"|"assistant"|"system", "content": str}, ...]``。末尾は現在の user 発話。
        recommend_item: 推薦 top1 track の grounding 文字列（title/artist/year/tags 連結）。

    Returns:
        system_prompt とは別に渡す user 側プロンプト文字列。
    """
    lines = ["Here is the conversation so far:\n"]
    role_label = {"user": "User", "assistant": "Assistant", "system": "System"}
    for msg in chat_history:
        label = role_label.get(msg.get("role"), "Assistant")
        lines.append(f"{label}: {msg.get('content', '')}")
    lines.append("\nThe recommendation system selected this track to recommend:")
    lines.append(recommend_item)
    # 出力を本文だけに固定する（先頭ラベル / markdown / 全体引用符は postclean でも落とすが、
    # そもそも出させない方が judge へ scaffold が漏れにくい）。
    lines.append(
        "\nWrite the user-facing reply now. Output ONLY the reply text itself — "
        "no preamble, no role label, no markdown, and do not wrap the whole message in quotes."
    )
    return "\n".join(lines)


class ClaudeAgentBackend:
    """``claude-agent-sdk`` を使い 1 task ずつ Claude へ単発生成させる backend。

    各 task は独立した単ターン生成（``max_turns=1``、tool 無効）。``batch_size`` を最大同時実行数として
    asyncio で並列化する（各 query は内部で ``claude`` CLI の子プロセスを起動するため、並列度は控えめに）。
    応答は ``AssistantMessage`` 内の ``TextBlock`` を全て連結し、``ResultMessage`` で完了を確認する。

    生成は非決定的（Claude の単発生成）であり、再現は seed では保証されない。これは Gemini backend
    （temperature=0.7）と同じ扱いで、ranking 固定下の response 軸では許容する。
    """

    def __init__(self, model=None, max_concurrency=4, max_retries=2, verbose=True):
        """SDK の遅延 import と生成設定の保持を行う。

        Args:
            model: 応答生成モデル（None なら DEFAULT_MODEL=Opus を使う）。
            max_concurrency: 同時に走らせる query() 数（子 ``claude`` プロセス数）。
            max_retries: 1 task あたりの再試行回数（空応答 / 一過性失敗時）。
            verbose: 進捗と実モデルをログ出力するか。
        """
        # 遅延 import: claude-agent-sdk 未導入環境で Qwen/Gemini 経路を壊さないため。
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        self._query = query
        self._Options = ClaudeAgentOptions
        self._AssistantMessage = AssistantMessage
        self._TextBlock = TextBlock
        self._ResultMessage = ResultMessage

        self.model = model or DEFAULT_MODEL
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.verbose = verbose
        self._reported_model = None  # 実際に応答したモデル（ResultMessage 由来）を 1 度だけ記録

    def _build_options(self, system_prompt):
        """単発・純テキスト・leak-safe な生成のための ClaudeAgentOptions を作る。

        - ``system_prompt``(plain str) は子 ``claude`` の system prompt を **置換**する（コーディング
          エージェント prompt ではなく音楽コンシェルジュとして応答させる）。
        - tool 制約の実体は ``disallowed_tools``（主要 tool を明示拒否）+ ``permission_mode="bypassPermissions"``
          （パーミッション待ちハングを防止）+ ``max_turns=1``（1 ターンで tool 連鎖不可）。``allowed_tools=[]``
          は allowlist として効くわけではない（空配列は CLI へ ``--allowedTools`` を出さない）が、意図を
          明示するため添える。これらで純テキスト生成に固定する。
        - ``setting_sources=[]`` で repo の CLAUDE.md / AGENTS.md / 設定 / skill を読み込ませない
          （生成器をコンペ作業文脈で汚染しない・余計な latency を避ける）。
        - ``max_turns=1`` で 1 応答に限定する。
        """
        return self._Options(
            system_prompt=system_prompt,
            model=self.model,
            max_turns=1,
            allowed_tools=[],
            disallowed_tools=[
                "Bash", "Read", "Write", "Edit", "Glob", "Grep",
                "WebFetch", "WebSearch", "Task", "NotebookEdit", "TodoWrite",
            ],
            permission_mode="bypassPermissions",
            setting_sources=[],
        )

    async def _generate_one(self, sem, idx, n, system_prompt, chat_history, recommend_item):
        """1 task を生成する（semaphore で同時実行数を制限）。

        Returns:
            生成テキスト（全 TextBlock 連結）。全 retry 失敗・空なら ""（呼び出し側が grounded fallback で埋める）。
        """
        user_prompt = _render_user_prompt(chat_history, recommend_item)
        options = self._build_options(system_prompt)
        async with sem:
            for attempt in range(self.max_retries + 1):
                parts = []
                subtype = None
                try:
                    async for message in self._query(prompt=user_prompt, options=options):
                        if isinstance(message, self._AssistantMessage):
                            # 1 応答に複数 TextBlock が来ることがあるので全て連結する
                            # （末尾だけ拾うと本文を取りこぼす）。thinking / tool ブロックは無視。
                            for block in message.content:
                                if isinstance(block, self._TextBlock):
                                    parts.append(block.text)
                        elif isinstance(message, self._ResultMessage):
                            subtype = getattr(message, "subtype", None)
                            # 実際に応答したモデルを 1 度だけ記録（pin 検証用）。
                            if self._reported_model is None:
                                self._reported_model = getattr(message, "model", None) or self.model
                except Exception as exc:  # noqa: BLE001 — SDK / 子プロセスの一過性失敗を握って retry
                    if self.verbose:
                        print(f"[claude] task {idx} attempt {attempt} failed: {exc}", flush=True)
                    continue
                text = "".join(parts).strip()
                if text and subtype in (None, "success"):
                    return text
                if self.verbose:
                    print(
                        f"[claude] task {idx} attempt {attempt} empty/non-success "
                        f"(subtype={subtype}); retrying",
                        flush=True,
                    )
            return ""  # 全 retry 失敗。呼び出し側の fallback_response が埋める。

    async def _run(self, sys_prompts, chat_histories, recommend_items):
        """全 task を semaphore 付きで並列生成し、入力順のリストで返す。"""
        sem = asyncio.Semaphore(self.max_concurrency)
        n = len(sys_prompts)
        tasks = [
            self._generate_one(sem, i, n, sys_prompts[i], chat_histories[i], recommend_items[i])
            for i in range(n)
        ]
        out = await asyncio.gather(*tasks)  # gather は入力順を保つ（順序保証）
        if self.verbose and self._reported_model:
            print(f"[claude] responded model = {self._reported_model}", flush=True)
        return list(out)

    def generate(self, sys_prompts, chat_histories, recommend_items, max_new_tokens, batch_size):
        """Qwen/Gemini backend と同じ署名の同期 API（内部で asyncio を回す）。

        ``max_new_tokens`` は Claude では prompt 側の語数指示で長さを制御するため未使用（署名互換のため受ける）。
        ``batch_size`` は同時実行数の上書きに使う（>0 のとき max_concurrency を置き換える）。
        """
        if batch_size and batch_size > 0:
            self.max_concurrency = batch_size
        n = len(sys_prompts)
        assert len(chat_histories) == n and len(recommend_items) == n, "backend 入力の件数不一致"
        if self.verbose:
            print(
                f"[claude] generating {n} responses (model={self.model}, "
                f"concurrency={self.max_concurrency})",
                flush=True,
            )
        return asyncio.run(self._run(sys_prompts, chat_histories, recommend_items))
