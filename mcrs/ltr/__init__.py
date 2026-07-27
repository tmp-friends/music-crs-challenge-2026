"""LightGBM LTR の dataset / feature / model 共有パッケージ。

exp014 由来の ``data_utils`` / ``features`` / ``build_dataset`` / ``apply_lgbm`` を experiment
フォルダから切り出した共有ライブラリ。旧 exp014 パスは re-export shim 経由でここに解決される。
``train_lgbm`` は train_lowmem の monkeypatch 互換のため exp014 に凍結したまま残す。
"""
