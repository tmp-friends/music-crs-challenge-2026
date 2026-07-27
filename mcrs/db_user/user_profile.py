"""ユーザープロファイル dataset を会話生成向けに参照する。"""

import os
import json
import random
from datasets import load_dataset, concatenate_datasets

class UserProfileDB:
    """user_id から年齢層・性別・国などの profile を取得する DB。"""

    def __init__(self,
            dataset_name: str,
            split_types: list[str],
        ):
        """指定 split の user metadata を読み込み、user_id で引ける形にする。

        Args:
            dataset_name: Hugging Face 上の user metadata dataset 名。
            split_types: 読み込む split 名のリスト。
        """
        metadata_dataset = load_dataset(dataset_name)
        metadata_concat_dataset = concatenate_datasets([metadata_dataset[split_type] for split_type in split_types])
        self.default_columns = ['user_id', 'age_group', 'gender', 'country_name']
        self.user_profiles = {item["user_id"]: item for item in metadata_concat_dataset}

    def id_to_profile(self, user_id: str):
        """user_id に対応するプロファイル辞書を返す。

        Args:
            user_id: 参照するユーザ ID。

        Returns:
            dataset の 1 行に相当するユーザープロファイル。
        """
        user_profile = self.user_profiles[user_id]
        return user_profile

    def id_to_profile_str(self, user_id: str):
        """LLM prompt に埋め込みやすい複数行のプロフィール文字列を返す。

        Args:
            user_id: 参照するユーザ ID。

        Returns:
            ``key: value`` 形式を改行で連結した文字列。
        """
        user_profile = self.user_profiles[user_id]
        profile_str = [f"{key}: {user_profile[key]}" for key in self.default_columns]
        return "\n".join(profile_str)
