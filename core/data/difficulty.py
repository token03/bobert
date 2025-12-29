import os
import json
from typing import Tuple, List, Optional, Dict
import concurrent.futures
import itertools
import rosu_pp_py as rosu
from tqdm import tqdm

from .types import DIFFICULTY_ATTRIBUTES


class DifficultyManager:
    def __init__(self, cache_path: str, raw_path: str):
        self.cache_path = cache_path
        self.raw_path = raw_path
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict:
        try:
            with open(self.cache_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_cache(self):
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f)

    def get_attributes(self, beatmap_id: int, seq_len: int) -> Optional[Dict]:
        str_bid, str_seq_len = str(beatmap_id), str(seq_len)
        attrs = self.cache.get(str_bid, {}).get(str_seq_len)
        if attrs and all(k in attrs for k in DIFFICULTY_ATTRIBUTES):
            return attrs
        return None

    def update_missing(self, tasks: List[Tuple[int, int]]):
        if not tasks:
            return

        if not os.path.isdir(self.raw_path):
            raise FileNotFoundError(f"Raw beatmap path '{self.raw_path}' not found.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_task = {
                executor.submit(
                    _calculate_difficulty_attributes_worker, bid, seq_len, self.raw_path
                ): (bid, seq_len)
                for bid, seq_len in tasks
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_task),
                total=len(future_to_task),
                desc="Calculating Attributes",
            ):
                bid, seq_len = future_to_task[future]
                new_attrs = future.result()
                if new_attrs is not None:
                    str_bid, str_seq_len = str(bid), str(seq_len)
                    if str_bid not in self.cache:
                        self.cache[str_bid] = {}
                    self.cache[str_bid][str_seq_len] = new_attrs
        self.save_cache()


def _calculate_difficulty_attributes_worker(
    beatmap_id: int, seq_len: int, raw_beatmap_path: str
) -> Optional[Dict[str, float]]:
    osu_file_path = os.path.join(raw_beatmap_path, f"{beatmap_id}.osu")
    if not os.path.exists(osu_file_path):
        return None
    try:
        with open(osu_file_path, "r", encoding="utf-8") as f:
            beatmap_content = f.read()

        beatmap = rosu.Beatmap(content=beatmap_content)

        if beatmap.mode != 0 or beatmap.n_objects < 2:
            return None

        objects_to_process = (
            min(seq_len, beatmap.n_objects) if seq_len else beatmap.n_objects
        )

        diff_attrs_calculator = rosu.Difficulty(
            ar=10.0,
            cs=4.0,
        )

        gradual_result_iterator = diff_attrs_calculator.gradual_difficulty(beatmap)

        target_index = objects_to_process - 2
        target_attrs = next(
            itertools.islice(gradual_result_iterator, target_index, None), None
        )

        if target_attrs:
            return {
                k: getattr(target_attrs, k)
                if k in ["stars", "aim", "speed", "slider_factor"]
                else getattr(beatmap, k)
                for k in DIFFICULTY_ATTRIBUTES
            }

        return None
    except Exception as e:
        return None
