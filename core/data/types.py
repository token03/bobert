# types.py
from typing import NamedTuple, List, Dict, Any, Optional, Tuple
import numpy as np
from enum import Enum

DURATION_BINS = [1/16, 1/12, 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 4, 8, 16, 32, 64]

class NormalizationType(Enum):
    CATEGORICAL = "categorical"
    STANDARD = "standard"
    LOG = "log"
    MINMAX = "minmax"
    NONE = "none"

def quantize_to_bins(values: np.ndarray, bins: List[float]) -> np.ndarray:
    bins_arr = np.array(bins)
    diffs = np.abs(values[:, np.newaxis] - bins_arr)
    result = np.argmin(diffs, axis=1)
    result[values <= 0] = 0
    result[values >= bins_arr[-1]] = len(bins) - 1
    return result

SLIDER_TYPE_INDEX = 1

class HitObjectVector(NamedTuple):
    # Continuous Features
    norm_x: float
    norm_y: float
    delta_x: float
    delta_y: float
    log_time_diff_ms: float
    bpm: float
    log_slider_pixel_length: float
    slider_repeats: float
    slider_end_x: float
    slider_end_y: float
    
    # Categorical Features
    object_type: int
    is_new_combo: int
    kiai_time: int
    time_diff_bin: int
    duration_bin: int
    
    @classmethod
    def get_field_names(cls):
        return list(cls._fields)

    @staticmethod
    def get_raw_field_names() -> List[str]:
        return [
            'x', 'y', 'slider_end_x', 'slider_end_y', 'slider_repeats', 
            'pixel_length'
        ]
    
    @classmethod
    def get_vector_dim(cls):
        return len(cls._fields)

    @classmethod
    def get_feature_info(cls):
        field_names = cls.get_field_names()

        slider_feature_names = [
            'log_slider_pixel_length', 'slider_repeats', 
            'slider_end_x', 'slider_end_y', 'duration_bin'
        ]

        categorical_features = [
            'object_type', 'is_new_combo', 'kiai_time',
            'time_diff_bin', 'duration_bin'
        ]
        
        continuous_features = [f for f in field_names if f not in categorical_features]
        
        cat_cardinalities = {
            'object_type': 3,
            'is_new_combo': 2,
            'kiai_time': 2,
            'time_diff_bin': len(DURATION_BINS),
            'duration_bin': len(DURATION_BINS),
        }

        info = {
            'categorical': {
                name: {
                    'index': field_names.index(name),
                    'cardinality': cat_cardinalities[name]
                } for name in categorical_features
            },
            'continuous': {
                name: field_names.index(name) for name in continuous_features
            },
            'slider': {
                name: field_names.index(name) for name in slider_feature_names
            },
            'names': field_names
        }
        return info

    @classmethod
    def get_normalization_specs(cls) -> Dict[str, NormalizationType]:
        return {
            'norm_x': NormalizationType.NONE,
            'norm_y': NormalizationType.NONE,
            'delta_x': NormalizationType.STANDARD,
            'delta_y': NormalizationType.STANDARD,
            'log_time_diff_ms': NormalizationType.STANDARD,
            'bpm': NormalizationType.STANDARD,
            'log_slider_pixel_length': NormalizationType.STANDARD,
            'slider_repeats': NormalizationType.STANDARD,
            'slider_end_x': NormalizationType.STANDARD,
            'slider_end_y': NormalizationType.STANDARD,
            'object_type': NormalizationType.CATEGORICAL,
            'is_new_combo': NormalizationType.CATEGORICAL,
            'kiai_time': NormalizationType.CATEGORICAL,
            'time_diff_bin': NormalizationType.CATEGORICAL,
            'duration_bin': NormalizationType.CATEGORICAL,
        }
    
    @classmethod
    def get_field_descriptions(cls) -> Dict[str, str]:
        return {
            'norm_x': "Absolute x-coordinate, normalized to [-1, 1]",
            'norm_y': "Absolute y-coordinate, normalized to [-1, 1]",
            'delta_x': "Change in x from the previous object",
            'delta_y': "Change in y from the previous object",
            'log_time_diff_ms': "Log-transformed time difference in milliseconds from previous object",
            'bpm': "Beats Per Minute at the time of the hit object",
            'log_slider_pixel_length': "Log-transformed pixel length of slider's curve path",
            'slider_repeats': "Number of slider repeats (slides - 1)",
            'slider_end_x': "Slider's end x-coordinate, normalized to [-1, 1]",
            'slider_end_y': "Slider's end y-coordinate, normalized to [-1, 1]",
            'object_type': "Type of hit object (circle, slider, spinner)",
            'is_new_combo': "Whether this hit object starts a new combo",
            'kiai_time': "Whether the hit object is in kiai time",
            'time_diff_bin': "Quantized time difference to previous hit object in beats",
            'duration_bin': "Quantized duration of the hit object in beats",
        }

VECTOR_DIM = HitObjectVector.get_vector_dim()

class RawTimingPoint(NamedTuple):
    time: int
    beat_length: float
    uninherited: bool
    effects: int

class RawHitObject(NamedTuple):
    x: int
    y: int
    time: int
    object_type: int 
    is_new_combo: int
    curve_type: Optional[str]       
    curve_points: Optional[List[Tuple[int, int, int]]]
    slides: Optional[int]
    pixel_length: Optional[float]
    end_time: int
    hit_sound: int

class RawBeatmap(NamedTuple):
    beatmap_id: int
    category: str
    hp_drain: float
    cs: float
    od: float
    ar: float
    slider_multiplier: float
    slider_tick: float
    difficulty_rating: float
    timing_points: List[RawTimingPoint]
    hit_objects: List[RawHitObject]