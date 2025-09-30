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
    distance_diff_end: float
    velocity: float
    cos_relative_angle: float
    sin_relative_angle: float
    object_type: int
    is_new_combo: int
    slider_absolute_length: float
    cos_slider_absolute_angle: float
    sin_slider_absolute_angle: float
    slider_curve_type: int
    slider_num_anchors: int
    slider_pixel_length: float
    slider_repeats: int
    time_diff_bin: int
    duration_bin: int
    bpm: float
    kiai_time: int
    
    @classmethod
    def get_field_names(cls):
        return list(cls._fields)

    @staticmethod
    def get_raw_field_names() -> List[str]:
        return [
            'x', 'y', 'slider_end_x', 'slider_end_y', 'slider_repeats', 
            'num_anchors', 'pixel_length', 'curve_type_char', 
        ]
    
    @classmethod
    def get_vector_dim(cls):
        return len(cls._fields)

    @classmethod
    def get_feature_info(cls):
        field_names = cls.get_field_names()

        slider_feature_names = [
            'slider_absolute_length', 'cos_slider_absolute_angle',
            'sin_slider_absolute_angle', 'slider_num_anchors',
            'slider_pixel_length', 'slider_repeats'
        ]

        categorical_features = [
            'object_type', 'is_new_combo', 'slider_curve_type',
            'time_diff_bin', 'duration_bin', 'kiai_time'
        ]
        
        continuous_features = [f for f in field_names if f not in categorical_features]
        
        cat_cardinalities = {
            'object_type': 3,
            'is_new_combo': 2,
            'slider_curve_type': 5,
            'time_diff_bin': len(DURATION_BINS),
            'duration_bin': len(DURATION_BINS),
            'kiai_time': 2
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
            'distance_diff_end': NormalizationType.LOG,
            'velocity': NormalizationType.LOG,
            'cos_relative_angle': NormalizationType.STANDARD,
            'sin_relative_angle': NormalizationType.STANDARD,
            'object_type': NormalizationType.CATEGORICAL,
            'is_new_combo': NormalizationType.CATEGORICAL,
            'slider_absolute_length': NormalizationType.LOG,
            'cos_slider_absolute_angle': NormalizationType.STANDARD,
            'sin_slider_absolute_angle': NormalizationType.STANDARD,
            'slider_curve_type': NormalizationType.CATEGORICAL,
            'slider_num_anchors': NormalizationType.LOG,
            'slider_pixel_length': NormalizationType.LOG,
            'slider_repeats': NormalizationType.LOG,
            'time_diff_bin': NormalizationType.CATEGORICAL,
            'duration_bin': NormalizationType.CATEGORICAL,
            'bpm': NormalizationType.LOG,
            'kiai_time': NormalizationType.CATEGORICAL
        }
    
    @classmethod
    def get_field_descriptions(cls) -> Dict[str, str]:
        return {
            'distance_diff_end': "Distance from previous hit object's end point (jump distance)",
            'velocity': "Velocity to previous hit object (pixels/ms)",
            'cos_relative_angle': "Cosine of angle between current and previous jump vectors (flow aim)",
            'sin_relative_angle': "Sine of angle between current and previous jump vectors (flow aim)",
            'object_type': "Type of hit object (circle, slider, spinner)",
            'is_new_combo': "Whether this hit object starts a new combo",
            'slider_absolute_length': "Straight-line distance from slider head to slider tail",
            'cos_slider_absolute_angle': "Cosine of angle between jump-in vector and slider's head-to-tail vector",
            'sin_slider_absolute_angle': "Sine of angle between jump-in vector and slider's head-to-tail vector",
            'slider_curve_type': "Curve type of slider (0 if not a slider)",
            'slider_num_anchors': "Number of anchor points in slider (0 if not a slider)",
            'slider_pixel_length': "Pixel length of slider's curve path (0 if not a slider)",
            'slider_repeats': "Number of slider repeats (slides - 1)",
            'time_diff_bin': "Quantized time difference to previous hit object",
            'duration_bin': "Quantized duration of the hit object",
            'bpm': "Beats Per Minute at the time of the hit object",
            'kiai_time': "Whether the hit object is in kiai time"
        }

class BeatmapMetadata(NamedTuple):
    ar: float
    od: float
    cs: float
    hp_drain: float
    slider_multiplier: float
    slider_tick: float

    @classmethod
    def get_field_names(cls):
        return list(cls._fields)
    
    @classmethod
    def get_metadata_dim(cls):
        return len(cls._fields)
    
    @classmethod
    def get_normalization_specs(cls) -> Dict[str, NormalizationType]:
        return {
            'ar': NormalizationType.STANDARD,
            'od': NormalizationType.STANDARD,
            'cs': NormalizationType.STANDARD,
            'hp_drain': NormalizationType.STANDARD,
            'slider_multiplier': NormalizationType.STANDARD,
            'slider_tick': NormalizationType.STANDARD,
        }

    @classmethod
    def get_field_descriptions(cls) -> Dict[str, str]:
        return {
            'ar': "Approach Rate",
            'od': "Overall Difficulty",
            'cs': "Circle Size",
            'hp_drain': "HP Drain Rate",
            'slider_multiplier': "Slider Velocity Multiplier",
            'slider_tick': "Slider Tick Rate",
        }

VECTOR_DIM = HitObjectVector.get_vector_dim()
METADATA_DIM = BeatmapMetadata.get_metadata_dim()

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