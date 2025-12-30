# hitobject.py
from typing import NamedTuple, List, Dict, Any, Optional, Tuple
import numpy as np
from enum import Enum

OSU_STAGE_WIDTH = 512
OSU_STAGE_HEIGHT = 384
CENTER_X = OSU_STAGE_WIDTH / 2.0
CENTER_Y = OSU_STAGE_HEIGHT / 2.0
DEFAULT_PRE_START_MS = 200.0

MAX_METER_CARDINALITY = 8

DURATION_BINS = [1/16, 1/12, 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 4, 8, 16, 32, 64]

FEATURE_GROUPS = {
    'spatial': {
        'features': ['norm_x', 'norm_y', 'delta_x', 'delta_y', 'relative_angle'],
        'output_dim': 192
    },
    'rhythm': {
        'features': ['log_time_diff_ms', 'bpm', 'notes_per_second', 'velocity', 'rhythm_change'],
        'output_dim': 160
    },
    'slider': {
        'features': ['log_slider_pixel_length', 'slider_repeats', 'slider_tortuosity'],
        'output_dim': 64
    },
    'categorical': {
        'features': ['object_type', 'is_new_combo', 'beat_in_measure', 'time_diff_bin', 'rhythmic_snap'],
        'output_dim': 96
    }
}

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

OBJECT_TYPE_CIRCLE = 0
OBJECT_TYPE_SLIDER_HEAD = 1
OBJECT_TYPE_SLIDER_END = 2
OBJECT_TYPE_SPINNER_START = 3
OBJECT_TYPE_SPINNER_END = 4

SLIDER_TYPE_INDEX = OBJECT_TYPE_SLIDER_HEAD

class HitObject(NamedTuple):
    norm_x: float
    norm_y: float
    delta_x: float
    delta_y: float
    log_time_diff_ms: float
    bpm: float
    notes_per_second: float
    velocity: float
    relative_angle: float
    rhythm_change: float
    
    log_slider_pixel_length: float
    slider_repeats: float
    slider_tortuosity: float
    beat_in_measure: float
    
    object_type: int
    is_new_combo: int
    time_diff_bin: int
    rhythmic_snap: int
    
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
            'log_slider_pixel_length', 'slider_repeats', 'slider_tortuosity'
        ]

        categorical_features = [
            'object_type', 'is_new_combo', 'beat_in_measure',
            'time_diff_bin', 'rhythmic_snap', 
        ]
        
        continuous_features = [f for f in field_names if f not in categorical_features]
        
        cat_cardinalities = {
            'object_type': 5, 
            'is_new_combo': 2,
            'beat_in_measure': MAX_METER_CARDINALITY,
            'time_diff_bin': len(DURATION_BINS),
            'rhythmic_snap': 6,
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
            'notes_per_second': NormalizationType.STANDARD,
            'velocity': NormalizationType.STANDARD,
            'relative_angle': NormalizationType.NONE,
            'rhythm_change': NormalizationType.STANDARD,
            'log_slider_pixel_length': NormalizationType.STANDARD,
            'slider_repeats': NormalizationType.STANDARD,
            'slider_tortuosity': NormalizationType.STANDARD,
            
            'object_type': NormalizationType.CATEGORICAL,
            'is_new_combo': NormalizationType.CATEGORICAL,
            'beat_in_measure': NormalizationType.CATEGORICAL,
            'time_diff_bin': NormalizationType.CATEGORICAL,
            'rhythmic_snap': NormalizationType.CATEGORICAL,
        }
    
VECTOR_DIM = HitObject.get_vector_dim()