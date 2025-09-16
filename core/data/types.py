# types.py
from typing import NamedTuple, List
import numpy as np

DURATION_BINS = [1/16, 1/12, 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 4, 8, 16, 32, 64]

def quantize_to_bins(value: float, bins: List[float]) -> int:
    """Quantize a value to the nearest bin and return the bin index."""
    if value <= 0:
        return 0
    if value >= bins[-1]:
        return len(bins) - 1
    
    min_diff = float('inf')
    best_idx = 0
    for i, bin_val in enumerate(bins):
        diff = abs(value - bin_val)
        if diff < min_diff:
            min_diff = diff
            best_idx = i
    return best_idx

class HitObjectVector(NamedTuple):
    """Represents a single hit object's vector data."""
    distance_diff: float
    angle_cos: float
    angle_sin: float
    abs_x: float
    abs_y: float
    object_type: int
    is_new_combo: int
    slider_curve_type: int
    slider_num_anchors: int
    slider_pixel_length: float
    time_diff_bin: int
    duration_bin: int

    @classmethod
    def get_field_names(cls):
        return list(cls._fields)
    
    @classmethod
    def get_vector_dim(cls):
        return len(cls._fields)
    
    @classmethod
    def get_feature_info(cls):
        field_names = cls.get_field_names()
        
        categorical_features = [
            'object_type', 'is_new_combo', 'slider_curve_type',
            'time_diff_bin', 'duration_bin'
        ]
        
        continuous_features = [f for f in field_names if f not in categorical_features]
        
        cat_cardinalities = {
            'object_type': 3,
            'is_new_combo': 2,
            'slider_curve_type': 5,
            'time_diff_bin': len(DURATION_BINS),
            'duration_bin': len(DURATION_BINS)
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
            'names': field_names
        }
        return info

    @classmethod
    def create_with_quantization(cls, distance_diff: float, angle_cos: float, angle_sin: float,
                                time_diff: float, abs_x: float, abs_y: float, object_type: int,
                                is_new_combo: int, slider_curve_type: int,
                                slider_num_anchors: int, slider_pixel_length: float,
                                duration_beats: float):
        time_diff_bin_idx = quantize_to_bins(time_diff, DURATION_BINS)
        duration_bin_idx = quantize_to_bins(duration_beats, DURATION_BINS)
        
        return cls(
            distance_diff=distance_diff,
            angle_cos=angle_cos,
            angle_sin=angle_sin,
            abs_x=abs_x,
            abs_y=abs_y,
            object_type=object_type,
            is_new_combo=is_new_combo,
            slider_curve_type=slider_curve_type,
            slider_num_anchors=slider_num_anchors,
            slider_pixel_length=slider_pixel_length,
            time_diff_bin=time_diff_bin_idx,
            duration_bin=duration_bin_idx
        )
    
    def to_array(self):
        return np.array(self, dtype=np.float32)

class BeatmapMetadata(NamedTuple):
    """Represents beatmap metadata."""
    ar: float
    od: float
    cs: float
    difficulty_rating: float
    bpm: float

    @classmethod
    def get_field_names(cls):
        return list(cls._fields)
    
    @classmethod
    def get_metadata_dim(cls):
        return len(cls._fields)
    
    def to_array(self):
        return np.array(self, dtype=np.float32)

VECTOR_DIM = HitObjectVector.get_vector_dim()
METADATA_DIM = BeatmapMetadata.get_metadata_dim()