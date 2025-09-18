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
    x_diff: float
    y_diff: float
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
    def create_with_quantization(cls, x_diff: float, y_diff: float,
                                time_diff: float, object_type: int,
                                is_new_combo: int, slider_curve_type: int,
                                slider_num_anchors: int, slider_pixel_length: float,
                                duration_beats: float):
        time_diff_bin_idx = quantize_to_bins(time_diff, DURATION_BINS)
        duration_bin_idx = quantize_to_bins(duration_beats, DURATION_BINS)
        
        return cls(
            x_diff=x_diff,
            y_diff=y_diff,
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


class BeatmapData(NamedTuple):
    """Comprehensive beatmap data structure - single source of truth for beatmap fields."""
    beatmap_id: int
    category: str
    hp_drain: float
    circle_size: float
    od: float
    ar: float
    slider_multiplier: float
    slider_tick: float
    main_bpm: float
    difficulty_rating: float
    vectors: List[HitObjectVector]

    @classmethod
    def get_field_names(cls):
        """Get field names for database table creation."""
        return [field for field in cls._fields if field != 'vectors']
    
    @classmethod
    def get_db_field_types(cls):
        """Get database field types for table creation."""
        return {
            'beatmap_id': 'INTEGER UNIQUE',
            'category': 'TEXT',
            'hp_drain': 'REAL',
            'circle_size': 'REAL', 
            'od': 'REAL',
            'ar': 'REAL',
            'slider_multiplier': 'REAL',
            'slider_tick': 'REAL',
            'main_bpm': 'REAL',
            'difficulty_rating': 'REAL'
        }
    
    def get_metadata(self) -> BeatmapMetadata:
        """Extract metadata from beatmap data."""
        return BeatmapMetadata(
            ar=self.ar,
            od=self.od,
            cs=self.circle_size,
            difficulty_rating=self.difficulty_rating,
            bpm=self.main_bpm
        )
    
    def to_db_tuple(self):
        """Convert to tuple for database insertion (excluding vectors)."""
        return (
            self.beatmap_id, self.category, self.hp_drain,
            self.circle_size, self.od, self.ar,
            self.slider_multiplier, self.slider_tick, 
            self.main_bpm, self.difficulty_rating
        )

VECTOR_DIM = HitObjectVector.get_vector_dim()
METADATA_DIM = BeatmapMetadata.get_metadata_dim()