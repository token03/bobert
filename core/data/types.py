# types.py
from typing import NamedTuple, List, Dict, Any
import numpy as np
from enum import Enum

DURATION_BINS = [1/16, 1/12, 1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 4, 8, 16, 32, 64]

class NormalizationType(Enum):
    """Enumeration of normalization types."""
    CATEGORICAL = "categorical"
    STANDARD = "standard"      
    LOG = "log"               
    MINMAX = "minmax"        
    NONE = "none"             

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
    cos_angle: float
    sin_angle: float
    velocity: float
    cos_inner_angle: float
    sin_inner_angle: float
    object_type: int
    is_new_combo: int
    slider_curve_type: int
    slider_num_anchors: int
    slider_pixel_length: float
    time_diff_bin: int
    duration_bin: int
    kiai_time: int
    
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
            'names': field_names
        }
        return info

    @classmethod
    def get_normalization_specs(cls) -> Dict[str, NormalizationType]:
        """Get normalization specifications for each field."""
        return {
            'distance_diff': NormalizationType.LOG,
            'cos_angle': NormalizationType.STANDARD,
            'sin_angle': NormalizationType.STANDARD,
            'velocity': NormalizationType.LOG,
            'cos_inner_angle': NormalizationType.STANDARD,
            'sin_inner_angle': NormalizationType.STANDARD,
            'object_type': NormalizationType.CATEGORICAL,
            'is_new_combo': NormalizationType.CATEGORICAL,
            'slider_curve_type': NormalizationType.CATEGORICAL,
            'slider_num_anchors': NormalizationType.LOG,
            'slider_pixel_length': NormalizationType.LOG,
            'time_diff_bin': NormalizationType.CATEGORICAL,
            'duration_bin': NormalizationType.CATEGORICAL,
            'kiai_time': NormalizationType.CATEGORICAL
        }

    @classmethod
    def get_hit_object_feature_indices(cls) -> List[int]:
        """Get indices of features related to hit objects (position, type, timing)."""
        field_names = cls.get_field_names()
        hit_object_features = [
            'distance_diff', 'cos_angle', 'sin_angle', 'velocity',
            'cos_inner_angle', 'sin_inner_angle', 'object_type', 'is_new_combo', 
            'time_diff_bin', 'kiai_time'
        ]
        return [field_names.index(name) for name in hit_object_features]
    
    @classmethod 
    def get_slider_feature_indices(cls) -> List[int]:
        """Get indices of features related to sliders and duration."""
        field_names = cls.get_field_names()
        slider_features = ['slider_curve_type', 'slider_num_anchors', 'slider_pixel_length', 'duration_bin']
        return [field_names.index(name) for name in slider_features]
    
    @classmethod
    def get_hit_object_dim(cls) -> int:
        """Get dimension of hit object features."""
        return len(cls.get_hit_object_feature_indices())
    
    @classmethod
    def get_slider_dim(cls) -> int:
        """Get dimension of slider/duration features."""
        return len(cls.get_slider_feature_indices())

    @classmethod
    def create_with_quantization(cls, distance_diff: float, cos_angle: float, sin_angle: float,
                                velocity: float, cos_inner_angle: float, sin_inner_angle: float,
                                time_diff: float, object_type: int,
                                is_new_combo: int, slider_curve_type: int,
                                slider_num_anchors: int, slider_pixel_length: float,
                                duration_beats: float, kiai_time: int):
        time_diff_bin_idx = quantize_to_bins(time_diff, DURATION_BINS)
        duration_bin_idx = quantize_to_bins(duration_beats, DURATION_BINS)
        
        return cls(
            distance_diff=distance_diff,
            cos_angle=cos_angle,
            sin_angle=sin_angle,
            velocity=velocity,
            cos_inner_angle=cos_inner_angle,
            sin_inner_angle=sin_inner_angle,
            object_type=object_type,
            is_new_combo=is_new_combo,
            slider_curve_type=slider_curve_type,
            slider_num_anchors=slider_num_anchors,
            slider_pixel_length=slider_pixel_length,
            time_diff_bin=time_diff_bin_idx,
            duration_bin=duration_bin_idx,
            kiai_time=kiai_time
        )
    
    def to_array(self):
        return np.array(self, dtype=np.float32)

    @classmethod
    def get_field_descriptions(cls) -> Dict[str, str]:
        """Get human-readable descriptions for each field."""
        return {
            'distance_diff': 'Distance difference',
            'cos_angle': 'Cosine of angle',
            'sin_angle': 'Sine of angle',
            'velocity': 'Velocity',
            'cos_inner_angle': 'Cosine of inner angle',
            'sin_inner_angle': 'Sine of inner angle',
            'object_type': 'Object type (categorical)',
            'is_new_combo': 'New combo flag (categorical)', 
            'slider_curve_type': 'Slider curve type (categorical)',
            'slider_num_anchors': 'Number of anchors (log)',
            'slider_pixel_length': 'Slider pixel length (log)',
            'time_diff_bin': 'Time diff bin (categorical)',
            'duration_bin': 'Duration bin (categorical)',
            'kiai_time': 'Kiai time (categorical)',
        }

class BeatmapMetadata(NamedTuple):
    """Represents beatmap metadata."""
    ar: float
    od: float
    cs: float
    hp_drain: float
    slider_multiplier: float
    slider_tick: float
    difficulty_rating: float
    bpm: float

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
            'difficulty_rating': NormalizationType.STANDARD,
            'bpm': NormalizationType.LOG
        }
    
    def to_array(self):
        return np.array(self, dtype=np.float32)

    @classmethod
    def get_field_descriptions(cls) -> Dict[str, str]:
        """Get human-readable descriptions for each field."""
        return {
            'ar': 'Approach Rate',
            'od': 'Overall Difficulty',
            'cs': 'Circle Size',
            'hp_drain': 'HP Drain Rate',
            'slider_multiplier': 'Slider Multiplier',
            'slider_tick': 'Slider Tick Rate',
            'difficulty_rating': 'Star Rating',
            'bpm': 'Beats Per Minute (log)'
        }


class BeatmapData(NamedTuple):
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
            hp_drain=self.hp_drain,
            slider_multiplier=self.slider_multiplier,
            slider_tick=self.slider_tick,
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