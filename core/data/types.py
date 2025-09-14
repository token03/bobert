from typing import NamedTuple
import numpy as np

class HitObjectVector(NamedTuple):
    """Represents a single hit object's vector data."""
    distance_diff: float
    angle_cos: float
    angle_sin: float
    time_diff: float
    abs_x: float
    abs_y: float
    # One-hot encoded object types (3 values: circle, slider, spinner)
    is_circle: int
    is_slider: int
    is_spinner: int
    is_new_combo: int
    # One-hot encoded slider curve types (4 values: B, C, L, P)
    slider_curve_b: int
    slider_curve_c: int
    slider_curve_l: int
    slider_curve_p: int
    slider_num_anchors: int
    slider_pixel_length: float
    duration_beats: float

    @classmethod
    def get_field_names(cls):
        """Returns list of field names in order."""
        return list(cls._fields)
    
    @classmethod
    def get_vector_dim(cls):
        """Returns the total vector dimension."""
        return len(cls._fields)
    
    def to_array(self):
        """Convert to numpy array with proper type handling."""
        return np.array([
            float(self.distance_diff),
            float(self.angle_cos),
            float(self.angle_sin),
            float(self.time_diff),
            float(self.abs_x),
            float(self.abs_y),
            float(self.is_circle),
            float(self.is_slider),
            float(self.is_spinner),
            float(self.is_new_combo),
            float(self.slider_curve_b),
            float(self.slider_curve_c),
            float(self.slider_curve_l),
            float(self.slider_curve_p),
            float(self.slider_num_anchors),
            float(self.slider_pixel_length),
            float(self.duration_beats)
        ], dtype=np.float32)

class BeatmapMetadata(NamedTuple):
    """Represents beatmap metadata."""
    ar: float
    od: float
    cs: float
    difficulty_rating: float
    bpm: float

    @classmethod
    def get_field_names(cls):
        """Returns list of field names in order."""
        return list(cls._fields)
    
    @classmethod
    def get_metadata_dim(cls):
        """Returns the total metadata dimension."""
        return len(cls._fields)
    
    def to_array(self):
        """Convert to numpy array."""
        return np.array(self, dtype=np.float32)

VECTOR_DIM = HitObjectVector.get_vector_dim()
METADATA_DIM = BeatmapMetadata.get_metadata_dim()
