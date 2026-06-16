from typing import NamedTuple, List, Dict
from enum import Enum

OSU_STAGE_WIDTH = 512
OSU_STAGE_HEIGHT = 384
CENTER_X = OSU_STAGE_WIDTH / 2.0
CENTER_Y = OSU_STAGE_HEIGHT / 2.0
DEFAULT_PRE_START_MS = 200.0

MAX_METER_CARDINALITY = 8

DURATION_BINS = [
    1 / 16,
    1 / 12,
    1 / 9,
    1 / 8,
    1 / 7,
    1 / 6,
    1 / 5,
    1 / 4,
    1 / 3,
    1 / 2,
    1,
    2,
    4,
    8,
    16,
    32,
    64,
]

CANONICAL_BPM_MIN = 120.0
CANONICAL_BPM_MAX = 240.0

class NormalizationType(Enum):
    CATEGORICAL = "categorical"
    STANDARD = "standard"
    NONE = "none"


OBJECT_TYPE_CIRCLE = 0
OBJECT_TYPE_SLIDER_HEAD = 1
OBJECT_TYPE_SLIDER_END = 2
OBJECT_TYPE_SPINNER_START = 3
OBJECT_TYPE_SPINNER_END = 4

SLIDER_TYPE_INDEX = OBJECT_TYPE_SLIDER_HEAD


class Feature(NamedTuple):
    name: str
    norm: NormalizationType
    cardinality: int | None = None
    slider_only: bool = False


FEATURES = [
    Feature("norm_x", NormalizationType.NONE),
    Feature("norm_y", NormalizationType.NONE),
    Feature("delta_x", NormalizationType.STANDARD),
    Feature("delta_y", NormalizationType.STANDARD),
    Feature("log_time_diff_ms", NormalizationType.STANDARD),
    Feature("notes_per_second", NormalizationType.STANDARD),
    Feature("velocity", NormalizationType.STANDARD),
    Feature("relative_cos", NormalizationType.NONE),
    Feature("relative_sin", NormalizationType.NONE),
    Feature("rhythm_change", NormalizationType.STANDARD),
    Feature("log_slider_pixel_length", NormalizationType.STANDARD, slider_only=True),
    Feature("log_slider_repeats", NormalizationType.STANDARD, slider_only=True),
    Feature("slider_tortuosity", NormalizationType.STANDARD, slider_only=True),
    Feature("beat_in_measure", NormalizationType.CATEGORICAL, MAX_METER_CARDINALITY),
    Feature("object_type", NormalizationType.CATEGORICAL, 5),
    Feature("is_new_combo", NormalizationType.CATEGORICAL, 2),
    Feature("time_diff_bin", NormalizationType.CATEGORICAL, len(DURATION_BINS)),
    Feature("rhythmic_snap", NormalizationType.CATEGORICAL, 6),
]

CATEGORICAL_FEATURE_ORDER = (
    "object_type",
    "is_new_combo",
    "beat_in_measure",
    "time_diff_bin",
    "rhythmic_snap",
)


class HitObject(NamedTuple):
    norm_x: float
    norm_y: float
    delta_x: float
    delta_y: float
    log_time_diff_ms: float
    notes_per_second: float
    velocity: float
    relative_cos: float
    relative_sin: float
    rhythm_change: float

    log_slider_pixel_length: float
    log_slider_repeats: float
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
    def get_slider_only_features() -> List[str]:
        return [feature.name for feature in FEATURES if feature.slider_only]

    @classmethod
    def get_vector_dim(cls):
        return len(cls._fields)

    @classmethod
    def get_feature_info(cls):
        field_names = cls.get_field_names()
        index = {name: i for i, name in enumerate(field_names)}
        features = {feature.name: feature for feature in FEATURES}
        continuous = [
            feature
            for feature in FEATURES
            if feature.norm is not NormalizationType.CATEGORICAL
        ]

        info = {
            "categorical": {
                name: {
                    "index": index[name],
                    "cardinality": features[name].cardinality,
                }
                for name in CATEGORICAL_FEATURE_ORDER
            },
            "continuous": {
                feature.name: index[feature.name] for feature in continuous
            },
            "slider": {
                feature.name: index[feature.name]
                for feature in FEATURES
                if feature.slider_only
            },
            "names": field_names,
        }
        return info

    @classmethod
    def get_normalization_specs(cls) -> Dict[str, NormalizationType]:
        return {feature.name: feature.norm for feature in FEATURES}


VECTOR_DIM = HitObject.get_vector_dim()
