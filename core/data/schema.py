from typing import NamedTuple
from enum import Enum

MAP_FEATURE_ATTRIBUTES = [
    "ar",
    "cs",
    "od",
    "hp_drain",
    "drain_time",
    "slider_multiplier",
]

OSU_STAGE_WIDTH = 512
OSU_STAGE_HEIGHT = 384
CENTER_X = OSU_STAGE_WIDTH / 2.0
CENTER_Y = OSU_STAGE_HEIGHT / 2.0
DEFAULT_PRE_START_MS = 200.0

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
BEAT_PHASE_CARDINALITY = 49


class NormalizationType(Enum):
    CATEGORICAL = "categorical"
    STANDARD = "standard"
    NONE = "none"


OBJECT_TYPE_CIRCLE = 0
OBJECT_TYPE_SLIDER = 1
OBJECT_TYPE_SPINNER = 2


class Feature(NamedTuple):
    name: str
    norm: NormalizationType
    cardinality: int | None = None
    conditional: str | None = None


FEATURES = [
    Feature("norm_x", NormalizationType.NONE),
    Feature("norm_y", NormalizationType.NONE),
    Feature("incoming_dx", NormalizationType.STANDARD),
    Feature("incoming_dy", NormalizationType.STANDARD),
    Feature("log_onset_ioi_ms", NormalizationType.STANDARD),
    Feature("log_span_duration_ms", NormalizationType.STANDARD, conditional="slider"),
    Feature("log_span_length", NormalizationType.STANDARD, conditional="slider"),
    Feature("log_span_count", NormalizationType.STANDARD, conditional="slider"),
    Feature("span_end_dx", NormalizationType.STANDARD, conditional="slider"),
    Feature("span_end_dy", NormalizationType.STANDARD, conditional="slider"),
    Feature("curve_residual_1_dx", NormalizationType.STANDARD, conditional="slider"),
    Feature("curve_residual_1_dy", NormalizationType.STANDARD, conditional="slider"),
    Feature("curve_residual_2_dx", NormalizationType.STANDARD, conditional="slider"),
    Feature("curve_residual_2_dy", NormalizationType.STANDARD, conditional="slider"),
    Feature(
        "log_spinner_duration_ms", NormalizationType.STANDARD, conditional="spinner"
    ),
    Feature("object_type", NormalizationType.CATEGORICAL, 3),
    Feature("is_new_combo", NormalizationType.CATEGORICAL, 2),
    Feature("onset_duration_bin", NormalizationType.CATEGORICAL, len(DURATION_BINS)),
    Feature("beat_phase", NormalizationType.CATEGORICAL, BEAT_PHASE_CARDINALITY),
    Feature("incoming_motion_valid", NormalizationType.CATEGORICAL, 2),
    Feature(
        "span_duration_bin",
        NormalizationType.CATEGORICAL,
        len(DURATION_BINS),
        conditional="slider",
    ),
    Feature("ends_at_head", NormalizationType.CATEGORICAL, 2, conditional="slider"),
    Feature(
        "spinner_duration_bin",
        NormalizationType.CATEGORICAL,
        len(DURATION_BINS),
        conditional="spinner",
    ),
]

CATEGORICAL_FEATURE_ORDER = (
    "object_type",
    "is_new_combo",
    "onset_duration_bin",
    "beat_phase",
    "incoming_motion_valid",
    "span_duration_bin",
    "ends_at_head",
    "spinner_duration_bin",
)


FIELD_NAMES = tuple(feature.name for feature in FEATURES)
VECTOR_DIM = len(FIELD_NAMES)
FEATURE_INDEX = {name: i for i, name in enumerate(FIELD_NAMES)}
FEATURES_BY_NAME = {feature.name: feature for feature in FEATURES}
SLIDER_ONLY_FEATURES = tuple(
    feature.name for feature in FEATURES if feature.conditional == "slider"
)
SPINNER_ONLY_FEATURES = tuple(
    feature.name for feature in FEATURES if feature.conditional == "spinner"
)
NORMALIZATION_SPECS = {feature.name: feature.norm for feature in FEATURES}
FEATURE_INFO = {
    "categorical": {
        name: {
            "index": FEATURE_INDEX[name],
            "cardinality": FEATURES_BY_NAME[name].cardinality,
        }
        for name in CATEGORICAL_FEATURE_ORDER
    },
    "continuous": {
        feature.name: FEATURE_INDEX[feature.name]
        for feature in FEATURES
        if feature.norm is not NormalizationType.CATEGORICAL
    },
    "slider": {name: FEATURE_INDEX[name] for name in SLIDER_ONLY_FEATURES},
    "spinner": {name: FEATURE_INDEX[name] for name in SPINNER_ONLY_FEATURES},
    "common": {
        feature.name: FEATURE_INDEX[feature.name]
        for feature in FEATURES
        if feature.conditional is None
    },
    "names": FIELD_NAMES,
}
