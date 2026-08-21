import bisect
import io
import math
import os
from collections.abc import Iterable, Iterator
from typing import NamedTuple

OBJECT_TYPE_CIRCLE = 0
OBJECT_TYPE_SLIDER = 1
OBJECT_TYPE_SPINNER = 2

MAX_TIME_MS = 36000000
MAX_COORDINATE = 100000
INT32_MAX = 2**31 - 1
BEZIER_TOLERANCE = 0.25
CATMULL_DETAIL = 50
ACTIVE_SECTIONS = frozenset(("metadata", "difficulty", "timingpoints", "hitobjects"))
CATMULL_T = tuple(
    (t, t * t, t * t * t)
    for t in (step / CATMULL_DETAIL for step in range(CATMULL_DETAIL))
)

Point = tuple[float, float]

DIFFICULTY_KEYS = {
    "hpdrainrate": "hp_drain",
    "circlesize": "cs",
    "overalldifficulty": "od",
    "approachrate": "ar",
    "slidermultiplier": "slider_multiplier",
    "slidertickrate": "slider_tick",
    "difficultyrating": "difficulty_rating",
}


class RawTimingPoint(NamedTuple):
    time: int
    beat_length: float
    uninherited: bool
    effects: int


class RawHitObject(NamedTuple):
    object_index: int
    x: int
    y: int
    time: int
    object_type: int
    is_new_combo: int
    curve_type: str | None
    slides: int | None
    pixel_length: float | None
    end_time: int
    hit_sound: int
    bpm: float
    timing_origin: int
    end_bpm: float
    end_timing_origin: int
    kiai_time: int
    num_anchors: int
    hard_anchor_ratio: float
    slider_end_x: int
    slider_end_y: int
    slider_path_valid: int
    span_end_dx: float
    span_end_dy: float
    curve_residual_1_dx: float
    curve_residual_1_dy: float
    curve_residual_2_dx: float
    curve_residual_2_dy: float


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
    timing_points: list[RawTimingPoint]
    hit_objects: list[RawHitObject]


def extract_beatmap_record(beatmap: RawBeatmap) -> dict:
    return {
        "beatmap_id": beatmap.beatmap_id,
        "category": beatmap.category,
        "hp_drain": beatmap.hp_drain,
        "cs": beatmap.cs,
        "od": beatmap.od,
        "ar": beatmap.ar,
        "slider_multiplier": beatmap.slider_multiplier,
        "slider_tick": beatmap.slider_tick,
        "difficulty_rating": beatmap.difficulty_rating,
    }


def extract_hitobject_records(beatmap: RawBeatmap) -> list[dict]:
    return [
        {
            "beatmap_id": beatmap.beatmap_id,
            "category": beatmap.category,
            "object_index": hitobject.object_index,
            "x": hitobject.x,
            "y": hitobject.y,
            "time": hitobject.time,
            "object_type": hitobject.object_type,
            "is_new_combo": hitobject.is_new_combo,
            "hit_sound": hitobject.hit_sound,
            "end_time": hitobject.end_time,
            "pixel_length": hitobject.pixel_length or 0.0,
            "bpm": hitobject.bpm,
            "timing_origin": hitobject.timing_origin,
            "end_bpm": hitobject.end_bpm,
            "end_timing_origin": hitobject.end_timing_origin,
            "curve_type_char": hitobject.curve_type or "",
            "num_anchors": hitobject.num_anchors,
            "kiai_time": hitobject.kiai_time,
            "slider_repeats": (
                hitobject.slides - 1 if hitobject.slides is not None else 0
            ),
            "hard_anchor_ratio": hitobject.hard_anchor_ratio,
            "slider_end_x": hitobject.slider_end_x,
            "slider_end_y": hitobject.slider_end_y,
            "slider_path_valid": hitobject.slider_path_valid,
            "span_end_dx": hitobject.span_end_dx,
            "span_end_dy": hitobject.span_end_dy,
            "curve_residual_1_dx": hitobject.curve_residual_1_dx,
            "curve_residual_1_dy": hitobject.curve_residual_1_dy,
            "curve_residual_2_dx": hitobject.curve_residual_2_dx,
            "curve_residual_2_dy": hitobject.curve_residual_2_dy,
        }
        for hitobject in beatmap.hit_objects
    ]


class TimingSection(NamedTuple):
    start_time: int
    beat_length: float
    bpm: float
    timing_origin: int
    sv_multiplier: float
    kiai: int


def _flatten_bezier_vertices(points: list[Point], depth: int = 0) -> Iterator[Point]:
    n = len(points)
    if n == 1:
        return
    if n == 2:
        yield points[1]
        return

    if n in (3, 4):
        curvature = max(
            math.hypot(
                points[index - 1][0] - 2 * points[index][0] + points[index + 1][0],
                points[index - 1][1] - 2 * points[index][1] + points[index + 1][1],
            )
            for index in range(1, n - 1)
        )
        steps = max(1, math.ceil(math.sqrt(curvature / (8 * BEZIER_TOLERANCE))))
        if n == 3:
            a, b, c = points
            for step in range(1, steps + 1):
                t = step / steps
                s = 1 - t
                yield (
                    s * s * a[0] + 2 * s * t * b[0] + t * t * c[0],
                    s * s * a[1] + 2 * s * t * b[1] + t * t * c[1],
                )
        else:
            a, b, c, d = points
            for step in range(1, steps + 1):
                t = step / steps
                s = 1 - t
                yield (
                    s**3 * a[0]
                    + 3 * s * s * t * b[0]
                    + 3 * s * t * t * c[0]
                    + t**3 * d[0],
                    s**3 * a[1]
                    + 3 * s * s * t * b[1]
                    + 3 * s * t * t * c[1]
                    + t**3 * d[1],
                )
        return

    tolerance = BEZIER_TOLERANCE * BEZIER_TOLERANCE * 4
    if (
        all(
            (points[i - 1][0] - 2 * points[i][0] + points[i + 1][0]) ** 2
            + (points[i - 1][1] - 2 * points[i][1] + points[i + 1][1]) ** 2
            <= tolerance
            for i in range(1, n - 1)
        )
        or depth >= 16
    ):
        yield points[-1]
        return

    left = [points[0]]
    right = [points[-1]]
    work = points
    while len(work) > 1:
        work = [
            (
                (work[i][0] + work[i + 1][0]) * 0.5,
                (work[i][1] + work[i + 1][1]) * 0.5,
            )
            for i in range(len(work) - 1)
        ]
        left.append(work[0])
        right.append(work[-1])

    yield from _flatten_bezier_vertices(left, depth + 1)
    yield from _flatten_bezier_vertices(list(reversed(right)), depth + 1)


def _split_bezier_vertices(points: list[Point]) -> Iterator[Point]:
    yield points[0]
    segment = [points[0]]
    for point in points[1:]:
        if point == segment[-1]:
            if len(segment) > 1:
                yield from _flatten_bezier_vertices(segment)
            segment = [point]
        else:
            segment.append(point)
    yield from _flatten_bezier_vertices(segment)


def _perfect_vertices(points: list[Point]) -> Iterator[Point]:
    a, b, c = points
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(cross) < 1e-7:
        yield from _split_bezier_vertices(points)
        return

    a2 = a[0] * a[0] + a[1] * a[1]
    b2 = b[0] * b[0] + b[1] * b[1]
    c2 = c[0] * c[0] + c[1] * c[1]
    divisor = 2 * cross
    center = (
        (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / divisor,
        (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / divisor,
    )
    radius = math.hypot(a[0] - center[0], a[1] - center[1])
    if not math.isfinite(radius) or radius <= 0:
        yield from _split_bezier_vertices(points)
        return

    start_angle = math.atan2(a[1] - center[1], a[0] - center[0])
    end_angle = math.atan2(c[1] - center[1], c[0] - center[0])
    angle = end_angle - start_angle
    if cross > 0:
        angle %= 2 * math.pi
    else:
        angle = -((-angle) % (2 * math.pi))
    max_step = 2 * math.acos(max(-1.0, 1 - BEZIER_TOLERANCE / radius))
    segments = max(1, math.ceil(abs(angle) / max_step)) if max_step > 0 else 1
    for index in range(segments + 1):
        theta = start_angle + angle * index / segments
        yield (
            center[0] + radius * math.cos(theta),
            center[1] + radius * math.sin(theta),
        )


def _catmull_vertices(points: list[Point]) -> Iterator[Point]:
    for index in range(len(points) - 1):
        v1 = points[index - 1] if index > 0 else points[index]
        v2 = points[index]
        v3 = points[index + 1]
        v4 = (
            points[index + 2]
            if index + 2 < len(points)
            else (
                2 * v3[0] - v2[0],
                2 * v3[1] - v2[1],
            )
        )
        for t, t2, t3 in CATMULL_T:
            yield (
                0.5
                * (
                    2 * v2[0]
                    + (-v1[0] + v3[0]) * t
                    + (2 * v1[0] - 5 * v2[0] + 4 * v3[0] - v4[0]) * t2
                    + (-v1[0] + 3 * v2[0] - 3 * v3[0] + v4[0]) * t3
                ),
                0.5
                * (
                    2 * v2[1]
                    + (-v1[1] + v3[1]) * t
                    + (2 * v1[1] - 5 * v2[1] + 4 * v3[1] - v4[1]) * t2
                    + (-v1[1] + 3 * v2[1] - 3 * v3[1] + v4[1]) * t3
                ),
            )
    yield points[-1]


def _path_vertices(curve_type: str, points: list[Point]) -> Iterable[Point]:
    if curve_type == "L":
        return points
    if curve_type == "B":
        return _split_bezier_vertices(points)
    if curve_type == "P":
        return (
            _perfect_vertices(points)
            if len(points) == 3
            else _split_bezier_vertices(points)
        )
    if curve_type == "C":
        return _catmull_vertices(points)
    raise ValueError("unknown slider path type")


def _slider_geometry(
    curve_type: str, points: list[Point], expected_length: float
) -> tuple[float, float, float, float, float, float, float]:
    invalid = (0.0,) * 7
    if not points or expected_length < 0 or not math.isfinite(expected_length):
        return invalid
    if curve_type == "L" and len(points) == 2:
        dx = points[1][0] - points[0][0]
        dy = points[1][1] - points[0][1]
        length = math.hypot(dx, dy)
        if length == 0:
            return (
                (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) if expected_length == 0 else invalid
            )
        scale = expected_length / length
        return (1.0, dx * scale, dy * scale, 0.0, 0.0, 0.0, 0.0)

    targets = (expected_length / 3, expected_length * 2 / 3, expected_length)
    samples: list[Point] = []
    previous: Point | None = None
    last_direction: Point | None = None
    distance = 0.0
    try:
        for vertex in _path_vertices(curve_type, points):
            if previous is None:
                previous = vertex
                while len(samples) < 3 and targets[len(samples)] == 0:
                    samples.append(vertex)
                continue
            segment_length = math.hypot(
                vertex[0] - previous[0], vertex[1] - previous[1]
            )
            if not math.isfinite(segment_length):
                return invalid
            if segment_length > 0:
                while (
                    len(samples) < 3
                    and targets[len(samples)] <= distance + segment_length
                ):
                    ratio = (targets[len(samples)] - distance) / segment_length
                    samples.append(
                        (
                            previous[0] + (vertex[0] - previous[0]) * ratio,
                            previous[1] + (vertex[1] - previous[1]) * ratio,
                        )
                    )
                last_direction = (
                    (vertex[0] - previous[0]) / segment_length,
                    (vertex[1] - previous[1]) / segment_length,
                )
                distance += segment_length
            previous = vertex
            if len(samples) == 3:
                break
    except (ArithmeticError, ValueError):
        return invalid

    if previous is None:
        return invalid
    if len(samples) < 3:
        if last_direction is None:
            if expected_length > 0:
                return invalid
            samples = [points[0], points[0], points[0]]
        else:
            samples.extend(
                (
                    previous[0] + last_direction[0] * (target - distance),
                    previous[1] + last_direction[1] * (target - distance),
                )
                for target in targets[len(samples) :]
            )

    start = points[0]
    one, two, end = samples
    end_dx, end_dy = end[0] - start[0], end[1] - start[1]
    return (
        1.0,
        float(end_dx),
        float(end_dy),
        float(one[0] - start[0] - end_dx / 3),
        float(one[1] - start[1] - end_dy / 3),
        float(two[0] - start[0] - end_dx * 2 / 3),
        float(two[1] - start[1] - end_dy * 2 / 3),
    )


def _preprocess_timing_points(
    timing_points: list[RawTimingPoint],
) -> list[TimingSection]:
    sections = []
    last_uninherited = None
    for point in timing_points:
        if point.uninherited:
            last_uninherited = point
        if last_uninherited is None:
            last_uninherited = RawTimingPoint(
                time=point.time,
                beat_length=500.0,
                uninherited=True,
                effects=0,
            )

        beat_length = last_uninherited.beat_length
        bpm = 60000.0 / beat_length if beat_length > 0 else 120.0
        sv_multiplier = 1.0
        if not point.uninherited and point.beat_length < 0:
            sv_multiplier = -100.0 / point.beat_length
        sections.append(
            TimingSection(
                start_time=point.time,
                beat_length=beat_length,
                bpm=bpm,
                timing_origin=last_uninherited.time,
                sv_multiplier=sv_multiplier,
                kiai=1 if point.effects & 1 else 0,
            )
        )
    return sections


def parse_osu_file(
    file_path: str,
    max_hitobject_lines: int | None = None,
    max_curve_points: int | None = None,
    *,
    validate_dataset: bool = False,
    _content: bytes | None = None,
    _beatmap_id: int | None = None,
) -> RawBeatmap | None:
    if _content is None:
        try:
            filename_beatmap_id = int(os.path.splitext(os.path.basename(file_path))[0])
        except ValueError:
            filename_beatmap_id = None
        category = os.path.basename(os.path.dirname(file_path))
        source = open(  # noqa: SIM115
            file_path, "r", encoding="utf-8", errors="ignore"
        )
    else:
        filename_beatmap_id = _beatmap_id
        category = ""
        source = io.TextIOWrapper(
            io.BytesIO(_content), encoding="utf-8", errors="ignore"
        )

    data = {
        "beatmap_id": filename_beatmap_id,
        "hp_drain": 5.0,
        "cs": 5.0,
        "od": 5.0,
        "ar": 5.0,
        "slider_multiplier": 1.4,
        "slider_tick": 1.0,
        "category": category,
        "difficulty_rating": 0.0,
    }

    timing_points = []
    timing_sections = None
    dataset_valid_sections = []
    section_start_times = []
    timing_index = -1
    previous_object_time = -1
    hit_objects = []
    hitobject_line_count = 0
    curve_point_count = 0
    section_name = ""

    with source as file:
        for raw_line in file:
            if section_name not in ACTIVE_SECTIONS and not raw_line.startswith("["):
                continue

            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section_name = line[1:-1].lower()
                continue

            if section_name == "metadata":
                if filename_beatmap_id is None and line.lower().startswith(
                    "beatmapid:"
                ):
                    data["beatmap_id"] = int(line.split(":", 1)[1])
                continue

            if section_name == "difficulty":
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    if key in DIFFICULTY_KEYS:
                        try:
                            data[DIFFICULTY_KEYS[key]] = float(parts[1].strip())
                        except ValueError:
                            pass
                continue

            if section_name == "timingpoints":
                parts = line.split(",", 8)
                if len(parts) >= 2:
                    beat_length = float(parts[1])
                    if math.isfinite(beat_length) and beat_length != 0:
                        timing_points.append(
                            RawTimingPoint(
                                time=int(float(parts[0])),
                                beat_length=beat_length,
                                uninherited=len(parts) >= 7 and parts[6] == "1",
                                effects=int(parts[7]) if len(parts) >= 8 else 0,
                            )
                        )
                continue

            if section_name != "hitobjects":
                continue

            hitobject_line_count += 1
            if (
                max_hitobject_lines is not None
                and hitobject_line_count > max_hitobject_lines
            ):
                return None

            parts = line.split(",", 8)
            try:
                x = int(parts[0])
                y = int(parts[1])
                time = int(parts[2])
                type_flags = int(parts[3])
                hit_sound = int(parts[4])
            except (ValueError, IndexError):
                continue

            is_circle = type_flags & 1
            is_slider = type_flags & 2
            is_spinner = type_flags & 8
            if is_circle:
                object_type = OBJECT_TYPE_CIRCLE
            elif is_slider:
                object_type = OBJECT_TYPE_SLIDER
            elif is_spinner:
                object_type = OBJECT_TYPE_SPINNER
            else:
                continue

            curve_data = parts[5].split("|") if is_slider and len(parts) > 5 else None
            if curve_data is not None:
                curve_point_count += len(curve_data) - 1
                if (
                    max_curve_points is not None
                    and curve_point_count > max_curve_points
                ):
                    return None

            if abs(x) > MAX_COORDINATE or abs(y) > MAX_COORDINATE:
                continue
            if validate_dataset and not -(2**31) <= hit_sound <= INT32_MAX:
                return None

            if timing_sections is None:
                if not timing_points:
                    return None
                timing_points.sort(key=lambda point: point.time)
                timing_sections = _preprocess_timing_points(timing_points)
                if validate_dataset:
                    dataset_valid_sections = [
                        math.isfinite(section.bpm)
                        and abs(section.bpm) <= 3.4028235e38
                        and -(2**31) <= section.timing_origin < 2**31
                        for section in timing_sections
                    ]
                section_start_times = [item.start_time for item in timing_sections]

            if time >= previous_object_time:
                while (
                    timing_index + 1 < len(section_start_times)
                    and section_start_times[timing_index + 1] <= time
                ):
                    timing_index += 1
            else:
                timing_index = bisect.bisect_right(section_start_times, time) - 1
            previous_object_time = time

            active_section = (
                timing_sections[timing_index] if timing_index >= 0 else None
            )
            if (
                validate_dataset
                and timing_index >= 0
                and not dataset_valid_sections[timing_index]
            ):
                return None
            bpm = active_section.bpm if active_section is not None else 120.0
            timing_origin = (
                active_section.timing_origin if active_section is not None else 0
            )
            kiai = active_section.kiai if active_section is not None else 0

            curve_type = None
            slides = None
            pixel_length = None
            end_time = time
            num_anchors = 0
            num_hard_anchors = 0
            slider_end_x = 0
            slider_end_y = 0
            slider_geometry = (0.0,) * 7

            if object_type == OBJECT_TYPE_SLIDER:
                try:
                    assert curve_data is not None
                    curve_type = curve_data[0]
                    control_points: list[Point] = [(float(x), float(y))]
                    path_valid = bool(curve_type)
                    previous_point = None
                    previous_was_hard = False
                    for raw_point in curve_data[1:]:
                        try:
                            coordinates = raw_point.split(":", 2)
                            point = (int(coordinates[0]), int(coordinates[1]))
                        except (ValueError, IndexError):
                            path_valid = False
                            continue
                        if validate_dataset and (
                            abs(point[0]) > MAX_COORDINATE
                            or abs(point[1]) > MAX_COORDINATE
                        ):
                            path_valid = False
                            continue
                        control_points.append((float(point[0]), float(point[1])))
                        if point == previous_point:
                            if not previous_was_hard:
                                num_hard_anchors += 1
                                previous_was_hard = True
                            continue
                        previous_point = point
                        previous_was_hard = False
                        num_anchors += 1
                        slider_end_x, slider_end_y = point

                    slides = int(parts[6])
                    pixel_length = float(parts[7])
                    if validate_dataset and not 1 <= slides <= 2**31:
                        return None
                    if path_valid:
                        slider_geometry = _slider_geometry(
                            curve_type, control_points, pixel_length
                        )
                    if active_section is not None and math.isfinite(pixel_length):
                        slider_velocity = (
                            data["slider_multiplier"]
                            * 100.0
                            * active_section.sv_multiplier
                        )
                        if active_section.beat_length > 0 and slider_velocity > 0:
                            time_per_slide_ms = (
                                pixel_length / slider_velocity
                            ) * active_section.beat_length
                            end_time = time + int(time_per_slide_ms * slides)
                except (ValueError, IndexError):
                    continue
            elif object_type == OBJECT_TYPE_SPINNER:
                try:
                    end_time = int(parts[5])
                except (ValueError, IndexError):
                    continue

            if time > MAX_TIME_MS or end_time > MAX_TIME_MS:
                continue

            end_bpm = bpm
            end_timing_origin = timing_origin
            if end_time != time:
                end_index = bisect.bisect_right(section_start_times, end_time) - 1
                if end_index >= 0:
                    if validate_dataset and not dataset_valid_sections[end_index]:
                        return None
                    end_section = timing_sections[end_index]
                    end_bpm = end_section.bpm
                    end_timing_origin = end_section.timing_origin

            hard_anchor_ratio = (
                num_hard_anchors / num_anchors if num_anchors > 0 else 0.0
            )
            if validate_dataset and not (
                -(2**31) <= time < 2**31
                and -(2**31) <= end_time < 2**31
                and (
                    object_type != OBJECT_TYPE_SLIDER
                    or all(
                        math.isfinite(value) and abs(value) <= 3.4028235e38
                        for value in (pixel_length or 0.0, *slider_geometry[1:])
                    )
                )
            ):
                return None
            hit_objects.append(
                RawHitObject(
                    object_index=len(hit_objects),
                    x=x,
                    y=y,
                    time=time,
                    object_type=object_type,
                    is_new_combo=1 if type_flags & 4 else 0,
                    curve_type=curve_type,
                    slides=slides,
                    pixel_length=pixel_length,
                    end_time=end_time,
                    hit_sound=hit_sound,
                    bpm=bpm,
                    timing_origin=timing_origin,
                    end_bpm=end_bpm,
                    end_timing_origin=end_timing_origin,
                    kiai_time=kiai,
                    num_anchors=num_anchors,
                    hard_anchor_ratio=hard_anchor_ratio,
                    slider_end_x=slider_end_x,
                    slider_end_y=slider_end_y,
                    slider_path_valid=int(slider_geometry[0]),
                    span_end_dx=slider_geometry[1],
                    span_end_dy=slider_geometry[2],
                    curve_residual_1_dx=slider_geometry[3],
                    curve_residual_1_dy=slider_geometry[4],
                    curve_residual_2_dx=slider_geometry[5],
                    curve_residual_2_dy=slider_geometry[6],
                )
            )

    if (
        data["beatmap_id"] is None
        or not timing_points
        or not hit_objects
        or (
            validate_dataset
            and (
                not -(2**63) <= data["beatmap_id"] < 2**63
                or len(hit_objects) < 2
                or len(hit_objects) > 16_384
                or not all(
                    math.isfinite(data[field]) and abs(data[field]) <= 3.4028235e38
                    for field in DIFFICULTY_KEYS.values()
                )
            )
        )
    ):
        return None
    return RawBeatmap(**data, timing_points=timing_points, hit_objects=hit_objects)


def parse_osu_bytes(
    content: bytes,
    beatmap_id: int | None = None,
    max_hitobject_lines: int | None = None,
    max_curve_points: int | None = None,
) -> RawBeatmap | None:
    return parse_osu_file(
        "",
        max_hitobject_lines=max_hitobject_lines,
        max_curve_points=max_curve_points,
        _content=content,
        _beatmap_id=beatmap_id,
    )
