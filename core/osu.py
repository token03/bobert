import bisect
import io
import math
from collections.abc import Collection, Iterable, Iterator
from typing import Final, NamedTuple

OBJECT_TYPE_CIRCLE: Final = 0
OBJECT_TYPE_SLIDER: Final = 1
OBJECT_TYPE_SPINNER: Final = 2
OBJECT_TYPE_HOLD: Final = 3

MODE_MANIA: Final = 3

MAX_TIME_MS: Final = 36000000
MAX_COORDINATE: Final = 100000
MAX_OBJECTS: Final = 16384
INT32_MIN: Final = -(2**31)
INT32_MAX: Final = 2**31 - 1
INT64_MIN: Final = -(2**63)
INT64_MAX: Final = 2**63 - 1
FLOAT32_MAX: Final = 3.4028235e38
DEFAULT_BPM: Final = 120.0
BEZIER_TOLERANCE: Final = 0.25
BEZIER_MAX_DEPTH: Final = 16
CATMULL_DETAIL: Final = 50
CATMULL_T = tuple(
    (t, t * t, t * t * t)
    for t in (step / CATMULL_DETAIL for step in range(CATMULL_DETAIL))
)
ACTIVE_SECTIONS = frozenset(
    ("general", "metadata", "difficulty", "timingpoints", "hitobjects")
)
DIFFICULTY_KEYS = {
    "hpdrainrate": "hp_drain",
    "circlesize": "cs",
    "overalldifficulty": "od",
    "approachrate": "ar",
    "slidermultiplier": "slider_multiplier",
    "slidertickrate": "slider_tick",
    "difficultyrating": "difficulty_rating",
}
DEFAULT_DIFFICULTY = {
    "hp_drain": 5.0,
    "cs": 5.0,
    "od": 5.0,
    "ar": 5.0,
    "slider_multiplier": 1.4,
    "slider_tick": 1.0,
    "difficulty_rating": 0.0,
}

Point = tuple[float, float]


class Rejected(Exception):
    pass


class SliderShape(NamedTuple):
    valid: int
    end_dx: float
    end_dy: float
    residual_1_dx: float
    residual_1_dy: float
    residual_2_dx: float
    residual_2_dy: float


INVALID_SHAPE = SliderShape(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
POINT_SHAPE = SliderShape(1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class HitObject(NamedTuple):
    x: int
    y: int
    time: int
    object_type: int
    is_new_combo: int
    hit_sound: int
    end_time: int
    pixel_length: float
    bpm: float
    end_bpm: float
    num_anchors: int
    slider_repeats: int
    hard_anchor_ratio: float
    shape: SliderShape


class HitObjects:
    def __init__(self) -> None:
        self.object_index: list[int] = []
        self.x: list[int] = []
        self.y: list[int] = []
        self.time: list[int] = []
        self.object_type: list[int] = []
        self.is_new_combo: list[int] = []
        self.hit_sound: list[int] = []
        self.end_time: list[int] = []
        self.pixel_length: list[float] = []
        self.bpm: list[float] = []
        self.end_bpm: list[float] = []
        self.num_anchors: list[int] = []
        self.slider_repeats: list[int] = []
        self.hard_anchor_ratio: list[float] = []
        self.slider_path_valid: list[int] = []
        self.span_end_dx: list[float] = []
        self.span_end_dy: list[float] = []
        self.curve_residual_1_dx: list[float] = []
        self.curve_residual_1_dy: list[float] = []
        self.curve_residual_2_dx: list[float] = []
        self.curve_residual_2_dy: list[float] = []

    def __len__(self) -> int:
        return len(self.time)

    def append(self, hit_object: HitObject) -> None:
        shape = hit_object.shape
        self.object_index.append(len(self.time))
        self.x.append(hit_object.x)
        self.y.append(hit_object.y)
        self.time.append(hit_object.time)
        self.object_type.append(hit_object.object_type)
        self.is_new_combo.append(hit_object.is_new_combo)
        self.hit_sound.append(hit_object.hit_sound)
        self.end_time.append(hit_object.end_time)
        self.pixel_length.append(hit_object.pixel_length)
        self.bpm.append(hit_object.bpm)
        self.end_bpm.append(hit_object.end_bpm)
        self.num_anchors.append(hit_object.num_anchors)
        self.slider_repeats.append(hit_object.slider_repeats)
        self.hard_anchor_ratio.append(hit_object.hard_anchor_ratio)
        self.slider_path_valid.append(shape.valid)
        self.span_end_dx.append(shape.end_dx)
        self.span_end_dy.append(shape.end_dy)
        self.curve_residual_1_dx.append(shape.residual_1_dx)
        self.curve_residual_1_dy.append(shape.residual_1_dy)
        self.curve_residual_2_dx.append(shape.residual_2_dx)
        self.curve_residual_2_dy.append(shape.residual_2_dy)


class RawBeatmap(NamedTuple):
    beatmap_id: int
    mode: int
    hp_drain: float
    cs: float
    od: float
    ar: float
    slider_multiplier: float
    slider_tick: float
    difficulty_rating: float
    hit_objects: HitObjects


class TimingPoint(NamedTuple):
    time: int
    beat_length: float
    uninherited: bool


class TimingSection(NamedTuple):
    start_time: int
    beat_length: float
    bpm: float
    sv_multiplier: float
    valid: bool


def _is_float32(value: float) -> bool:
    return math.isfinite(value) and abs(value) <= FLOAT32_MAX


def _is_int32(value: int) -> bool:
    return INT32_MIN <= value <= INT32_MAX


def _timing_sections(points: list[TimingPoint]) -> list[TimingSection]:
    sections = []
    beat = (
        points[0] if points[0].uninherited else TimingPoint(points[0].time, 500.0, True)
    )
    for point in points:
        if point.uninherited:
            beat = point
        bpm = 60000.0 / beat.beat_length if beat.beat_length > 0 else DEFAULT_BPM
        sv_multiplier = 1.0
        if not point.uninherited and point.beat_length < 0:
            sv_multiplier = -100.0 / point.beat_length
        valid = _is_float32(bpm) and _is_int32(beat.time)
        sections.append(
            TimingSection(point.time, beat.beat_length, bpm, sv_multiplier, valid)
        )
    return sections


class Timeline:
    def __init__(self, points: list[TimingPoint], validate: bool) -> None:
        self.sections = _timing_sections(sorted(points, key=lambda point: point.time))
        self.starts = [section.start_time for section in self.sections]
        self.validate = validate
        self.cursor = -1
        self.previous_time = -1

    def advance(self, time: int) -> TimingSection | None:
        if time >= self.previous_time:
            while (
                self.cursor + 1 < len(self.starts)
                and self.starts[self.cursor + 1] <= time
            ):
                self.cursor += 1
        else:
            self.cursor = bisect.bisect_right(self.starts, time) - 1
        self.previous_time = time
        return self._section(self.cursor)

    def find(self, time: int) -> TimingSection | None:
        return self._section(bisect.bisect_right(self.starts, time) - 1)

    def _section(self, index: int) -> TimingSection | None:
        if index < 0:
            return None
        section = self.sections[index]
        if self.validate and not section.valid:
            raise Rejected
        return section


def _second_difference(a: Point, b: Point, c: Point) -> Point:
    return (a[0] - 2 * b[0] + c[0], a[1] - 2 * b[1] + c[1])


def _quadratic(a: float, b: float, c: float, t: float) -> float:
    s = 1 - t
    return s * s * a + 2 * s * t * b + t * t * c


def _cubic(a: float, b: float, c: float, d: float, t: float) -> float:
    s = 1 - t
    return s**3 * a + 3 * s * s * t * b + 3 * s * t * t * c + t**3 * d


def _catmull(
    before: float,
    start: float,
    end: float,
    after: float,
    t: float,
    t2: float,
    t3: float,
) -> float:
    return 0.5 * (
        2 * start
        + (-before + end) * t
        + (2 * before - 5 * start + 4 * end - after) * t2
        + (-before + 3 * start - 3 * end + after) * t3
    )


def _low_order_bezier(points: list[Point]) -> Iterator[Point]:
    curvature = 0.0
    for i in range(1, len(points) - 1):
        dx, dy = _second_difference(points[i - 1], points[i], points[i + 1])
        curvature = max(curvature, math.hypot(dx, dy))
    steps = max(1, math.ceil(math.sqrt(curvature / (8 * BEZIER_TOLERANCE))))
    for step in range(1, steps + 1):
        t = step / steps
        if len(points) == 3:
            a, b, c = points
            yield (_quadratic(a[0], b[0], c[0], t), _quadratic(a[1], b[1], c[1], t))
        else:
            a, b, c, d = points
            yield (
                _cubic(a[0], b[0], c[0], d[0], t),
                _cubic(a[1], b[1], c[1], d[1], t),
            )


def _is_flat(points: list[Point]) -> bool:
    tolerance = BEZIER_TOLERANCE * BEZIER_TOLERANCE * 4
    for i in range(1, len(points) - 1):
        dx, dy = _second_difference(points[i - 1], points[i], points[i + 1])
        if dx**2 + dy**2 > tolerance:
            return False
    return True


def _subdivide(points: list[Point]) -> tuple[list[Point], list[Point]]:
    left = [points[0]]
    right = [points[-1]]
    level = points
    while len(level) > 1:
        level = [
            (
                (level[i][0] + level[i + 1][0]) * 0.5,
                (level[i][1] + level[i + 1][1]) * 0.5,
            )
            for i in range(len(level) - 1)
        ]
        left.append(level[0])
        right.append(level[-1])
    right.reverse()
    return left, right


def _bezier_segment(points: list[Point], depth: int = 0) -> Iterator[Point]:
    if len(points) == 1:
        return
    if len(points) == 2:
        yield points[1]
    elif len(points) <= 4:
        yield from _low_order_bezier(points)
    elif depth >= BEZIER_MAX_DEPTH or _is_flat(points):
        yield points[-1]
    else:
        left, right = _subdivide(points)
        yield from _bezier_segment(left, depth + 1)
        yield from _bezier_segment(right, depth + 1)


def _bezier_vertices(points: list[Point]) -> Iterator[Point]:
    yield points[0]
    segment = [points[0]]
    for point in points[1:]:
        if point != segment[-1]:
            segment.append(point)
            continue
        if len(segment) > 1:
            yield from _bezier_segment(segment)
        segment = [point]
    yield from _bezier_segment(segment)


def _circle_vertices(points: list[Point]) -> Iterator[Point]:
    a, b, c = points
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(cross) < 1e-7:
        yield from _bezier_vertices(points)
        return

    a2 = a[0] * a[0] + a[1] * a[1]
    b2 = b[0] * b[0] + b[1] * b[1]
    c2 = c[0] * c[0] + c[1] * c[1]
    center_x = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / (
        2 * cross
    )
    center_y = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / (
        2 * cross
    )
    radius = math.hypot(a[0] - center_x, a[1] - center_y)
    if not math.isfinite(radius) or radius <= 0:
        yield from _bezier_vertices(points)
        return

    start_angle = math.atan2(a[1] - center_y, a[0] - center_x)
    end_angle = math.atan2(c[1] - center_y, c[0] - center_x)
    if cross > 0:
        sweep = (end_angle - start_angle) % (2 * math.pi)
    else:
        sweep = -((start_angle - end_angle) % (2 * math.pi))
    max_step = 2 * math.acos(max(-1.0, 1 - BEZIER_TOLERANCE / radius))
    segments = max(1, math.ceil(abs(sweep) / max_step)) if max_step > 0 else 1
    for index in range(segments + 1):
        theta = start_angle + sweep * index / segments
        yield (center_x + radius * math.cos(theta), center_y + radius * math.sin(theta))


def _catmull_vertices(points: list[Point]) -> Iterator[Point]:
    last = len(points) - 1
    for i in range(last):
        before = points[max(i - 1, 0)]
        start, end = points[i], points[i + 1]
        if i + 2 <= last:
            after = points[i + 2]
        else:
            after = (2 * end[0] - start[0], 2 * end[1] - start[1])
        for t, t2, t3 in CATMULL_T:
            yield (
                _catmull(before[0], start[0], end[0], after[0], t, t2, t3),
                _catmull(before[1], start[1], end[1], after[1], t, t2, t3),
            )
    yield points[-1]


def _path_vertices(curve_type: str, points: list[Point]) -> Iterable[Point]:
    if curve_type == "L":
        return points
    if curve_type == "B":
        return _bezier_vertices(points)
    if curve_type == "P":
        return (
            _circle_vertices(points) if len(points) == 3 else _bezier_vertices(points)
        )
    if curve_type == "C":
        return _catmull_vertices(points)
    raise ValueError("unknown slider path type")


def _sample_path(
    vertices: Iterable[Point], targets: tuple[float, float, float]
) -> list[Point] | None:
    samples: list[Point] = []
    previous: Point | None = None
    direction: Point | None = None
    travelled = 0.0
    for vertex in vertices:
        if previous is None:
            previous = vertex
            while len(samples) < 3 and targets[len(samples)] == 0:
                samples.append(vertex)
            continue
        dx, dy = vertex[0] - previous[0], vertex[1] - previous[1]
        length = math.hypot(dx, dy)
        if not math.isfinite(length):
            raise ValueError("non-finite slider path")
        if length > 0:
            while len(samples) < 3 and targets[len(samples)] <= travelled + length:
                ratio = (targets[len(samples)] - travelled) / length
                samples.append((previous[0] + dx * ratio, previous[1] + dy * ratio))
            direction = (dx / length, dy / length)
            travelled += length
        previous = vertex
        if len(samples) == 3:
            return samples

    if len(samples) == 3:
        return samples
    if previous is None or direction is None:
        return None
    for target in targets[len(samples) :]:
        samples.append(
            (
                previous[0] + direction[0] * (target - travelled),
                previous[1] + direction[1] * (target - travelled),
            )
        )
    return samples


def _straight_shape(start: Point, end: Point, expected_length: float) -> SliderShape:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return POINT_SHAPE if expected_length == 0 else INVALID_SHAPE
    scale = expected_length / length
    return SliderShape(1, dx * scale, dy * scale, 0.0, 0.0, 0.0, 0.0)


def slider_shape(
    curve_type: str, points: list[Point], expected_length: float
) -> SliderShape:
    if not points or expected_length < 0 or not math.isfinite(expected_length):
        return INVALID_SHAPE
    if curve_type == "L" and len(points) == 2:
        return _straight_shape(points[0], points[1], expected_length)

    targets = (expected_length / 3, expected_length * 2 / 3, expected_length)
    try:
        samples = _sample_path(_path_vertices(curve_type, points), targets)
    except (ArithmeticError, ValueError):
        return INVALID_SHAPE
    if samples is None:
        return INVALID_SHAPE

    start = points[0]
    one, two, end = samples
    end_dx, end_dy = end[0] - start[0], end[1] - start[1]
    return SliderShape(
        1,
        end_dx,
        end_dy,
        one[0] - start[0] - end_dx / 3,
        one[1] - start[1] - end_dy / 3,
        two[0] - start[0] - end_dx * 2 / 3,
        two[1] - start[1] - end_dy * 2 / 3,
    )


class SliderPath(NamedTuple):
    points: list[Point]
    valid: bool
    anchors: int
    hard_anchors: int


def _slider_path(x: int, y: int, curve: list[str], validate: bool) -> SliderPath:
    points: list[Point] = [(float(x), float(y))]
    valid = bool(curve[0])
    anchors = 0
    hard_anchors = 0
    previous: tuple[int, int] | None = None
    previous_was_hard = False
    for token in curve[1:]:
        try:
            coordinates = token.split(":", 2)
            point = (int(coordinates[0]), int(coordinates[1]))
        except (ValueError, IndexError):
            valid = False
            continue
        if validate and max(abs(point[0]), abs(point[1])) > MAX_COORDINATE:
            valid = False
            continue
        points.append((float(point[0]), float(point[1])))
        if point == previous:
            if not previous_was_hard:
                hard_anchors += 1
                previous_was_hard = True
            continue
        previous = point
        previous_was_hard = False
        anchors += 1
    return SliderPath(points, valid, anchors, hard_anchors)


def _object_type(flags: int, mode: int) -> int:
    if flags & 1:
        return OBJECT_TYPE_CIRCLE
    if flags & 2:
        return OBJECT_TYPE_SLIDER
    if flags & 8:
        return OBJECT_TYPE_SPINNER
    if mode == MODE_MANIA and flags & 128:
        return OBJECT_TYPE_HOLD
    return -1


def _section_lines(content: bytes) -> Iterator[tuple[str, str]]:
    section = ""
    text = io.StringIO(content.decode("utf-8", errors="ignore"), newline=None)
    for raw_line in text:
        if section not in ACTIVE_SECTIONS and not raw_line.startswith("["):
            continue
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].lower()
        elif section in ACTIVE_SECTIONS:
            yield section, line


def _value(line: str) -> str:
    return line.split(":", 1)[1].strip()


class BeatmapParser:
    def __init__(
        self,
        beatmap_id: int | None,
        max_hitobject_lines: int | None,
        max_curve_points: int | None,
        validate: bool,
        modes: Collection[int],
    ) -> None:
        self.beatmap_id = beatmap_id
        self.id_from_metadata = beatmap_id is None
        self.max_hitobject_lines = max_hitobject_lines
        self.max_curve_points = max_curve_points
        self.validate = validate
        self.modes = modes
        self.mode = 0
        self.difficulty = dict(DEFAULT_DIFFICULTY)
        self.timing_points: list[TimingPoint] = []
        self.timeline: Timeline | None = None
        self.hit_objects = HitObjects()
        self.hitobject_lines = 0
        self.curve_points = 0

    def feed(self, section: str, line: str) -> None:
        if section == "general":
            self.general(line)
        elif section == "metadata":
            self.metadata(line)
        elif section == "difficulty":
            self.difficulty_setting(line)
        elif section == "timingpoints":
            self.timing_point(line)
        else:
            self.hit_object(line)

    def general(self, line: str) -> None:
        if not line.lower().startswith("mode:"):
            return
        try:
            self.mode = int(_value(line) or 0)
        except ValueError:
            pass
        if self.validate and self.mode not in self.modes:
            raise Rejected

    def metadata(self, line: str) -> None:
        if self.id_from_metadata and line.lower().startswith("beatmapid:"):
            self.beatmap_id = int(line.split(":", 1)[1])

    def difficulty_setting(self, line: str) -> None:
        key, separator, value = line.partition(":")
        field = DIFFICULTY_KEYS.get(key.strip().lower())
        if not separator or field is None:
            return
        try:
            self.difficulty[field] = float(value.strip())
        except ValueError:
            pass

    def timing_point(self, line: str) -> None:
        parts = line.split(",", 8)
        if len(parts) < 2:
            return
        beat_length = float(parts[1])
        if math.isfinite(beat_length) and beat_length != 0:
            uninherited = len(parts) >= 7 and parts[6] == "1"
            self.timing_points.append(
                TimingPoint(int(float(parts[0])), beat_length, uninherited)
            )

    def start_timeline(self) -> Timeline:
        if self.timeline is None:
            if not self.timing_points:
                raise Rejected
            self.timeline = Timeline(self.timing_points, self.validate)
        return self.timeline

    def count_lines(self, curve: list[str] | None) -> None:
        self.hitobject_lines += 1
        limit = self.max_hitobject_lines
        if limit is not None and self.hitobject_lines > limit:
            raise Rejected
        if curve is not None:
            self.curve_points += len(curve) - 1
            limit = self.max_curve_points
            if limit is not None and self.curve_points > limit:
                raise Rejected

    def hit_object(self, line: str) -> None:
        parts = line.split(",", 8)
        try:
            x = int(parts[0])
            y = int(parts[1])
            time = int(parts[2])
            flags = int(parts[3])
            hit_sound = int(parts[4])
        except (ValueError, IndexError):
            self.count_lines(None)
            return
        object_type = _object_type(flags, self.mode)
        is_slider = object_type == OBJECT_TYPE_SLIDER
        curve = parts[5].split("|") if flags & 2 and len(parts) > 5 else None
        self.count_lines(curve)
        if object_type < 0:
            return
        if max(abs(x), abs(y)) > MAX_COORDINATE:
            return
        if self.validate and not _is_int32(hit_sound):
            raise Rejected

        timeline = self.start_timeline()
        section = timeline.advance(time)
        bpm = section.bpm if section is not None else DEFAULT_BPM
        end_time = time
        pixel_length = 0.0
        repeats = 0
        anchors = 0
        hard_anchor_ratio = 0.0
        shape = INVALID_SHAPE
        try:
            if is_slider:
                if curve is None:
                    return
                path = _slider_path(x, y, curve, self.validate)
                slides = int(parts[6])
                pixel_length = float(parts[7])
                if self.validate and not 1 <= slides <= 2**31:
                    raise Rejected
                if path.valid:
                    shape = slider_shape(curve[0], path.points, pixel_length)
                end_time = self.slider_end(time, section, pixel_length, slides)
                repeats = slides - 1
                anchors = path.anchors
                if anchors > 0:
                    hard_anchor_ratio = path.hard_anchors / anchors
            elif object_type == OBJECT_TYPE_SPINNER:
                end_time = int(parts[5])
            elif object_type == OBJECT_TYPE_HOLD:
                end_time = int(parts[5].split(":", 1)[0])
        except (ValueError, IndexError):
            return

        if time > MAX_TIME_MS or end_time > MAX_TIME_MS:
            return
        end_bpm = bpm
        if end_time != time:
            end_section = timeline.find(end_time)
            if end_section is not None:
                end_bpm = end_section.bpm
        if self.validate and not (
            _is_int32(time)
            and _is_int32(end_time)
            and (not is_slider or self.finite_slider(pixel_length, shape))
        ):
            raise Rejected
        self.hit_objects.append(
            HitObject(
                x=x,
                y=y,
                time=time,
                object_type=object_type,
                is_new_combo=1 if flags & 4 else 0,
                hit_sound=hit_sound,
                end_time=end_time,
                pixel_length=pixel_length,
                bpm=bpm,
                end_bpm=end_bpm,
                num_anchors=anchors,
                slider_repeats=repeats,
                hard_anchor_ratio=hard_anchor_ratio,
                shape=shape,
            )
        )

    def slider_end(
        self, time: int, section: TimingSection | None, pixel_length: float, slides: int
    ) -> int:
        if section is None or not math.isfinite(pixel_length):
            return time
        velocity = self.difficulty["slider_multiplier"] * 100.0 * section.sv_multiplier
        if section.beat_length <= 0 or velocity <= 0:
            return time
        slide_ms = (pixel_length / velocity) * section.beat_length
        return time + int(slide_ms * slides)

    @staticmethod
    def finite_slider(pixel_length: float, shape: SliderShape) -> bool:
        return all(_is_float32(value) for value in (pixel_length, *shape[1:]))

    def finish(self) -> RawBeatmap | None:
        beatmap_id = self.beatmap_id
        count = len(self.hit_objects)
        if beatmap_id is None or not self.timing_points or count == 0:
            return None
        if self.validate and not (
            INT64_MIN <= beatmap_id <= INT64_MAX
            and 2 <= count <= (self.max_hitobject_lines or MAX_OBJECTS)
            and all(_is_float32(value) for value in self.difficulty.values())
        ):
            return None
        return RawBeatmap(
            beatmap_id=beatmap_id,
            mode=self.mode,
            hp_drain=self.difficulty["hp_drain"],
            cs=self.difficulty["cs"],
            od=self.difficulty["od"],
            ar=self.difficulty["ar"],
            slider_multiplier=self.difficulty["slider_multiplier"],
            slider_tick=self.difficulty["slider_tick"],
            difficulty_rating=self.difficulty["difficulty_rating"],
            hit_objects=self.hit_objects,
        )


def is_osu_file(content: bytes) -> bool:
    if len(content) < 100:
        return False
    first_line = content[:256].decode("utf-8", errors="ignore").split("\n", 1)[0]
    return first_line.lstrip("\ufeff").strip().startswith("osu file format v")


def parse_beatmap(
    content: bytes,
    beatmap_id: int | None = None,
    *,
    max_hitobject_lines: int | None = None,
    max_curve_points: int | None = None,
    validate_dataset: bool = False,
    modes: Collection[int] = (0,),
) -> RawBeatmap | None:
    parser = BeatmapParser(
        beatmap_id, max_hitobject_lines, max_curve_points, validate_dataset, modes
    )
    try:
        for section, line in _section_lines(content):
            parser.feed(section, line)
    except Rejected:
        return None
    return parser.finish()
