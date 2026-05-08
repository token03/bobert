import os
import bisect
from typing import Optional, List, NamedTuple, Tuple

OBJECT_TYPE_CIRCLE = 0
OBJECT_TYPE_SLIDER = 1
OBJECT_TYPE_SPINNER = 2

MAX_TIME_MS = 36000000


class RawTimingPoint(NamedTuple):
    time: int
    beat_length: float
    meter: int
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
    bpm: float
    kiai_time: int
    num_anchors: int
    hard_anchor_ratio: float
    slider_end_x: int
    slider_end_y: int


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


class TimingSection(NamedTuple):
    start_time: int
    uninherited: RawTimingPoint
    effective: RawTimingPoint


def _preprocess_timing_points(
    timing_points: List[RawTimingPoint],
) -> List[TimingSection]:
    if not timing_points:
        return []
    sections = []
    last_uninherited = None
    for point in timing_points:
        if point.uninherited:
            last_uninherited = point
        if last_uninherited is None:
            last_uninherited = RawTimingPoint(
                time=point.time, beat_length=500.0, meter=4, uninherited=True, effects=0
            )
        sections.append(
            TimingSection(
                start_time=point.time, uninherited=last_uninherited, effective=point
            )
        )
    return sections


def parse_osu_file(file_path: str) -> Optional[RawBeatmap]:
    data = {
        "beatmap_id": None,
        "hp_drain": 5.0,
        "cs": 5.0,
        "od": 5.0,
        "ar": 5.0,
        "slider_multiplier": 1.4,
        "slider_tick": 1.0,
        "category": "unknown",
        "difficulty_rating": 0.0,
    }

    timing_lines = []
    hitobject_lines = []

    try:
        data["category"] = os.path.basename(os.path.dirname(file_path))

        with open(file_path, "rb") as f:
            content = f.read().decode("utf-8", errors="ignore")

        lines = content.split("\n")
        section = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            if line.startswith("[") and line.endswith("]"):
                section = line[1:-1].lower()
                continue

            if section == "metadata":
                if line.lower().startswith("beatmapid:"):
                    data["beatmap_id"] = int(line.split(":", 1)[1])
            elif section == "difficulty":
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    try:
                        val = float(parts[1].strip())
                        if key == "hpdrainrate":
                            data["hp_drain"] = val
                        elif key == "circlesize":
                            data["cs"] = val
                        elif key == "overalldifficulty":
                            data["od"] = val
                        elif key == "approachrate":
                            data["ar"] = val
                        elif key == "slidermultiplier":
                            data["slider_multiplier"] = val
                        elif key == "slidertickrate":
                            data["slider_tick"] = val
                        elif key == "difficultyrating":
                            data["difficulty_rating"] = val
                    except ValueError:
                        continue
            elif section == "timingpoints":
                timing_lines.append(line)
            elif section == "hitobjects":
                hitobject_lines.append(line)

        if data["beatmap_id"] is None or not hitobject_lines:
            return None

        timing_points = []
        for line in timing_lines:
            parts = line.split(",")
            if len(parts) >= 2:
                beat_length = float(parts[1])
                if beat_length != 0:
                    timing_points.append(
                        RawTimingPoint(
                            time=int(float(parts[0])),
                            beat_length=beat_length,
                            meter=int(parts[2]) if len(parts) >= 3 else 4,
                            uninherited=len(parts) >= 7 and parts[6] == "1",
                            effects=int(parts[7]) if len(parts) >= 8 else 0,
                        )
                    )
        timing_points.sort(key=lambda p: p.time)

        if not timing_points:
            return None

        timing_sections = _preprocess_timing_points(timing_points)
        section_start_times = [s.start_time for s in timing_sections]

        hit_objects = []
        for line in hitobject_lines:
            parts = line.split(",")
            try:
                x = int(parts[0])
                y = int(parts[1])
                time = int(parts[2])
                type_flags = int(parts[3])
                hit_sound = int(parts[4])
            except (ValueError, IndexError):
                continue

            if abs(x) > 100000 or abs(y) > 100000:
                continue

            is_circle = type_flags & 1
            is_slider = type_flags & 2
            is_spinner = type_flags & 8

            obj_type = (
                OBJECT_TYPE_CIRCLE
                if is_circle
                else OBJECT_TYPE_SLIDER
                if is_slider
                else OBJECT_TYPE_SPINNER
                if is_spinner
                else -1
            )
            if obj_type == -1:
                continue

            new_combo = 1 if (type_flags & 4) else 0

            bpm = 120.0
            kiai = 0

            idx = bisect.bisect_right(section_start_times, time) - 1
            if idx >= 0:
                section = timing_sections[idx]
                beat_length = section.uninherited.beat_length

                if beat_length > 0:
                    bpm = 60000.0 / beat_length
                if section.effective.effects & 1:
                    kiai = 1

            curve_type = None
            curve_points = None
            slides = None
            pixel_length = None
            end_time = time
            num_anchors = 0
            hard_anchor_ratio = 0.0
            slider_end_x = 0
            slider_end_y = 0

            if obj_type == OBJECT_TYPE_SLIDER:
                try:
                    curve_data = parts[5].split("|")
                    curve_type = curve_data[0]

                    raw_points = [
                        (int(p.split(":")[0]), int(p.split(":")[1]))
                        for p in curve_data[1:]
                    ]
                    if raw_points:
                        unique_points_with_hardness = []
                        for i, p in enumerate(raw_points):
                            is_hard_anchor_marker = i > 0 and p == raw_points[i - 1]
                            if not is_hard_anchor_marker:
                                unique_points_with_hardness.append([p[0], p[1], 0])
                            elif unique_points_with_hardness:
                                unique_points_with_hardness[-1][2] = 1

                        curve_points = [tuple(p) for p in unique_points_with_hardness]
                        num_anchors = len(curve_points)
                        num_hard_anchors = sum(p[2] for p in curve_points)
                        hard_anchor_ratio = (
                            num_hard_anchors / num_anchors if num_anchors > 0 else 0.0
                        )
                        slider_end_x = curve_points[-1][0]
                        slider_end_y = curve_points[-1][1]

                    slides = int(parts[6])
                    pixel_length = float(parts[7])

                    if idx >= 0:
                        section = timing_sections[idx]
                        beat_length_ms = section.uninherited.beat_length

                        sv_multiplier = 1.0
                        if (
                            not section.effective.uninherited
                            and section.effective.beat_length < 0
                        ):
                            sv_multiplier = -100.0 / section.effective.beat_length

                        if (
                            beat_length_ms > 0
                            and data["slider_multiplier"] > 0
                            and sv_multiplier > 0
                        ):
                            slider_velocity = (
                                data["slider_multiplier"] * 100.0 * sv_multiplier
                            )
                            time_per_slide_ms = (
                                pixel_length / slider_velocity
                            ) * beat_length_ms
                            end_time = time + int(time_per_slide_ms * slides)

                except (ValueError, IndexError):
                    continue
            elif obj_type == OBJECT_TYPE_SPINNER:
                try:
                    end_time = int(parts[5])
                except (ValueError, IndexError):
                    continue

            if time > MAX_TIME_MS or end_time > MAX_TIME_MS:
                continue

            hit_objects.append(
                RawHitObject(
                    x=x,
                    y=y,
                    time=time,
                    object_type=obj_type,
                    is_new_combo=new_combo,
                    curve_type=curve_type,
                    curve_points=curve_points,
                    slides=slides,
                    pixel_length=pixel_length,
                    end_time=end_time,
                    hit_sound=hit_sound,
                    bpm=bpm,
                    kiai_time=kiai,
                    num_anchors=num_anchors,
                    hard_anchor_ratio=hard_anchor_ratio,
                    slider_end_x=slider_end_x,
                    slider_end_y=slider_end_y,
                )
            )

        if not hit_objects:
            return None

        return RawBeatmap(**data, timing_points=timing_points, hit_objects=hit_objects)

    except Exception:
        return None
