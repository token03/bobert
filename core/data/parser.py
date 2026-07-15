import bisect
import math
import os
from typing import List, NamedTuple, Optional

OBJECT_TYPE_CIRCLE = 0
OBJECT_TYPE_SLIDER = 1
OBJECT_TYPE_SPINNER = 2

MAX_TIME_MS = 36000000

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
    slides: Optional[int]
    pixel_length: Optional[float]
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
    beat_length: float
    bpm: float
    timing_origin: int
    sv_multiplier: float
    kiai: int


def _preprocess_timing_points(
    timing_points: List[RawTimingPoint],
) -> List[TimingSection]:
    sections = []
    last_uninherited = None
    for point in timing_points:
        if point.uninherited:
            last_uninherited = point
        if last_uninherited is None:
            last_uninherited = RawTimingPoint(
                time=point.time,
                beat_length=500.0,
                meter=4,
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
) -> Optional[RawBeatmap]:
    try:
        filename_beatmap_id = int(os.path.splitext(os.path.basename(file_path))[0])
    except ValueError:
        filename_beatmap_id = None

    data = {
        "beatmap_id": filename_beatmap_id,
        "hp_drain": 5.0,
        "cs": 5.0,
        "od": 5.0,
        "ar": 5.0,
        "slider_multiplier": 1.4,
        "slider_tick": 1.0,
        "category": os.path.basename(os.path.dirname(file_path)),
        "difficulty_rating": 0.0,
    }

    timing_points = []
    timing_sections = None
    section_start_times = []
    timing_index = -1
    previous_object_time = -1
    hit_objects = []
    hitobject_line_count = 0
    curve_point_count = 0
    section_name = ""

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            if section_name not in {"metadata", "difficulty", "timingpoints", "hitobjects"}:
                if not raw_line.startswith("["):
                    continue

            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section_name = line[1:-1].lower()
                continue

            if section_name == "metadata":
                if filename_beatmap_id is None and line.lower().startswith("beatmapid:"):
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
                                meter=int(parts[2]) if len(parts) >= 3 else 4,
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

            if is_slider and len(parts) > 5:
                curve_point_count += parts[5].count("|")
                if (
                    max_curve_points is not None
                    and curve_point_count > max_curve_points
                ):
                    return None

            if abs(x) > 100000 or abs(y) > 100000:
                continue

            if timing_sections is None:
                if not timing_points:
                    return None
                timing_points.sort(key=lambda point: point.time)
                timing_sections = _preprocess_timing_points(timing_points)
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

            if object_type == OBJECT_TYPE_SLIDER:
                try:
                    curve_data = parts[5].split("|")
                    curve_type = curve_data[0]
                    previous_point = None
                    previous_was_hard = False
                    for raw_point in curve_data[1:]:
                        coordinates = raw_point.split(":", 2)
                        point = (int(coordinates[0]), int(coordinates[1]))
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
                    if active_section is not None:
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
                    end_section = timing_sections[end_index]
                    end_bpm = end_section.bpm
                    end_timing_origin = end_section.timing_origin

            hard_anchor_ratio = (
                num_hard_anchors / num_anchors if num_anchors > 0 else 0.0
            )
            hit_objects.append(
                RawHitObject(
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
                )
            )

    if data["beatmap_id"] is None or not timing_points or not hit_objects:
        return None
    return RawBeatmap(**data, timing_points=timing_points, hit_objects=hit_objects)
