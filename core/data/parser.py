# parser.py
import os
import bisect
from collections import Counter
from typing import Optional, List, NamedTuple
from .types import RawBeatmap, RawHitObject, RawTimingPoint

OBJECT_TYPE_CIRCLE = 0
OBJECT_TYPE_SLIDER = 1
OBJECT_TYPE_SPINNER = 2

class TimingSection(NamedTuple):
    start_time: int
    uninherited: RawTimingPoint
    effective: RawTimingPoint

def _preprocess_timing_points(timing_points: List[RawTimingPoint]) -> List[TimingSection]:
    """Pre-processes timing points into sections for O(log n) lookups."""
    if not timing_points:
        return []
        
    sections = []
    last_uninherited = None
    
    for point in timing_points:
        if point.uninherited:
            last_uninherited = point
        
        if last_uninherited is None:
            last_uninherited = RawTimingPoint(time=point.time, beat_length=500.0, uninherited=True, effects=0)

        sections.append(TimingSection(start_time=point.time, uninherited=last_uninherited, effective=point))
    return sections

def _calculate_main_bpm_from_sections(sections: List[TimingSection], hit_objects_times: List[int]) -> Optional[float]:
    """Calculates BPM efficiently using pre-processed timing sections."""
    if not sections or not hit_objects_times:
        return None

    beat_lengths = []
    section_start_times = [s.start_time for s in sections]

    for t in hit_objects_times:
        idx = bisect.bisect_right(section_start_times, t) - 1
        if idx >= 0:
            beat_lengths.append(sections[idx].uninherited.beat_length)

    if not beat_lengths:
        for section in sections:
            if section.uninherited.beat_length > 0:
                return round(60000.0 / section.uninherited.beat_length)
        return None

    most_common_beat_length = Counter(beat_lengths).most_common(1)[0][0]
    if most_common_beat_length <= 0: return None
    
    return round(60000.0 / most_common_beat_length)

def parse_osu_file(file_path: str) -> Optional[RawBeatmap]:
    data = {
        'beatmap_id': None, 'hp_drain': 5.0, 'cs': 5.0, 'od': 5.0,
        'ar': 5.0, 'slider_multiplier': 1.4, 'slider_tick': 1.0,
        'category': 'unknown', 'main_bpm': 120.0, 'difficulty_rating': 0.0
    }
    
    raw_timing_points_lines = []
    raw_hit_objects_lines = []

    try:
        data['category'] = os.path.basename(os.path.dirname(file_path))

        with open(file_path, 'r', encoding='utf-8') as f:
            section = None
            for line in f:
                line = line.strip()
                if not line or line.startswith('//'): continue
                if line.startswith('[') and line.endswith(']'):
                    section = line[1:-1].lower()
                    continue

                if section == 'metadata':
                    if line.lower().startswith('beatmapid:'):
                        data['beatmap_id'] = int(line.split(':')[1])
                elif section == 'difficulty':
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key, value = parts[0].strip().lower(), parts[1].strip()
                        try:
                            val_float = float(value)
                            if key == 'hpdrainrate': data['hp_drain'] = val_float
                            elif key == 'circlesize': data['cs'] = val_float
                            elif key == 'overalldifficulty': data['od'] = val_float
                            elif key == 'approachrate': data['ar'] = val_float
                            elif key == 'slidermultiplier': data['slider_multiplier'] = val_float
                            elif key == 'slidertickrate': data['slider_tick'] = val_float
                        except ValueError: continue
                elif section == 'events':
                     if 'difficultyrating' in line.lower():
                         try:
                             data['difficulty_rating'] = float(line.split(':')[-1])
                         except:
                             pass
                elif section == 'timingpoints':
                    raw_timing_points_lines.append(line)
                elif section == 'hitobjects':
                    raw_hit_objects_lines.append(line)

        if data['beatmap_id'] is None or not raw_hit_objects_lines:
            return None

        timing_points = []
        for line in raw_timing_points_lines:
            parts = line.split(',')
            if len(parts) >= 2 and float(parts[1]) != 0:
                timing_points.append(RawTimingPoint(
                    time=int(float(parts[0])),
                    beat_length=float(parts[1]),
                    uninherited=len(parts) >= 7 and parts[6] == '1',
                    effects=int(parts[7]) if len(parts) >= 8 else 0
                ))
        timing_points.sort(key=lambda p: p.time)
        
        if not timing_points:
            return None

        timing_sections = _preprocess_timing_points(timing_points)
        hit_object_times = [int(line.split(',')[2]) for line in raw_hit_objects_lines]
        data['main_bpm'] = _calculate_main_bpm_from_sections(timing_sections, hit_object_times)

        hit_objects = []
        for line in raw_hit_objects_lines:
            parts = line.split(',')
            try:
                x, y, time, type_flags, hit_sound = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            except (ValueError, IndexError):
                continue
            
            is_circle = type_flags & 1
            is_slider = type_flags & 2
            is_spinner = type_flags & 8
            
            obj_type = OBJECT_TYPE_CIRCLE if is_circle else OBJECT_TYPE_SLIDER if is_slider else OBJECT_TYPE_SPINNER if is_spinner else -1
            if obj_type == -1: continue

            new_combo = 1 if (type_flags & 4) else 0

            curve_type, curve_points, slides, pixel_length = None, None, None, None
            end_time = time

            if obj_type == OBJECT_TYPE_SLIDER:
                try:
                    curve_data = parts[5].split('|')
                    curve_type = curve_data[0]
                    curve_points = [(int(p.split(':')[0]), int(p.split(':')[1])) for p in curve_data[1:]]
                    slides = int(parts[6])
                    pixel_length = float(parts[7])
                except (ValueError, IndexError): continue
            elif obj_type == OBJECT_TYPE_SPINNER:
                try:
                    end_time = int(parts[5])
                except (ValueError, IndexError): continue

            hit_objects.append(RawHitObject(
                x=x, y=y, time=time, object_type=obj_type, is_new_combo=new_combo,
                curve_type=curve_type, curve_points=curve_points, slides=slides, pixel_length=pixel_length,
                end_time=end_time, hit_sound=hit_sound
            ))

        if not hit_objects:
            return None

        return RawBeatmap(**data, timing_points=timing_points, hit_objects=hit_objects)

    except Exception:
        return None