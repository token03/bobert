"""
osu! beatmap parser module

This module provides functionality to parse .osu beatmap files and extract
musical features and metadata for analysis.

Constants:
    OBJECT_TYPE_CIRCLE: Hit circle object type (0)
    OBJECT_TYPE_SLIDER: Slider object type (1) 
    OBJECT_TYPE_SPINNER: Spinner object type (2)
    OBJECT_TYPE_UNKNOWN: Unknown object type (-1)
    SLIDER_CURVE_TYPES: Mapping of curve type characters to numeric values
"""

import os
import bisect
from collections import Counter
import math

# Hit object type constants
OBJECT_TYPE_CIRCLE = 0
OBJECT_TYPE_SLIDER = 1
OBJECT_TYPE_SPINNER = 2
OBJECT_TYPE_UNKNOWN = -1

# Slider curve type mapping
SLIDER_CURVE_TYPES = {'B': 0, 'C': 1, 'L': 2, 'P': 3}


def _find_timing_points(t, timing_points, timing_points_times):
    """
    Finds the active uninherited and effective timing points for a given time `t`.
    Internal helper function for the parser.
    """
    idx = bisect.bisect_right(timing_points_times, t) - 1
    if idx < 0:
        return None, None

    effective_point = timing_points[idx]
    uninherited_point = None
    for i in range(idx, -1, -1):
        if timing_points[i]['uninherited']:
            uninherited_point = timing_points[i]
            break
    return uninherited_point, effective_point

def _calculate_main_bpm(timing_points, hit_objects_lines):
    """
    Calculates the most common BPM in a beatmap, weighted by how many
    objects fall under each timing section.
    Internal helper function for the parser.
    """
    if not timing_points or not hit_objects_lines:
        return None

    timing_points_times = [p['time'] for p in timing_points]
    beat_lengths_encountered = []

    for obj_line in hit_objects_lines:
        try:
            t = int(obj_line.split(',')[2])
            uninherited_tp, _ = _find_timing_points(t, timing_points, timing_points_times)
            if uninherited_tp:
                beat_lengths_encountered.append(uninherited_tp['beatLength'])
        except (IndexError, ValueError):
            continue

    if not beat_lengths_encountered:
        for tp in timing_points:
            if tp['uninherited'] and tp['beatLength'] > 0:
                return round(60000.0 / tp['beatLength'])
        return None

    most_common_beat_length = Counter(beat_lengths_encountered).most_common(1)[0][0]

    if most_common_beat_length <= 0:
        return None

    return round(60000.0 / most_common_beat_length)


def parse_osu_file(file_path, print_info=False):
    """
    Parses a .osu file, calculating musically relevant vectors and main BPM.
    
    Returns a dictionary containing beatmap metadata and calculated vectors,
    or None if parsing fails.
    
    - time_diff is in beats and rounded to 5 decimal places.
    - x_diff and y_diff are RAW PIXEL DIFFERENCES.
    - main_bpm is the most common BPM weighted by hit object count.
    - abs_x and abs_y are the absolute pixel positions of the current object.
    - New fields added for hit objects: object_type, is_new_combo, slider properties, etc.
    """
    data = {
        'beatmap_id': None, 'hp_drain': None, 'circle_size': None, 'od': None,
        'ar': None, 'slider_multiplier': 1.4, 'slider_tick': 1.0,
        'hit_objects_lines': [], 'label': None, 'vectors': [], 'main_bpm': None,
        'difficulty_rating': None
    }
    timing_points = []

    try:
        parent_folder = os.path.dirname(file_path)
        data['label'] = os.path.basename(parent_folder)

        with open(file_path, 'r', encoding='utf-8') as file:
            section = None
            for raw in file:
                line = raw.strip()
                if not line or line.startswith('//'): continue
                if line.startswith('[') and line.endswith(']'):
                    section = line[1:-1].lower()
                    continue

                if section == 'metadata':
                    if ':' in line:
                        key, value = line.split(':', 1)
                        if key.strip().lower() == 'beatmapid': data['beatmap_id'] = int(value)
                elif section == 'difficulty':
                    if ':' in line:
                        key, value = map(str.strip, line.split(':', 1))
                        try:
                            val_float = float(value)
                            key_lower = key.lower()

                            if key_lower == 'hpdrainrate': data['hp_drain'] = val_float
                            elif key_lower == 'circlesize': data['circle_size'] = val_float
                            elif key_lower == 'overalldifficulty': data['od'] = val_float
                            elif key_lower == 'approachrate': data['ar'] = val_float
                            elif key_lower == 'slidermultiplier': data['slider_multiplier'] = val_float
                            elif key_lower == 'slidertickrate': data['slider_tick'] = val_float
                            elif key_lower == 'difficultyrating': data['difficulty_rating'] = val_float
                        except ValueError:
                            continue
                elif section == 'timingpoints':
                    parts = line.split(',')
                    if len(parts) >= 2 and float(parts[1]) != 0:
                        timing_points.append({
                            'time': int(float(parts[0])),
                            'beatLength': float(parts[1]),
                            'uninherited': len(parts) >= 7 and parts[6] == '1'
                        })
                elif section == 'hitobjects':
                    data['hit_objects_lines'].append(line)

        timing_points.sort(key=lambda p: p['time'])

        data['main_bpm'] = _calculate_main_bpm(timing_points, data['hit_objects_lines'])

        timing_points_times = [p['time'] for p in timing_points]

        if data['beatmap_id'] is None or not data['hit_objects_lines'] or not timing_points:
            return None

        prev_effective_x, prev_effective_y, prev_time = None, None, None

        for obj_line in data['hit_objects_lines']:
            obj_data = obj_line.split(',')
            if len(obj_data) < 4: continue

            x, y, t = int(obj_data[0]), int(obj_data[1]), int(obj_data[2])
            hit_object_type_flags = int(obj_data[3])
            
            is_circle_flag = hit_object_type_flags & 0b1
            is_slider_flag = hit_object_type_flags & 0b10
            is_spinner_flag = hit_object_type_flags & 0b1000

            if not (is_circle_flag or is_slider_flag or is_spinner_flag):
                continue
            
            current_object_type = OBJECT_TYPE_UNKNOWN
            is_new_combo = 1 if (hit_object_type_flags & 0b0100) else 0

            slider_curve_type_val = -1
            slider_num_anchors = -1
            slider_pixel_length_val = 0.0
            slider_complexity = 0.0
            spinner_duration_ms = 0.0
            
            effective_current_x, effective_current_y = float(x), float(y)

            if is_circle_flag:
                current_object_type = OBJECT_TYPE_CIRCLE
            elif is_slider_flag:
                current_object_type = OBJECT_TYPE_SLIDER
                if len(obj_data) >= 8:
                    try:
                        curve_str = obj_data[5]
                        curve_char = curve_str[0].upper()
                        slider_curve_type_val = SLIDER_CURVE_TYPES.get(curve_char, -1)

                        slides = int(obj_data[6])
                        slider_pixel_length_val = float(obj_data[7])
                        
                        slider_num_anchors = 1 + curve_str.count('|')

                        shape_multiplier = max(1, slider_num_anchors - 1)
                        raw_complexity = slider_pixel_length_val * slides * shape_multiplier
                        slider_complexity = math.log1p(raw_complexity)
                    except (ValueError, IndexError):
                        slider_curve_type_val = -1
                        slider_num_anchors = -1
                        slider_pixel_length_val = 0.0
                        slider_complexity = 0.0
                slider_complexity = min(slider_complexity, 20.0)

            elif is_spinner_flag:
                current_object_type = OBJECT_TYPE_SPINNER
                effective_current_x, effective_current_y = 256.0, 192.0
                if len(obj_data) >= 6:
                    try:
                        end_time = int(obj_data[5])
                        spinner_duration_ms = float(end_time - t)
                    except (ValueError, IndexError):
                        spinner_duration_ms = 0.0

            if prev_time is not None:
                uninherited_tp, _ = _find_timing_points(t, timing_points, timing_points_times)
                if uninherited_tp is None:
                    prev_effective_x, prev_effective_y, prev_time = effective_current_x, effective_current_y, t
                    continue

                time_diff_ms = t - prev_time
                beat_length = uninherited_tp['beatLength']

                time_diff_beats = round(time_diff_ms / beat_length, 5) if beat_length > 10 else 0
                time_diff_beats = min(time_diff_beats, 16.0)

                x_diff = float(effective_current_x - prev_effective_x)
                y_diff = float(effective_current_y - prev_effective_y)

                data['vectors'].append((
                    x_diff, y_diff, time_diff_beats,
                    effective_current_x, effective_current_y,
                    current_object_type, is_new_combo,
                    slider_curve_type_val, slider_num_anchors, slider_pixel_length_val,
                    slider_complexity, spinner_duration_ms
                ))

            prev_effective_x, prev_effective_y, prev_time = effective_current_x, effective_current_y, t

        data.pop('hit_objects_lines')
        return data
    except Exception as e:
        if print_info:
            print(f"Error parsing file {file_path}: {e}")
        return None