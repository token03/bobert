# parser.py
import os
import bisect
from collections import Counter
import math
import numpy as np
from .types import HitObjectVector, BeatmapMetadata

OBJECT_TYPE_CIRCLE = 0
OBJECT_TYPE_SLIDER = 1
OBJECT_TYPE_SPINNER = 2
OBJECT_TYPE_UNKNOWN = -1

SLIDER_CURVE_TYPES = {'B': 0, 'C': 1, 'L': 2, 'P': 3}

def _convert_to_polar_diff(x_diff, y_diff):
    distance = math.sqrt(x_diff * x_diff + y_diff * y_diff)
    if distance == 0:
        return 0.0, 1.0, 0.0
    cos_angle = x_diff / distance
    sin_angle = y_diff / distance
    return distance, cos_angle, sin_angle

def _find_timing_points(t, timing_points, timing_points_times):
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

        prev_end_x, prev_end_y, prev_end_time = None, None, None

        for obj_line in data['hit_objects_lines']:
            obj_data = obj_line.split(',')
            if len(obj_data) < 4: continue

            try:
                x, y, t, hit_object_type_flags = int(obj_data[0]), int(obj_data[1]), int(obj_data[2]), int(obj_data[3])
            except ValueError:
                continue

            is_circle_flag = hit_object_type_flags & 0b1
            is_slider_flag = hit_object_type_flags & 0b10
            is_spinner_flag = hit_object_type_flags & 0b1000

            if not (is_circle_flag or is_slider_flag or is_spinner_flag):
                continue

            current_start_x, current_start_y = float(x), float(y)
            current_end_x, current_end_y, current_end_time = None, None, None
            current_object_type = OBJECT_TYPE_UNKNOWN
            is_new_combo = 1 if (hit_object_type_flags & 0b0100) else 0

            slider_curve_type = 4  # 4 for N/A
            slider_num_anchors = -1
            slider_pixel_length_val = 0.0
            duration_beats = 0.0

            if is_circle_flag:
                current_object_type = OBJECT_TYPE_CIRCLE
                current_end_x, current_end_y, current_end_time = current_start_x, current_start_y, t
            elif is_slider_flag:
                current_object_type = OBJECT_TYPE_SLIDER
                try:
                    curve_parts = obj_data[5].split('|')
                    curve_char = curve_parts[0][0].upper()
                    slides = int(obj_data[6])
                    slider_pixel_length_val = float(obj_data[7])

                    slider_curve_type = SLIDER_CURVE_TYPES.get(curve_char, 4)
                    slider_num_anchors = len(curve_parts)
                    
                    if slides % 2 == 0:
                        current_end_x, current_end_y = current_start_x, current_start_y
                    else:
                        last_point_str = curve_parts[-1]
                        end_x_str, end_y_str = last_point_str.split(':')
                        current_end_x, current_end_y = float(end_x_str), float(end_y_str)

                    uninherited_tp, effective_tp = _find_timing_points(t, timing_points, timing_points_times)
                    if uninherited_tp:
                        base_beat_length = uninherited_tp['beatLength']
                        slider_velocity = 1.0
                        if effective_tp and effective_tp['beatLength'] < 0:
                            slider_velocity = -100.0 / effective_tp['beatLength']
                        
                        duration_beats = slider_pixel_length_val / (data['slider_multiplier'] * 100.0 * slider_velocity) * slides
                        duration_ms = duration_beats * base_beat_length
                        current_end_time = t + duration_ms
                    else:
                        duration_beats = 0.0
                        current_end_time = t
                except (ValueError, IndexError):
                    if current_end_x is None: current_end_x, current_end_y = current_start_x, current_start_y
                    if current_end_time is None: current_end_time = t
            elif is_spinner_flag:
                current_object_type = OBJECT_TYPE_SPINNER
                current_start_x, current_start_y = 256.0, 192.0
                try:
                    end_time = int(obj_data[5])
                    
                    uninherited_tp, _ = _find_timing_points(t, timing_points, timing_points_times)
                    if uninherited_tp and uninherited_tp['beatLength'] > 0:
                        duration_beats = (end_time - t) / uninherited_tp['beatLength']

                    current_end_x, current_end_y, current_end_time = 256.0, 192.0, end_time
                except (ValueError, IndexError):
                    current_end_x, current_end_y, current_end_time = 256.0, 192.0, t

            if prev_end_time is not None:
                uninherited_tp, _ = _find_timing_points(t, timing_points, timing_points_times)
                if uninherited_tp and uninherited_tp['beatLength'] > 10:
                    time_diff_ms = t - prev_end_time
                    beat_length = uninherited_tp['beatLength']
                    time_diff_beats = round(time_diff_ms / beat_length, 5)
                    x_diff = current_start_x - prev_end_x
                    y_diff = current_start_y - prev_end_y
                    
                    distance_diff, angle_cos, angle_sin = _convert_to_polar_diff(x_diff, y_diff)
                    
                    vector = HitObjectVector.create_with_quantization(
                        distance_diff=distance_diff,
                        angle_cos=angle_cos,
                        angle_sin=angle_sin,
                        time_diff=time_diff_beats,
                        abs_x=current_start_x,
                        abs_y=current_start_y,
                        object_type=current_object_type,
                        is_new_combo=is_new_combo,
                        slider_curve_type=slider_curve_type,
                        slider_num_anchors=slider_num_anchors,
                        slider_pixel_length=slider_pixel_length_val,
                        duration_beats=duration_beats
                    )
                    data['vectors'].append(vector)

            prev_end_x, prev_end_y, prev_end_time = current_end_x, current_end_y, current_end_time

        data.pop('hit_objects_lines')
        return data
    except Exception as e:
        if print_info:
            print(f"Error parsing file {file_path}: {e}")
        return None