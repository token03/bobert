import os
import sqlite3
import threading
import queue
import time
import bisect
import math
from collections import Counter


def _find_timing_points(t, timing_points, timing_points_times):
    """
    Finds the active uninherited and effective timing points for a given time `t`.
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

def calculate_main_bpm(timing_points, hit_objects_lines):
    """
    Calculates the most common BPM in a beatmap, weighted by how many
    objects fall under each timing section.
    Rounds BPM to the nearest whole number for semantic consistency,
    as many beatmap properties and human perception often operate on whole or half BPMs.
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

OBJECT_TYPE_CIRCLE = 0
OBJECT_TYPE_SLIDER = 1
OBJECT_TYPE_SPINNER = 2
OBJECT_TYPE_UNKNOWN = -1

SLIDER_CURVE_TYPES = {'B': 0, 'C': 1, 'L': 2, 'P': 3}


def parse_osu_file(file_path, print_info=False):
    """
    Parses a .osu file, calculating musically relevant vectors and main BPM.
    - time_diff is in beats and rounded to 5 decimal places.
    - x_diff and y_diff are now stored as RAW PIXEL DIFFERENCES. Normalization
      is deferred to the model's data loading pipeline for flexibility.
    - main_bpm is the most common BPM weighted by hit object count, rounded to nearest whole number.
    - abs_x and abs_y are the absolute pixel positions of the current object.
    - difficulty_rating is the star rating, usually added by the downloader script.
    - New fields added for hit objects: object_type, is_new_combo, slider_curve_type,
      slider_num_anchors, slider_pixel_length, slider_complexity, spinner_duration_ms.
    """
    data = {
        'beatmap_id': None, 'hp_drain': None, 'circle_size': None, 'od': None,
        'ar': None, 'slider_multiplier': 1.4, 'slider_tick': 1.0,
        'hit_objects': [], 'label': None, 'vectors': [], 'main_bpm': None,
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
                    data['hit_objects'].append(line)

        timing_points.sort(key=lambda p: p['time'])

        data['main_bpm'] = calculate_main_bpm(timing_points, data['hit_objects'])

        timing_points_times = [p['time'] for p in timing_points]

        if data['beatmap_id'] is None or not data['hit_objects'] or not timing_points:
            return None

        prev_effective_x, prev_effective_y, prev_time = None, None, None

        for obj_line in data['hit_objects']:
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

        data.pop('hit_objects')
        return data
    except Exception as e:
        if print_info:
            print(f"Error parsing file {file_path}: {e}")
        return None

def create_tables(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS beatmaps (
            id INTEGER PRIMARY KEY, beatmap_id INTEGER, category TEXT,
            hp_drain REAL, circle_size REAL, od REAL, ar REAL,
            slider_multiplier REAL, slider_tick REAL, main_bpm REAL,
            difficulty_rating REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS beatmap_vectors (
            id INTEGER PRIMARY KEY,
            beatmap_id INTEGER,
            x_diff REAL,
            y_diff REAL,
            time_diff REAL,
            abs_x REAL,
            abs_y REAL,
            object_type INTEGER,
            is_new_combo INTEGER,
            slider_curve_type INTEGER,
            slider_num_anchors INTEGER,
            slider_pixel_length REAL,
            slider_complexity REAL,
            spinner_duration_ms REAL,
            FOREIGN KEY (beatmap_id) REFERENCES beatmaps (id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_beatmap_vectors_beatmap_id ON beatmap_vectors (beatmap_id)')
    conn.commit()

def insert_beatmap_data(cursor, beatmap_data):
    """Inserts a single beatmap data dict into the database using the provided cursor."""
    cursor.execute('''
        INSERT INTO beatmaps (beatmap_id, category, hp_drain, circle_size, od, ar, slider_multiplier, slider_tick, main_bpm, difficulty_rating)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (beatmap_data['beatmap_id'], beatmap_data['label'], beatmap_data['hp_drain'],
         beatmap_data['circle_size'], beatmap_data['od'], beatmap_data['ar'],
         beatmap_data['slider_multiplier'], beatmap_data['slider_tick'], beatmap_data['main_bpm'],
         beatmap_data['difficulty_rating']))

    beatmap_row_id = cursor.lastrowid

    if beatmap_data['vectors']:
        vector_data_to_insert = [
            (beatmap_row_id, vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8], vec[9], vec[10], vec[11])
            for vec in beatmap_data['vectors']
        ]
        cursor.executemany(
            '''INSERT INTO beatmap_vectors (
                beatmap_id, x_diff, y_diff, time_diff, abs_x, abs_y,
                object_type, is_new_combo, slider_curve_type, slider_num_anchors,
                slider_pixel_length, slider_complexity, spinner_duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            vector_data_to_insert
        )

def worker(tasks_queue, results_queue):
    """Producer: parses files and enqueues the processed beatmap data."""
    while True:
        file_path = tasks_queue.get()
        if file_path is None:
            tasks_queue.task_done()
            break
        beatmap_data = parse_osu_file(file_path)
        if beatmap_data and beatmap_data['vectors'] and (len(beatmap_data['vectors']) <= 4000):
            results_queue.put(beatmap_data)
        tasks_queue.task_done()

def db_writer(results_queue, db_path):
    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA journal_mode=WAL;')
    cursor = conn.cursor()
    count = 0
    start_time = time.time()
    while True:
        beatmap_data = results_queue.get()
        if beatmap_data is None:
            results_queue.task_done()
            break
        insert_beatmap_data(cursor, beatmap_data)
        count += 1
        if count % 400 == 0:
            elapsed = time.time() - start_time
            maps_per_sec = count / elapsed if elapsed > 0 else 0
            print(f"Processed {count} beatmaps... ({maps_per_sec:.2f} beatmaps/sec)", end='\r')
        results_queue.task_done()
    conn.commit()
    conn.close()
    print(f"\nDatabase writer finished. Total beatmaps inserted: {count}")

def main():
    root_dir = '.'
    db_path = './beatmaps.db'
    num_worker_threads = max(1, (os.cpu_count() or 1) - 1)

    start_time = time.time()
    if os.path.exists(db_path):
        print(f"Database at {db_path} already exists. Deleting to start fresh.")
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    create_tables(conn)
    conn.close()

    tasks_queue = queue.Queue(maxsize=1024)
    results_queue = queue.Queue(maxsize=256)

    db_thread = threading.Thread(target=db_writer, args=(results_queue, db_path))
    db_thread.start()
    threads = []

    print(f"Starting {num_worker_threads} worker threads...")
    for _ in range(num_worker_threads):
        t = threading.Thread(target=worker, args=(tasks_queue, results_queue))
        t.daemon = True
        t.start()
        threads.append(t)

    print("Finding .osu files...")
    found_count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        if os.path.basename(dirpath).startswith('.'):
            continue
        for filename in filenames:
            if filename.endswith('.osu'):
                tasks_queue.put(os.path.join(dirpath, filename))
                found_count += 1
    print(f"Found {found_count} beatmaps to process.")

    for _ in range(num_worker_threads): tasks_queue.put(None)
    tasks_queue.join()
    print("All file processing tasks are complete.")

    results_queue.put(None)
    results_queue.join()
    
    for t in threads: t.join()
    db_thread.join()

    end_time = time.time()
    print(f"All done! Total time taken: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()