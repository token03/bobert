import os
import sqlite3
import threading
import queue
import time
import argparse
from core.data.parser import parse_osu_file
from core.data.types import HitObjectVector

def create_tables(conn):
    """Initializes the database schema."""
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS beatmaps (
            id INTEGER PRIMARY KEY, beatmap_id INTEGER UNIQUE, category TEXT,
            hp_drain REAL, circle_size REAL, od REAL, ar REAL,
            slider_multiplier REAL, slider_tick REAL, main_bpm REAL,
            difficulty_rating REAL
        )
    ''')
    
    vector_fields = ', '.join([f"{field} REAL" for field in HitObjectVector.get_field_names()])
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS beatmap_vectors (
            id INTEGER PRIMARY KEY,
            beatmap_id INTEGER,
            {vector_fields},
            FOREIGN KEY (beatmap_id) REFERENCES beatmaps (id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_beatmap_vectors_beatmap_id ON beatmap_vectors (beatmap_id)')
    conn.commit()

def insert_beatmap_data(cursor, beatmap_data):
    """Inserts a single beatmap data dict into the database."""
    cursor.execute('''
        INSERT OR IGNORE INTO beatmaps (beatmap_id, category, hp_drain, circle_size, od, ar, slider_multiplier, slider_tick, main_bpm, difficulty_rating)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (beatmap_data['beatmap_id'], beatmap_data['label'], beatmap_data['hp_drain'],
         beatmap_data['circle_size'], beatmap_data['od'], beatmap_data['ar'],
         beatmap_data['slider_multiplier'], beatmap_data['slider_tick'], beatmap_data['main_bpm'],
         beatmap_data['difficulty_rating']))

    beatmap_row_id = cursor.lastrowid
    if beatmap_row_id == 0:
        cursor.execute('SELECT id FROM beatmaps WHERE beatmap_id = ?', (beatmap_data['beatmap_id'],))
        result = cursor.fetchone()
        if result:
            beatmap_row_id = result[0]
        else:
            return

    if beatmap_data['vectors']:
        field_names = HitObjectVector.get_field_names()
        placeholders = ', '.join(['?' for _ in field_names])
        field_names_str = ', '.join(field_names)
        
        vector_data_to_insert = []
        for vec in beatmap_data['vectors']:
            vector_tuple = (
                beatmap_row_id,
                float(vec.distance_diff),
                float(vec.angle_cos),
                float(vec.angle_sin),
                float(vec.time_diff),
                float(vec.abs_x),
                float(vec.abs_y),
                int(vec.is_circle),
                int(vec.is_slider),
                int(vec.is_spinner),
                int(vec.is_new_combo),
                int(vec.slider_curve_b),
                int(vec.slider_curve_c),
                int(vec.slider_curve_l),
                int(vec.slider_curve_p),
                float(vec.slider_num_anchors),
                float(vec.slider_pixel_length),
                float(vec.duration_beats)
            )
            vector_data_to_insert.append(vector_tuple)
        
        cursor.executemany(
            f'''INSERT INTO beatmap_vectors (
                beatmap_id, {field_names_str}
            ) VALUES (?, {placeholders})''',
            vector_data_to_insert
        )

def worker(tasks_queue, results_queue):
    """
    Producer: parses files, validates the data, and enqueues valid beatmap data.
    A beatmap is considered invalid and skipped if:
    - It has more than 4000 hit objects.
    - Any hit object has a negative time_diff or duration_beats.
    - Any hit object is outside the playable area (coordinates 0-512 for x, 0-384 for y).
    """
    while True:
        file_path = tasks_queue.get()
        if file_path is None:
            tasks_queue.task_done()
            break
        
        try:
            beatmap_data = parse_osu_file(file_path)
            
            if not beatmap_data or not beatmap_data.get('vectors') or not (0 < len(beatmap_data['vectors']) <= 4000):
                continue

            is_map_valid = True
            for vec in beatmap_data['vectors']:
                # Access fields using NamedTuple attributes
                time_diff = vec.time_diff
                abs_x = vec.abs_x
                abs_y = vec.abs_y
                duration_beats = vec.duration_beats

                if time_diff < 0 or duration_beats < 0:
                    is_map_valid = False
                    break
                
                if not (0 <= abs_x <= 512 and 0 <= abs_y <= 384):
                    is_map_valid = False
                    break
            
            if is_map_valid:
                results_queue.put(beatmap_data)

        except Exception as e:
            pass
        finally:
            tasks_queue.task_done()


def db_writer(results_queue, db_path):
    """Consumer: takes processed data and writes it to the SQLite database."""
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
        try:
            insert_beatmap_data(cursor, beatmap_data)
            count += 1
            if count % 400 == 0:
                conn.commit() 
                elapsed = time.time() - start_time
                maps_per_sec = count / elapsed if elapsed > 0 else 0
                print(f"Processed {count} beatmaps... ({maps_per_sec:.2f} beatmaps/sec)", end='\r')
        except sqlite3.IntegrityError:
            pass
        except Exception as e:
            print(f"DB Writer error: {e}")
        finally:
            results_queue.task_done()
            
    conn.commit()
    conn.close()
    print(f"\nDatabase writer finished. Total beatmaps inserted: {count}")

def main():
    """Main function to orchestrate file discovery, parsing, and database writing."""
    parser = argparse.ArgumentParser(description='Create SQLite database from osu! beatmap files')
    parser.add_argument('directory', nargs='?', default='./data', 
                       help='Directory to search for .osu files (default: ./data)')
    args = parser.parse_args()
    
    root_dir = args.directory
    db_path = './beatmaps.db'
    num_worker_threads = max(1, (os.cpu_count() or 1) - 1)

    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist.")
        return

    start_time = time.time()
    if os.path.exists(db_path):
        print(f"Database at {db_path} already exists. Appending new data.")
    else:
        print("Creating new database.")
    
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

    print(f"Finding .osu files in '{root_dir}'...")
    found_count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        if os.path.basename(dirpath).startswith('.'):
            continue
        for filename in filenames:
            if filename.endswith('.osu'):
                tasks_queue.put(os.path.join(dirpath, filename))
                found_count += 1
    print(f"Found {found_count} beatmaps to process.")

    for _ in range(num_worker_threads):
        tasks_queue.put(None)
    
    tasks_queue.join()
    print("All file processing tasks are complete.")

    results_queue.put(None)
    results_queue.join()
    
    for t in threads:
        t.join()
    db_thread.join()

    end_time = time.time()
    print(f"All done! Total time taken: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()