import os
import sqlite3
import threading
import queue
import time
import argparse
from core.data.parser import parse_osu_file

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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS beatmap_vectors (
            id INTEGER PRIMARY KEY,
            beatmap_id INTEGER,
            x_diff REAL, y_diff REAL, time_diff REAL,
            abs_x REAL, abs_y REAL, object_type INTEGER, is_new_combo INTEGER,
            slider_curve_type INTEGER, slider_num_anchors INTEGER,
            slider_pixel_length REAL, slider_complexity REAL,
            spinner_duration_ms REAL,
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
    # If the insert was ignored, beatmap_row_id will be 0. We shouldn't add vectors.
    if beatmap_row_id == 0:
        return

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
        # The worker's job is simple: call the parser and put the result in a queue.
        beatmap_data = parse_osu_file(file_path)
        if beatmap_data and beatmap_data['vectors'] and (len(beatmap_data['vectors']) <= 4000):
            results_queue.put(beatmap_data)
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
                elapsed = time.time() - start_time
                maps_per_sec = count / elapsed if elapsed > 0 else 0
                print(f"Processed {count} beatmaps... ({maps_per_sec:.2f} beatmaps/sec)", end='\r')
        except sqlite3.IntegrityError:
            # This can happen if two workers parse the same mapset; we just ignore it.
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
    
    # Initialize database and tables
    conn = sqlite3.connect(db_path)
    create_tables(conn)
    conn.close()

    tasks_queue = queue.Queue(maxsize=1024)
    results_queue = queue.Queue(maxsize=256)

    # Start the single database writer thread
    db_thread = threading.Thread(target=db_writer, args=(results_queue, db_path))
    db_thread.start()
    
    # Start worker threads
    threads = []
    print(f"Starting {num_worker_threads} worker threads...")
    for _ in range(num_worker_threads):
        t = threading.Thread(target=worker, args=(tasks_queue, results_queue))
        t.daemon = True
        t.start()
        threads.append(t)

    # Find .osu files and add them to the tasks queue
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

    # Signal workers to stop when the queue is empty
    for _ in range(num_worker_threads):
        tasks_queue.put(None)
    tasks_queue.join()
    print("All file processing tasks are complete.")

    # Signal the database writer to stop
    results_queue.put(None)
    results_queue.join()
    
    # Wait for all threads to finish
    for t in threads:
        t.join()
    db_thread.join()

    end_time = time.time()
    print(f"All done! Total time taken: {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()