# create_db.py
import os
import sqlite3
import threading
import queue
import time
import argparse
import random 
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data.parser import parse_osu_file
from core.data.types import HitObjectVector, BeatmapData

def create_tables(conn):
    cursor = conn.cursor()
    
    beatmap_fields = BeatmapData.get_db_field_types()
    beatmap_fields_sql = ', '.join([f"{field} {field_type}" for field, field_type in beatmap_fields.items()])
    
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS beatmaps (
            id INTEGER PRIMARY KEY,
            {beatmap_fields_sql}
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
    field_names = BeatmapData.get_field_names()
    placeholders = ', '.join(['?' for _ in field_names])
    field_names_str = ', '.join(field_names)
    
    cursor.execute(f'''
        INSERT OR IGNORE INTO beatmaps ({field_names_str})
        VALUES ({placeholders})
        ''',
        beatmap_data.to_db_tuple())

    beatmap_row_id = cursor.lastrowid
    if beatmap_row_id == 0:
        cursor.execute('SELECT id FROM beatmaps WHERE beatmap_id = ?', (beatmap_data.beatmap_id,))
        result = cursor.fetchone()
        if result:
            beatmap_row_id = result[0]
        else:
            return

    if beatmap_data.vectors:
        vector_field_names = HitObjectVector.get_field_names()
        vector_placeholders = ', '.join(['?' for _ in vector_field_names])
        vector_field_names_str = ', '.join(vector_field_names)
        
        vector_data_to_insert = []
        for vec in beatmap_data.vectors:
            vec_array = vec.to_array()
            vector_tuple = (beatmap_row_id,) + tuple(float(x) for x in vec_array)
            vector_data_to_insert.append(vector_tuple)
        
        cursor.executemany(
            f'''INSERT INTO beatmap_vectors (
                beatmap_id, {vector_field_names_str}
            ) VALUES (?, {vector_placeholders})''',
            vector_data_to_insert
        )

def worker(tasks_queue, results_queue):
    while True:
        file_path = tasks_queue.get()
        if file_path is None:
            tasks_queue.task_done()
            break
        
        try:
            beatmap_data = parse_osu_file(file_path)
            
            if not beatmap_data or not beatmap_data.vectors or not (0 < len(beatmap_data.vectors) <= 4000):
                continue

            results_queue.put(beatmap_data)

        except Exception as e:
            pass
        finally:
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


def create_full_db(root_dir, output_dir='./data'):
    db_path = os.path.join(output_dir, 'beatmaps.db')
    os.makedirs(output_dir, exist_ok=True)
    num_worker_threads = max(1, (os.cpu_count() or 1) - 1)
    start_time = time.time()
    
    if os.path.exists(db_path):
        print(f"Database at {db_path} already exists. Appending new data.")
    else:
        print(f"Creating new database at {db_path}.")
    
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


def create_test_db(root_dir, sample_size=3000, output_dir='./data'):
    db_path = os.path.join(output_dir, 'beatmaps_test.db')
    os.makedirs(output_dir, exist_ok=True)
    num_worker_threads = max(1, (os.cpu_count() or 1) - 1)
    start_time = time.time()

    print(f"Finding all .osu files in '{root_dir}' to create a sample...")
    all_osu_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if os.path.basename(dirpath).startswith('.'):
            continue
        for filename in filenames:
            if filename.endswith('.osu'):
                all_osu_files.append(os.path.join(dirpath, filename))
    
    print(f"Found {len(all_osu_files)} total .osu files.")

    if len(all_osu_files) > sample_size:
        print(f"Randomly sampling {sample_size} beatmaps for the test database...")
        files_to_process = random.sample(all_osu_files, sample_size)
    else:
        print(f"Found fewer files than sample size. Processing all {len(all_osu_files)} files.")
        files_to_process = all_osu_files

    if os.path.exists(db_path):
        print(f"Test database at {db_path} already exists. Overwriting.")
        os.remove(db_path)
    else:
        print(f"Creating new test database at {db_path}.")
    
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

    print(f"Enqueuing {len(files_to_process)} beatmaps to be processed...")
    for file_path in files_to_process:
        tasks_queue.put(file_path)

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
    print(f"Test DB creation done! Total time taken: {end_time - start_time:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description='Create SQLite database from osu! beatmap files')
    parser.add_argument('directory', nargs='?', default='./data', 
                       help='Directory to search for .osu files (default: ./data)')
    parser.add_argument('--test', action='store_true', default=False,
                       help='Create a smaller, randomly sampled test database (default: False).')
    parser.add_argument('--sample_size', type=int, default=3000,
                       help='Number of beatmaps for the test database (default: 3000).')
    parser.add_argument('--output-dir', type=str, default='./data',
                       help='Directory to create the database files (default: ./data).')
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist.")
        return

    if args.test:
        print(f"Creating TEST database with {args.sample_size} samples...")
        create_test_db(args.directory, args.sample_size, args.output_dir)
    else:
        print("Creating FULL database with all available beatmaps...")
        create_full_db(args.directory, args.output_dir)


if __name__ == '__main__':
    main()