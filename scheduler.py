import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from concurrent.futures import ThreadPoolExecutor
import threading
import shutil
import datetime

lock = threading.Lock()
scheduled = 0
completed = 0

class NewFileHandler(FileSystemEventHandler):
    def __init__(self, script_path, folder_path, process_limit):
        super().__init__()
        self.script_path = script_path
        self.process_limit = process_limit
        self.folder_path = folder_path
        self.thread_pool = ThreadPoolExecutor(max_workers=process_limit)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            json_file = event.src_path
            print(f'New JSON file detected: {json_file}')
            self.run_experiment(json_file)

    def run_experiment(self, json_file):
        global scheduled
        print(f"Scheduling experiment: {json_file}")

        queued_folder = os.path.join(self.folder_path, 'queued')
        os.makedirs(queued_folder, exist_ok=True)

        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')        
        new_json_file = os.path.join(queued_folder, f"{timestamp}_{os.path.basename(json_file)}")
        shutil.move(json_file, new_json_file)
        
        scheduled += 1
        command = ['python', self.script_path, '--infile', new_json_file]
        future = self.thread_pool.submit(subprocess.run, command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        future.add_done_callback(self.on_experiment_done)

    def on_experiment_done(self, params):
        global scheduled, completed, lock
        with lock:
            json_file = params.result().args[-1]

            completed_folder = os.path.join(self.folder_path, 'completed')
            os.makedirs(completed_folder, exist_ok=True)

            new_json_file = os.path.join(completed_folder, os.path.basename(json_file))
            shutil.move(json_file, new_json_file)

            completed += 1
            # check the number of remaining tasks
            size = scheduled - completed
            # report the total number of tasks that remain
            print(f'About {size} tasks remain')

def start_experiment_scheduler(script_path, folder_path, process_limit):
    event_handler = NewFileHandler(script_path, folder_path, process_limit)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    print(f'Experiment scheduler started. Monitoring folder: {folder_path}')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

if __name__ == '__main__':
    script_path = 'main.py'
    folder_path = './test'
    process_limit = 3  # Set the desired process limit

    start_experiment_scheduler(script_path, folder_path, process_limit)
