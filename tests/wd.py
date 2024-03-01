import time
import os
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

def on_created(event):
    # Wait for the file to be completely written
    file = None
    while file is None:
        try:
            file = open(event.src_path, 'rb')
        except OSError:
            file = None
            time.sleep(1)
            continue
    file.close() # This will also update the file's access and modification times,
                 # hence the on_modified event will be triggered
    print(f"{event.src_path} has been created!")

def on_deleted(event):
    print(f"{event.src_path} has been deleted")

def on_modified(event):
    print(f"{event.src_path} has been modified")

def on_moved(event):
    print(f"{event.src_path} has been moved to {event.dest_path}")

if __name__ == "__main__":
    patterns = ["*.jpg", "*.png", "*.jpeg"]
    ignore_patterns = None
    ignore_directories = True
    case_sensitive = False
    my_event_handler = PatternMatchingEventHandler(
        patterns, ignore_patterns, ignore_directories, case_sensitive)
    my_event_handler.on_created = on_created
    my_event_handler.on_deleted = on_deleted
    my_event_handler.on_modified = on_modified
    my_event_handler.on_moved = on_moved
    path = os.path.join(os.getcwd(), "dataset")
    go_recursively = False
    my_observer = Observer()
    my_observer.schedule(my_event_handler, path, recursive=go_recursively)
    my_observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        my_observer.stop()
        my_observer.join()
