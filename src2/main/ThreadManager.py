import threading

thread_name = 0
lock = threading.Lock()


def init_threads():
    global thread_name, lock
    with lock:
        threading.current_thread().name = str(thread_name)
        thread_name += 1


def reset_thread_name():
    global thread_name
    thread_name = 0
