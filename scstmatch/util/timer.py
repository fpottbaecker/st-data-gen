import time


class Timer:
    start_time: float
    message_format: str

    def __init__(self, message_format="{indent}{time:.06f}s\t{message}"):
        self.message_format = message_format
        self.start()

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self, message=""):
        print(self.message_format.format(indent="", time=(time.perf_counter() - self.start_time), message=message))
        self.start()
