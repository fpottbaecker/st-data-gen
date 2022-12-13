import time


class Timer:
    """
    Utility class for time measurements
    """
    start_time: float
    message_format: str

    def __init__(self, message_format="{indent}{time:.06f}s\t{message}"):
        """
        Start a timer
        :param message_format: the format of messages to print (see `str.format`)
        """
        self.message_format = message_format
        self.start()

    def start(self):
        """
        Restart the timer
        """
        self.start_time = time.perf_counter()

    def restart(self, message=""):
        """
        Print the elapsed time and restart the timer
        :param message: message to print for the passed time
        """
        print(self.message_format.format(indent="", time=(time.perf_counter() - self.start_time), message=message))
        self.start()
