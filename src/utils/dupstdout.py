"""Enables duplicating stdout to a file."""
import sys
from contextlib import contextmanager


class DupStdout:
    """Duplicates stdout to a file."""


    def __init__(self):
        self._stdout = sys.stdout

    def open_file(self, *args, **kwargs):
        """Opens a file and duplicates stdout to it.

        Args:
            *args: Positional arguments to pass to open().
            **kwargs: Keyword arguments to pass to open().
        """
        self.f = open(*args, **kwargs)
        sys.stdout = self

        return self.f

    def close_file(self):
        """Closes the file and restores stdout."""
        self.f.close()
        sys.stdout = self._stdout

    @contextmanager
    def dup_to_file(self, *args, **kwargs):
        """Context manager for duplicating stdout to a file.

        Args:
            *args: Positional arguments to pass to open().
            **kwargs: Keyword arguments to pass to open().
        """
        try:
            yield self.open_file(*args, **kwargs)
        finally:
            self.close_file()



    def write(self, *args, **kwargs):
        self._stdout.write(*args, **kwargs)
        self.f.write(*args, **kwargs)

    def flush(self, *args, **kwargs):
        self._stdout.flush(*args, **kwargs)
        self.f.flush(*args, **kwargs)
