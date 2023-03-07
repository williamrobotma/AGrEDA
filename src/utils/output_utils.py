"""Utils for script output."""

import os
import shutil
import sys
import warnings
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


class TempFolderHolder:
    """Holds temporary directory and real directory for outputs."""

    def __init__(self):
        pass

    def set_output_folder(self, tmp_dir, out_folder):
        """Sets current output folder.

        If tmp_dir exists, return that as output folder, else use out_folder.

        Args:
            tmp_dir (str): Temporary directory for output to use.
            out_folder (str): Real, final directory to be used.

        Returns:
            Output folder path for the course of the script.

        """

        if tmp_dir:
            self.real_out_folder = out_folder
            self.curr_out_folder = tmp_dir

        print(f"Saving results to {self.curr_out_folder} ...")
        if not os.path.isdir(self.curr_out_folder):
            os.makedirs(self.curr_out_folder)
            print(self.curr_out_folder)
        else:
            warnings.warn("Output folder exists. Will overwrite.")

        return self.curr_out_folder

    def copy_out(self):
        """If temp folder is different from final folder, copy out to final."""

        if not os.path.samefile(self.real_out_folder, self.curr_out_folder):
            print(f"Copying from {self.curr_out_folder} to {self.real_out_folder} ...")
            if os.path.exists(self.real_out_folder):
                warnings.warn("Destination exists. Will overwrite.")

            shutil.copytree(
                self.curr_out_folder,
                self.real_out_folder,
                dirs_exist_ok=True,
            )
