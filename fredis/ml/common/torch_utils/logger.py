from typing import List, Optional, Dict, Any, Tuple
import os
import pandas as pd

class Metrics:
    def __init__(self, columns: List[str], timestamp_format='%Y:%m:%d_%H:%M:%S'):
        """
        Initialize the Metrics logger.

        Args:
            columns (List[str]): Column names to be used for logging.
            timestamp_format (str): Format for timestamp values (if needed).
        """
        self._columns = columns
        self._buffer = []  # Buffer to store log entries
        self._timestamp_format = timestamp_format

    def __repr__(self):
        return f"Logger(columns={self._columns})"

    def log(self, **kwargs):
        """
        Append a new log entry to the buffer.

        Args:
            kwargs: Column values, must match exactly the set of self.columns.

        Raises:
            ValueError: If missing or extra keys are provided.
        """
        given_keys = set(kwargs.keys())
        expected_keys = set(self._columns)

        missing = expected_keys - given_keys
        extra = given_keys - expected_keys

        if missing:
            raise ValueError(f"Missing keys: {missing}")
        if extra:
            raise ValueError(f"Unexpected keys: {extra}")

        # enforce order according to self.columns
        ordered_row = {col: kwargs[col] for col in self._columns}

        self._buffer.append(ordered_row)

    def flush(self, file_path: str):
        """
        Write the buffer to a TSV file. 
        If the file exists, append without header. Otherwise, create with header.
        After writing, clear the buffer.

        Args:
            file_path (str): Path to the TSV file.
        """
        if not self._buffer:
            return  # nothing to flush

        df = pd.DataFrame(self._buffer, columns=self._columns)

        # check if file exists
        write_header = not os.path.exists(file_path)

        df.to_csv(file_path, sep="\t", mode="a", header=write_header, index=False)

        # clear buffer
        self._buffer = []
        
    