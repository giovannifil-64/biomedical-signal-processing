"""
AF Data Loader
==============
Supports loading from multiple PhysioNet AF databases: AFDB (Surface ECG, 250 Hz) and IAFDB (Intracardiac recordings, 977 Hz).

Functions
---------
- None: This module does not export any functions.

Classes
-------
- `PhysioNetDataLoader`: Load PhysioNet AF Database records from af_data/ folder structure.

Example
-------
```python
from src.af_data_loader import PhysioNetDataLoader

loader = PhysioNetDataLoader()
signal, fs = loader.read_mit_data("00735", dataset="afdb")
```
"""

import numpy as np
import os

from typing import Tuple, Dict, List, Optional


class PhysioNetDataLoader:
    """
    PhysioNetDataLoader
    ===================
    Load PhysioNet AF Database records from `af_data/` folder structure.

    Methods
    -------
    - `__init__(data_dir)`: Initialize data loader with main af_data folder.
    - `get_available_datasets()`: Get list of available records for each dataset.
    - `read_mit_data(record_id, dataset, signal_index)`: Read signal data from MIT-BIH record.
    - `read_mit_header(record_id, data_dir)`: Read MIT-BIH record header file.

    Attributes
    ----------
    - `data_dir (str)`: Main data directory.
    - `afdb_dir (str)`: AFDB records directory.
    - `iafdb_dir (str)`: IAFDB records directory.

    Examples
    --------
    ```python
    loader = PhysioNetDataLoader()
    datasets = loader.get_available_datasets()
    signal, fs = loader.read_mit_data("00735", "afdb")
    ```
    """

    def __init__(self, data_dir: str = "af_data"):
        """
        Initialize data loader with main af_data folder.

        Parameters
        ----------
        - `data_dir (str)`: Path to the main data directory.

        Returns
        -------
        - `None`: This method does not return a value.

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        loader = PhysioNetDataLoader("af_data")
        ```

        Notes
        -----
        - Sets up paths for AFDB and IAFDB directories.
        """
        self.data_dir = data_dir
        self.afdb_dir = os.path.join(data_dir, "afdb_records")
        self.iafdb_dir = os.path.join(data_dir, "iafdb_records")
    
    def get_available_datasets(self) -> Dict[str, List[str]]:
        """
        Get list of available records for each dataset.

        Parameters
        ----------
        - `None`: This method takes no parameters.

        Returns
        -------
        - `Dict[str, List[str]]`: Dictionary with 'afdb' and 'iafdb' lists of record IDs.

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        datasets = loader.get_available_datasets()
        ```

        Notes
        -----
        - Scans directories for .hea files to find available records.
        """
        datasets = {"afdb": [], "iafdb": []}
        
        if os.path.exists(self.afdb_dir):
            all_files = os.listdir(self.afdb_dir)
            datasets["afdb"] = sorted(set([f.split('.')[0] for f in all_files if f.endswith('.hea')]))
        
        if os.path.exists(self.iafdb_dir):
            all_files = os.listdir(self.iafdb_dir)
            datasets["iafdb"] = sorted(set([f.split('.')[0] for f in all_files if f.endswith('.hea')]))
        
        return datasets

    def read_mit_data(
        self, record_id: str, dataset: str = "afdb", signal_index: int = 0
    ) -> Tuple[np.ndarray, int]:
        """
        Read signal data from MIT-BIH record.

        Parameters
        ----------
        - `record_id (str)`: Record identifier (e.g., '00735', '06453').
        - `dataset (str)`: Dataset to load from ('afdb' or 'iafdb').
        - `signal_index (int)`: Which signal to extract (0 or 1 for multi-channel).

        Returns
        -------
        - `Tuple[np.ndarray, int]`: Signal data in mV and sampling frequency.

        Raises
        ------
        - `ValueError`: If dataset is unknown.
        - `Exception`: If file reading fails.

        Example
        -------
        ```python
        signal, fs = loader.read_mit_data("00735", "afdb", 0)
        ```

        Notes
        -----
        - Converts ADC units to mV using gain of 200.
        """
        # Select appropriate directory
        if dataset.lower() == "afdb":
            data_dir = self.afdb_dir
        elif dataset.lower() == "iafdb":
            data_dir = self.iafdb_dir
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Use 'afdb' or 'iafdb'")
        
        info = self.read_mit_header(record_id, data_dir)
        fs = int(info["fs"])

        data_path = os.path.join(data_dir, f"{record_id}.dat")

        try:
            with open(data_path, "rb") as f:
                data = np.fromfile(f, dtype=np.int16)

            n_samples = len(data) // info["n_signals"]
            data = data[: n_samples * info["n_signals"]].reshape(-1, info["n_signals"])

            signal_data = data[:, signal_index].astype(float)

            # Convert to mV (typical gain is 200 ADC units/mV)
            signal_data_mV = signal_data / 200.0

            return signal_data_mV, fs

        except Exception as e:
            print(f"Error reading {dataset}/{record_id}: {e}")
            return np.array([]), fs

    def read_mit_header(self, record_id: str, data_dir: Optional[str] = None) -> Dict:
        """
        Read MIT-BIH record header file.

        Parameters
        ----------
        - `record_id (str)`: Record identifier.
        - `data_dir (Optional[str])`: Directory containing the record (uses self.data_dir if None).

        Returns
        -------
        - `Dict`: Dictionary with header information (fs, n_signals, etc.).

        Raises
        ------
        - `Exception`: If header file cannot be read.

        Example
        -------
        ```python
        info = loader.read_mit_header("00735")
        ```

        Notes
        -----
        - Parses .hea file for record metadata.
        """
        if data_dir is None:
            data_dir = self.data_dir
        
        header_path = os.path.join(data_dir, f"{record_id}.hea")

        info = {
            "fs": 250,
            "n_signals": 2,
            "length": 0,
            "signal_names": [],
        }

        try:
            with open(header_path, "r") as f:
                lines = f.readlines()

                parts = lines[0].split()
                info["n_signals"] = int(parts[1])
                info["fs"] = float(parts[2])
                info["length"] = int(parts[3])

                for i in range(1, min(len(lines), info["n_signals"] + 1)):
                    parts = lines[i].split()
                    if len(parts) > 8:
                        info["signal_names"].append(parts[8])

        except Exception as e:
            print(f"Error reading header {record_id}: {e}")

        return info
