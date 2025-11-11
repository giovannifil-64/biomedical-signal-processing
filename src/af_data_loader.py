"""
PhysioNet AF Data Loader

Supports loading from multiple PhysioNet AF databases:
- AFDB: Surface ECG (250 Hz)
- IAFDB: Intracardiac recordings (977 Hz)
"""

import numpy as np
import os

from typing import Tuple, Dict, List, Optional


class PhysioNetDataLoader:
    """
    Load PhysioNet AF Database records from af_data/ folder structure.
    
    Supports:
    - af_data/afdb_records/  (AFDB: 250 Hz)
    - af_data/iafdb_records/ (IAFDB: 977 Hz)
    """

    def __init__(self, data_dir: str = "af_data"):
        """Initialize data loader with main af_data folder."""
        self.data_dir = data_dir
        self.afdb_dir = os.path.join(data_dir, "afdb_records")
        self.iafdb_dir = os.path.join(data_dir, "iafdb_records")
    
    def get_available_datasets(self) -> Dict[str, List[str]]:
        """
        Get list of available records for each dataset.
        
        Returns:
            Dictionary with 'afdb' and 'iafdb' lists of record IDs
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
        
        Parameters:
        -----------
        record_id : str
            Record identifier (e.g., '00735', '06453')
        dataset : str
            Dataset to load from: 'afdb' (250 Hz) or 'iafdb' (977 Hz)
        signal_index : int
            Which signal to extract (0 or 1 for multi-channel)
        
        Returns:
            Tuple of (signal_data_mV, sampling_frequency)
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
        
        Parameters:
        -----------
        record_id : str
            Record identifier
        data_dir : str or None
            Directory containing the record. If None, uses self.data_dir
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
