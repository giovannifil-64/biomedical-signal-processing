#!/usr/bin/env python3
"""
PhysioNet Data Loader

This module provides functionality to load PhysioNet MIT-BIH AF Database records.
"""

import numpy as np
import os

from typing import Tuple, Dict


class PhysioNetDataLoader:
    """
    Load PhysioNet MIT-BIH AF Database records (already downloaded).
    """

    def __init__(self, data_dir: str = "af_data"):
        """Initialize data loader."""
        self.data_dir = data_dir

    def read_mit_data(
        self, record_id: str, signal_index: int = 0
    ) -> Tuple[np.ndarray, int]:
        """
        Read signal data from MIT-BIH record.
        """
        info = self.read_mit_header(record_id)
        fs = int(info["fs"])

        data_path = os.path.join(self.data_dir, f"{record_id}.dat")

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
            print(f"Error reading data: {e}")
            return np.array([]), fs

    def read_mit_header(self, record_id: str) -> Dict:
        """
        Read MIT-BIH record header file.
        """
        header_path = os.path.join(self.data_dir, f"{record_id}.hea")

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
            print(f"Error reading header: {e}")

        return info
