"""
AF Algorithm
============
This module contains the core implementation of the algorithm for quantifying atrial fibrillation organization based on wave-morphology similarity, as proposed in the paper "Faes, Luca et al. “A method for quantifying atrial fibrillation organization based on wave-morphology similarity.”"

Functions
---------
- None: This module does not export any functions.

Classes
-------
- `AFOrganizationAnalyzer`: Main implementation of the AF organization algorithm.
- `AFPerformanceEvaluator`: Evaluate algorithm performance and compare to paper results.

Example
-------
```python
from src.af_algorithm import AFOrganizationAnalyzer, AFPerformanceEvaluator

analyzer = AFOrganizationAnalyzer()
evaluator = AFPerformanceEvaluator(analyzer)
```
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import signal as sgnl
from typing import List, Tuple, Dict, Optional


class AFOrganizationAnalyzer:
    """
    AFOrganizationAnalyzer
    ======================
    Main implementation of the AF organization algorithm for quantifying atrial fibrillation organization based on wave-morphology similarity.

    Methods
    -------
    - `__init__(fs, epsilon, law_duration_ms, blanking_ms)`: Initialize the AF organization analyzer.
    - `bandpass_filter(data, lowcut, highcut, order)`: Apply bandpass filter (40-250 Hz) using Kaiser window FIR filter.
    - `lowpass_filter(data, cutoff, order)`: Apply lowpass filter (20 Hz cutoff) using Kaiser window FIR filter.
    - `detect_atrial_activations(signal)`: Detect atrial activations using adaptive thresholding.
    - `extract_laws(signal, activation_times)`: Extract Local Activation Waves (LAWs) around activation times.
    - `normalize_laws(laws)`: L2 normalize LAWs to unit sphere.
    - `compute_geodesic_distances(normalized_laws)`: Compute geodesic distances on unit sphere between all pairs of LAWs.
    - `compute_regularity_index(distances, epsilon)`: Compute Regularity Index (SI) based on LAW similarity.
    - `analyze_signal(signal, plot)`: Complete analysis pipeline: detect activations, extract LAWs, compute SI.
    - `plot_analysis(signal, activations, laws, SI)`: Plot analysis results.

    Attributes
    ----------
    - `fs (int)`: Sampling frequency in Hz.
    - `epsilon (float)`: Threshold distance in radians for LAW similarity.
    - `law_duration_ms (int)`: Duration of Local Activation Wave window in ms.
    - `blanking_ms (int)`: Blanking period to avoid multiple detections in ms.
    - `law_samples (int)`: Number of samples in LAW window.
    - `blanking_samples (int)`: Number of samples in blanking period.

    Examples
    --------
    ```python
    analyzer = AFOrganizationAnalyzer(fs=250)
    SI, activations, laws = analyzer.analyze_signal(signal)
    ```
    """

    def __init__(
        self,
        fs: int = 250,
        epsilon: float = np.pi / 3,
        law_duration_ms: int = 90,
        blanking_ms: int = 55,
    ):
        """
        Initialize the AF organization analyzer.

        Parameters
        ----------
        - `fs (int)`: Sampling frequency in Hz (defaults to 250 Hz as-per PhysioNet database).
        - `epsilon (float)`: Threshold distance in radians for LAW similarity.
        - `law_duration_ms (int)`: Duration of Local Activation Wave window in ms.
        - `blanking_ms (int)`: Blanking period to avoid multiple detections in ms.

        Returns
        -------
        - `None`: This method does not return a value.

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        analyzer = AFOrganizationAnalyzer(fs=250, epsilon=np.pi/3)
        ```

        Notes
        -----
        - Initializes internal parameters based on the provided values.
        """
        self.fs = fs
        self.epsilon = epsilon
        self.law_duration_ms = law_duration_ms
        self.blanking_ms = blanking_ms
        self.law_samples = int((law_duration_ms / 1000) * fs)
        self.blanking_samples = int((blanking_ms / 1000) * fs)

    def bandpass_filter(
        self,
        data: np.ndarray,
        lowcut: float = 40,
        highcut: float = 250,
        order: int = 40,
    ) -> np.ndarray:
        """
        Apply bandpass filter (40-250 Hz) using Kaiser window FIR filter.

        Parameters
        ----------
        - `data (np.ndarray)`: Input signal data.
        - `lowcut (float)`: Low cutoff frequency in Hz.
        - `highcut (float)`: High cutoff frequency in Hz.
        - `order (int)`: Filter order.

        Returns
        -------
        - `np.ndarray`: Filtered signal data.

        Raises
        ------
        - `ValueError`: If cutoff frequencies are invalid.

        Example
        -------
        ```python
        filtered = analyzer.bandpass_filter(signal)
        ```

        Notes
        -----
        - Uses Kaiser window with beta=8.6 for FIR filter design.
        """
        nyq = 0.5 * self.fs

        lowcut = max(lowcut, 0.1)
        highcut = min(highcut, nyq - 1)

        if lowcut >= highcut:
            raise ValueError(f"Invalid cutoff frequencies: lowcut ({lowcut}) >= highcut ({highcut})")

        low = lowcut / nyq
        high = highcut / nyq
        taps = sgnl.firwin(order + 1, [low, high], pass_zero=False, window=("kaiser", 8.6))

        filtered = sgnl.filtfilt(taps, 1.0, data)
        
        return filtered

    def lowpass_filter(
        self, data: np.ndarray, cutoff: float = 20, order: int = 40
    ) -> np.ndarray:
        """
        Apply lowpass filter (20 Hz cutoff) using Kaiser window FIR filter.

        Parameters
        ----------
        - `data (np.ndarray)`: Input signal data.
        - `cutoff (float)`: Cutoff frequency in Hz.
        - `order (int)`: Filter order.

        Returns
        -------
        - `np.ndarray`: Filtered signal data.

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        envelope = analyzer.lowpass_filter(np.abs(signal))
        ```

        Notes
        -----
        - Uses Kaiser window with beta=8.6 for FIR filter design.
        """
        nyq = 0.5 * self.fs
        cutoff = min(cutoff, nyq - 1)
        
        normalized_cutoff = cutoff / nyq
        taps = sgnl.firwin(order + 1, normalized_cutoff, window=("kaiser", 8.6))
        filtered = sgnl.filtfilt(taps, 1.0, data)
        
        return filtered

    def detect_atrial_activations(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect atrial activations using adaptive thresholding.

        Parameters
        ----------
        - `signal (np.ndarray)`: Input ECG signal.

        Returns
        -------
        - `np.ndarray`: Array of activation indices.

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        activations = analyzer.detect_atrial_activations(signal)
        ```

        Notes
        -----
        - Uses envelope-based adaptive thresholding with blanking period.
        """
        filtered = self.bandpass_filter(signal)
        envelope = self.lowpass_filter(np.abs(filtered))
        threshold = np.mean(envelope) + 2 * np.std(envelope)

        above_threshold = envelope > threshold
        activation_indices = []
        i = 0
        
        while i < len(above_threshold):
            if above_threshold[i]:
                peak_idx = i + np.argmax(envelope[i : i + self.blanking_samples])
                activation_indices.append(peak_idx)
                i += self.blanking_samples
            else:
                i += 1

        return np.array(activation_indices)

    def extract_laws(
        self, signal: np.ndarray, activation_times: np.ndarray
    ) -> List[np.ndarray]:
        """
        Extract Local Activation Waves (LAWs) around activation times.

        Parameters
        ----------
        - `signal (np.ndarray)`: Input signal data.
        - `activation_times (np.ndarray)`: Array of activation indices.

        Returns
        -------
        - `List[np.ndarray]`: List of extracted LAW arrays.

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        laws = analyzer.extract_laws(signal, activations)
        ```

        Notes
        -----
        - Extracts windows of length law_samples around each activation.
        """
        laws = []

        for act_time in activation_times:
            start_idx = act_time - self.law_samples // 2
            end_idx = act_time + self.law_samples // 2

            if start_idx >= 0 and end_idx < len(signal):
                law = signal[start_idx:end_idx]
                if len(law) == self.law_samples:
                    laws.append(law)

        return laws

    def normalize_laws(self, laws: List[np.ndarray]) -> np.ndarray:
        """
        L2 normalize LAWs to unit sphere.

        Parameters
        ----------
        - `laws (List[np.ndarray])`: List of LAW arrays.

        Returns
        -------
        - `np.ndarray`: Array of normalized LAWs.

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        normalized = analyzer.normalize_laws(laws)
        ```

        Notes
        -----
        - Normalizes each LAW to have unit L2 norm.
        """
        if not laws:
            return np.array([])

        laws_array = np.array(laws)
        norms = np.linalg.norm(laws_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = laws_array / norms

        return normalized

    def compute_geodesic_distances(self, normalized_laws: np.ndarray) -> np.ndarray:
        """
        Compute geodesic distances on unit sphere between all pairs of LAWs.

        Parameters
        ----------
        - `normalized_laws (np.ndarray)`: Array of normalized LAWs.

        Returns
        -------
        - `np.ndarray`: Distance matrix.

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        distances = analyzer.compute_geodesic_distances(normalized_laws)
        ```

        Notes
        -----
        - Uses arccos of dot product for geodesic distance.
        """
        if len(normalized_laws) < 2:
            return np.array([])

        similarities = np.dot(normalized_laws, normalized_laws.T)
        similarities = np.clip(similarities, -1, 1)
        distances = np.arccos(similarities)

        return distances

    def compute_regularity_index(
        self, distances: np.ndarray, epsilon: Optional[float] = None
    ) -> float:
        """
        Compute Regularity Index (SI) based on LAW similarity.

        Parameters
        ----------
        - `distances (np.ndarray)`: Distance matrix.
        - `epsilon (Optional[float])`: Threshold distance (uses self.epsilon if None).

        Returns
        -------
        - `float`: Regularity Index (SI).

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        SI = analyzer.compute_regularity_index(distances)
        ```

        Notes
        -----
        - SI is the fraction of LAW pairs with distance <= epsilon.
        """
        if epsilon is None:
            epsilon = self.epsilon

        if len(distances) == 0:
            return 0.0

        n = distances.shape[0]
        if n < 2:
            return 0.0

        mask = np.triu(np.ones_like(distances), k=1).astype(bool)
        valid_distances = distances[mask]

        similar_pairs = np.sum(valid_distances <= epsilon)
        total_pairs = len(valid_distances)

        if total_pairs == 0:
            return 0.0

        SI = similar_pairs / total_pairs

        return SI

    def analyze_signal(
        self, signal: np.ndarray, plot: bool = False
    ) -> Tuple[float, np.ndarray, List[np.ndarray]]:
        """
        Complete analysis pipeline: detect activations, extract LAWs, compute SI.

        Parameters
        ----------
        - `signal (np.ndarray)`: Input ECG signal.
        - `plot (bool)`: Whether to plot the analysis results.

        Returns
        -------
        - `Tuple[float, np.ndarray, List[np.ndarray]]`: SI, activations, laws.

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        SI, activations, laws = analyzer.analyze_signal(signal, plot=True)
        ```

        Notes
        -----
        - Performs the full AF organization analysis.
        """
        activations = self.detect_atrial_activations(signal)
        laws = self.extract_laws(signal, activations)
        normalized_laws = self.normalize_laws(laws)
        distances = self.compute_geodesic_distances(normalized_laws)

        SI = self.compute_regularity_index(distances)

        if plot:
            self.plot_analysis(signal, activations, laws, SI)

        return SI, activations, laws

    def plot_analysis(
        self,
        signal: np.ndarray,
        activations: np.ndarray,
        laws: List[np.ndarray],
        SI: float,
    ):
        """
        Plot analysis results.

        Parameters
        ----------
        - `signal (np.ndarray)`: Input signal.
        - `activations (np.ndarray)`: Activation indices.
        - `laws (List[np.ndarray])`: List of LAWs.
        - `SI (float)`: Regularity Index.

        Returns
        -------
        - `None`: This method does not return a value.

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        analyzer.plot_analysis(signal, activations, laws, SI)
        ```

        Notes
        -----
        - Displays plots of signal, activations, and example LAWs.
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        time = np.arange(len(signal)) / self.fs

        axes[0].plot(time, signal, "b-", linewidth=0.8, alpha=0.7)
        if len(activations) > 0:
            axes[0].plot(
                time[activations],
                signal[activations],
                "r^",
                markersize=8,
                label="Activations",
            )

        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Signal (mV)")
        axes[0].set_title(f"AF Signal Analysis (SI = {SI:.3f})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if laws:
            law_time = np.arange(len(laws[0])) / self.fs * 1000  # Convert to ms
            n_examples = min(5, len(laws))
            for i in range(n_examples):
                axes[1].plot(law_time, laws[i], label=f"LAW {i+1}", alpha=0.7)

        axes[1].set_xlabel("Time (ms)")
        axes[1].set_ylabel("Amplitude")
        axes[1].set_title("Example Local Activation Waves")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class AFPerformanceEvaluator:
    """
    AFPerformanceEvaluator
    ======================
    Evaluate algorithm performance and compare to paper results.

    Methods
    -------
    - `__init__(analyzer)`: Initialize evaluator with an AFOrganizationAnalyzer instance.
    - `evaluate_signal(signal, label, true_type)`: Evaluate a single signal.
    - `classify_af(SI)`: Classify AF type based on SI value using thresholds from paper.
    - `print_summary()`: Print summary of evaluation results.

    Attributes
    ----------
    - `analyzer (AFOrganizationAnalyzer)`: The analyzer instance.
    - `results (list)`: List of evaluation results.

    Examples
    --------
    ```python
    evaluator = AFPerformanceEvaluator(analyzer)
    result = evaluator.evaluate_signal(signal, "record1")
    evaluator.print_summary()
    ```
    """

    def __init__(self, analyzer):
        """
        Initialize evaluator with an AFOrganizationAnalyzer instance.

        Parameters
        ----------
        - `analyzer (AFOrganizationAnalyzer)`: The analyzer instance to use.

        Returns
        -------
        - `None`: This method does not return a value.

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        evaluator = AFPerformanceEvaluator(analyzer)
        ```

        Notes
        -----
        - Initializes the results list.
        """
        self.analyzer = analyzer
        self.results = []

    def evaluate_signal(
        self, signal: np.ndarray, label: str, true_type: Optional[str] = None
    ) -> Dict:
        """
        Evaluate a single signal.

        Parameters
        ----------
        - `signal (np.ndarray)`: Input ECG signal.
        - `label (str)`: Label for the signal.
        - `true_type (Optional[str])`: True AF type if known.

        Returns
        -------
        - `Dict`: Dictionary with evaluation results.

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        result = evaluator.evaluate_signal(signal, "record1", "type1")
        ```

        Notes
        -----
        - Computes SI and classifies AF type.
        """
        SI, activations, laws = self.analyzer.analyze_signal(signal, plot=False)

        predicted_type = self.classify_af(SI)

        result = {
            "label": label,
            "SI": SI,
            "activation_times": activations,
            "n_activations": len(activations),
            "n_laws": len(laws),
            "predicted_type": predicted_type,
            "true_type": true_type,
            "correct": predicted_type == true_type if true_type else None,
        }

        self.results.append(result)
        return result

    def classify_af(self, SI: float) -> str:
        """
        Classify AF type based on SI value using thresholds from paper.

        Parameters
        ----------
        - `SI (float)`: Regularity Index value.

        Returns
        -------
        - `str`: Predicted AF type ('flutter', 'type1', 'type2', 'type3').

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        af_type = evaluator.classify_af(0.5)
        ```

        Notes
        -----
        - Thresholds: flutter >=0.9, type1 >=0.49, type2 >=0.24, else type3.
        """
        if SI >= 0.9:
            return "flutter"
        elif SI >= 0.49:
            return "type1"
        elif SI >= 0.24:
            return "type2"
        else:
            return "type3"

    def print_summary(self):
        """
        Print summary of evaluation results.

        Parameters
        ----------
        - `None`: This method takes no parameters.

        Returns
        -------
        - `None`: This method does not return a value.

        Raises
        ------
        - `None`: No exceptions are raised by this method.

        Example
        -------
        ```python
        evaluator.print_summary()
        ```

        Notes
        -----
        - Prints statistics grouped by predicted AF type.
        """
        if not self.results:
            print("No results to summarize.")
            return

        df = pd.DataFrame(self.results)

        print(f"\nTotal signals analyzed: {len(df)}")
        print(f"Average SI: {df['SI'].mean():.3f} ± {df['SI'].std():.3f}")
        print(f"SI range: [{df['SI'].min():.3f}, {df['SI'].max():.3f}]")

        print("\nResults by predicted AF type:")
        print("-" * 64)
        for af_type in ["flutter", "type1", "type2", "type3"]:
            subset = df[df["predicted_type"] == af_type]
            if len(subset) > 0:
                print(f"\n{af_type.upper()}:")
                print(f"  Count: {len(subset)}")
                print(f"  SI: {subset['SI'].mean():.3f} ± {subset['SI'].std():.3f}")
                print(f"  Avg LAWs: {subset['n_laws'].mean():.1f}")

        print("=" * 64 + "\n")
