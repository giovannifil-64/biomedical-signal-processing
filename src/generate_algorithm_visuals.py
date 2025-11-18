"""
Generate Algorithm Visuals
==========================
This script creates visualizations for each step of the AF organization algorithm using IAFDB intracardiac data from the PhysioNet repository.

Functions
---------
- None: This module does not export any functions.

Classes
-------
- None: This module does not export any classes.

Example
-------
```python
python src/generate_algorithm_visuals.py
```

Notes
-----
- Output files are always created in the project root's presentation_visuals/ folder, regardless of where the script is run from.
- Output files:
  - presentation_visuals/step1_bandpass_filter.png
  - presentation_visuals/step2_activation_detection.png
  - presentation_visuals/step3_laws_extraction.png
  - presentation_visuals/step4_normalization_sphere.png
  - presentation_visuals/step5_distance_matrix.png
  - presentation_visuals/step6_si_interpretation.png
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import warnings

from matplotlib.patches import FancyBboxPatch

from src.af_algorithm import AFOrganizationAnalyzer
from src.af_data_loader import PhysioNetDataLoader


warnings.filterwarnings('ignore')

# Determine project root and output directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # src/ is in project root
output_dir = os.path.join(project_root, 'presentation_visuals')

os.makedirs(output_dir, exist_ok=True)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


print("Loading real intracardiac AF data from IAFDB...")
loader = PhysioNetDataLoader(data_dir="af_data")
signal, fs = loader.read_mit_data(record_id="iaf1_ivc", dataset="iafdb", signal_index=0)
analyzer = AFOrganizationAnalyzer(fs=fs)

# Use first 10 seconds for visualizations (IAFDB is 977 Hz, so ~9770 samples)
duration_samples = min(10 * fs, len(signal))
signal = signal[:duration_samples]

print(f"Loaded: {len(signal)} samples at {fs} Hz")


# ============================================================================
# VISUAL 1: BANDPASS FILTER (Before/After)
# ============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Raw signal
time = np.arange(len(signal)) / fs
axes[0].plot(time, signal, 'b-', linewidth=0.8, alpha=0.7)
axes[0].set_ylabel('Amplitude (mV)', fontsize=11)
axes[0].set_title('Step 1A: Raw ECG Signal', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 5])

# Plot 2: Filtered signal
filtered = analyzer.bandpass_filter(signal)
axes[1].plot(time, filtered, 'g-', linewidth=0.8, alpha=0.7)
axes[1].set_ylabel('Amplitude (mV)', fontsize=11)
axes[1].set_title('Step 1B: Bandpass Filtered (40-250 Hz)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 5])

# Plot 3: Comparison (zoomed)
zoom_start, zoom_end = 1.0, 2.0
zoom_idx = (time >= zoom_start) & (time <= zoom_end)
axes[2].plot(time[zoom_idx], signal[zoom_idx], 'b-', linewidth=1.2, alpha=0.6, label='Raw')
axes[2].plot(time[zoom_idx], filtered[zoom_idx], 'g-', linewidth=1.5, label='Filtered')
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].set_ylabel('Amplitude (mV)', fontsize=11)
axes[2].set_title('Step 1C: Comparison (Zoomed 1-2s)', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'step1_bandpass_filter.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Saved: step1_bandpass_filter.png")


# ============================================================================
# VISUAL 2: ACTIVATION DETECTION
# ============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Filter and get envelope
filtered = analyzer.bandpass_filter(signal)
envelope = analyzer.lowpass_filter(np.abs(filtered))
threshold = np.mean(envelope) + 2 * np.std(envelope)
activations = analyzer.detect_atrial_activations(signal)

# Plot 1: Filtered signal
axes[0].plot(time, filtered, 'b-', linewidth=0.8, alpha=0.7)
axes[0].set_ylabel('Amplitude (mV)', fontsize=11)
axes[0].set_title('Step 2A: Filtered Signal', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 5])

# Plot 2: Envelope with threshold
axes[1].plot(time, envelope, 'purple', linewidth=1.5, label='Envelope')
axes[1].axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                label=f'Threshold (μ+2σ = {threshold:.3f})')
axes[1].fill_between(time, 0, envelope, where=(envelope > threshold), 
                      alpha=0.3, color='red', label='Above threshold')
axes[1].set_ylabel('Envelope Amplitude', fontsize=11)
axes[1].set_title('Step 2B: Envelope + Adaptive Threshold', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 5])

# Plot 3: Detected activations
axes[2].plot(time, filtered, 'b-', linewidth=0.8, alpha=0.5, label='Filtered Signal')
if len(activations) > 0:
    act_times = activations / fs
    axes[2].plot(act_times, filtered[activations], 'r^', markersize=12, 
                 label=f'Activations Detected (n={len(activations)})')
axes[2].set_xlabel('Time (s)', fontsize=11)
axes[2].set_ylabel('Amplitude (mV)', fontsize=11)
axes[2].set_title('Step 2C: Detected Atrial Activations', fontsize=14, fontweight='bold')
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([0, 5])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'step2_activation_detection.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Saved: step2_activation_detection.png")


# ============================================================================
# VISUAL 3: LAWs EXTRACTION
# ============================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Extract LAWs
activations = analyzer.detect_atrial_activations(signal)
laws = analyzer.extract_laws(filtered, activations)

# Plot 1: Signal with LAW windows
axes[0].plot(time, filtered, 'b-', linewidth=0.8, alpha=0.5)
if len(activations) > 0:
    act_times = activations / fs
    axes[0].plot(act_times, filtered[activations], 'r^', markersize=10, 
                 label='Activation Centers')
    
    # Draw LAW windows (first 10 for clarity)
    law_duration = 0.09  # 90ms
    for i, act_idx in enumerate(activations[:10]):
        act_time = act_idx / fs
        rect = FancyBboxPatch((act_time - law_duration/2, axes[0].get_ylim()[0]), 
                               law_duration, 
                               axes[0].get_ylim()[1] - axes[0].get_ylim()[0],
                               boxstyle="round,pad=0.01", 
                               edgecolor='orange', facecolor='yellow', 
                               alpha=0.2, linewidth=2)
        axes[0].add_patch(rect)

axes[0].set_ylabel('Amplitude (mV)', fontsize=11)
axes[0].set_title('Step 3A: LAW Windows (90ms) on Signal', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 3])

# Plot 2: Overlapped LAWs
if len(laws) > 0:
    law_time = np.arange(len(laws[0])) / fs * 1000  # Convert to ms
    n_examples = min(10, len(laws))
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_examples))
    
    for i in range(n_examples):
        axes[1].plot(law_time, laws[i], linewidth=1.5, alpha=0.7, 
                     color=colors[i], label=f'LAW {i+1}')
    
    axes[1].axvline(x=45, color='r', linestyle='--', linewidth=2, 
                    label='Activation Center (45ms)')
    axes[1].set_xlabel('Time relative to activation (ms)', fontsize=11)
    axes[1].set_ylabel('Amplitude (mV)', fontsize=11)
    axes[1].set_title(f'Step 3B: Extracted LAWs (n={len(laws)}) - Overlapped View', 
                      fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=9, ncol=2)
    axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'step3_laws_extraction.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Saved: step3_laws_extraction.png")


# ============================================================================
# VISUAL 4: NORMALIZATION (Unit Sphere)
# ============================================================================
fig = plt.figure(figsize=(16, 7))
law_time = ""

# Left: Before normalization (2D representation)
ax1 = fig.add_subplot(121)
if len(laws) >= 3:
    # Show 3 LAWs with different amplitudes
    law_time = np.arange(len(laws[0])) / fs * 1000
    ax1.plot(law_time, laws[0], 'b-', linewidth=2, label=f'LAW 1 (||·|| = {np.linalg.norm(laws[0]):.2f})')
    ax1.plot(law_time, laws[1], 'g-', linewidth=2, label=f'LAW 2 (||·|| = {np.linalg.norm(laws[1]):.2f})')
    ax1.plot(law_time, laws[2], 'r-', linewidth=2, label=f'LAW 3 (||·|| = {np.linalg.norm(laws[2]):.2f})')
    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_ylabel('Amplitude (mV)', fontsize=11)
    ax1.set_title('Step 4A: Original LAWs (Different Amplitudes)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

# Right: After normalization
ax2 = fig.add_subplot(122)
if len(laws) >= 3:
    normalized_laws = analyzer.normalize_laws(laws)
    ax2.plot(law_time, normalized_laws[0], 'b-', linewidth=2, 
             label=f'LAW 1 norm (||·|| = {np.linalg.norm(normalized_laws[0]):.2f})')
    ax2.plot(law_time, normalized_laws[1], 'g-', linewidth=2, 
             label=f'LAW 2 norm (||·|| = {np.linalg.norm(normalized_laws[1]):.2f})')
    ax2.plot(law_time, normalized_laws[2], 'r-', linewidth=2, 
             label=f'LAW 3 norm (||·|| = {np.linalg.norm(normalized_laws[2]):.2f})')
    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_ylabel('Normalized Amplitude', fontsize=11)
    ax2.set_title('Step 4B: Normalized LAWs (||·|| = 1.0 for all)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'step4_normalization_sphere.png'), dpi=300, bbox_inches='tight')
plt.close()

print("Saved: step4_normalization_sphere.png")


# ============================================================================
# VISUAL 5: DISTANCE MATRIX (Heatmap)
# ============================================================================
if len(laws) >= 5:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Normalize and compute distances
    normalized_laws = analyzer.normalize_laws(laws)
    distances = analyzer.compute_geodesic_distances(normalized_laws)
    
    # Use subset for clarity (first 15 LAWs)
    n_show = min(15, len(laws))
    distances_subset = distances[:n_show, :n_show]
    
    # Plot 1: Full heatmap
    im1 = axes[0].imshow(distances_subset, cmap='RdYlGn_r', vmin=0, vmax=np.pi, aspect='auto')
    axes[0].set_xlabel('LAW Index', fontsize=11)
    axes[0].set_ylabel('LAW Index', fontsize=11)
    axes[0].set_title('Step 5A: Geodesic Distance Matrix', fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Distance (radians)', fontsize=10)
    
    # Add epsilon line
    axes[0].axhline(y=0, color='blue', linestyle='--', linewidth=2, alpha=0.5)
    axes[0].text(n_show-1, 0, f'ε = π/3 = {np.pi/3:.2f}', 
                 ha='right', va='bottom', fontsize=10, color='blue', fontweight='bold')
    
    # Plot 2: Binary similarity matrix (d ≤ ε)
    epsilon = np.pi / 3
    similarity_matrix = (distances_subset <= epsilon).astype(int)
    im2 = axes[1].imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[1].set_xlabel('LAW Index', fontsize=11)
    axes[1].set_ylabel('LAW Index', fontsize=11)
    axes[1].set_title(f'Step 5B: Similarity Matrix (d ≤ ε = π/3)', fontsize=14, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=axes[1], ticks=[0, 1])
    cbar2.set_label('Similar (1) / Dissimilar (0)', fontsize=10)
    
    # Calculate SI for this subset
    n = similarity_matrix.shape[0]
    total_pairs = n * (n - 1) // 2
    similar_pairs = np.sum(np.triu(similarity_matrix, k=1))
    si_subset = similar_pairs / total_pairs if total_pairs > 0 else 0
    
    axes[1].text(n_show/2, -1, f'SI = {similar_pairs}/{total_pairs} = {si_subset:.3f}', 
                 ha='center', fontsize=12, fontweight='bold', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'step5_distance_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved: step5_distance_matrix.png")


# ============================================================================
# VISUAL 6: SI INTERPRETATION (High vs Low Organization)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Generate synthetic examples
np.random.seed(42)

# High organization (Flutter-like): SI ~ 0.95
n_laws_high = 10
base_wave = np.sin(2*np.pi*np.linspace(0, 2, 50))
laws_high = np.array([base_wave + 0.05*np.random.randn(50) for _ in range(n_laws_high)])
laws_high_norm = laws_high / np.linalg.norm(laws_high, axis=1, keepdims=True)
dist_high = np.arccos(np.clip(np.dot(laws_high_norm, laws_high_norm.T), -1, 1))
si_high = np.sum(dist_high <= np.pi/3) / (n_laws_high * (n_laws_high - 1))

# Low organization (Type III): SI ~ 0.15
laws_low = np.array([np.random.randn(50) for _ in range(n_laws_high)])
laws_low_norm = laws_low / np.linalg.norm(laws_low, axis=1, keepdims=True)
dist_low = np.arccos(np.clip(np.dot(laws_low_norm, laws_low_norm.T), -1, 1))
si_low = np.sum(dist_low <= np.pi/3) / (n_laws_high * (n_laws_high - 1))

# Plot HIGH organization
cmap = plt.cm.get_cmap('tab10')
# Overlapped LAWs
for i, law in enumerate(laws_high):
    axes[0, 0].plot(law, alpha=0.7, linewidth=1.5, color=cmap(i))
axes[0, 0].set_title('High Organization: Overlapped LAWs\n(Very Similar Shapes)', 
                      fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Sample', fontsize=10)
axes[0, 0].set_ylabel('Amplitude', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Distance matrix
im1 = axes[0, 1].imshow(dist_high, cmap='RdYlGn_r', vmin=0, vmax=np.pi)
axes[0, 1].set_title(f'High Organization: Distance Matrix\nSI = {si_high:.3f} (Flutter)', 
                      fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('LAW Index', fontsize=10)
axes[0, 1].set_ylabel('LAW Index', fontsize=10)
plt.colorbar(im1, ax=axes[0, 1], label='Distance (rad)')

# Plot LOW organization
# Overlapped LAWs
for i, law in enumerate(laws_low):
    axes[1, 0].plot(law, alpha=0.7, linewidth=1.5, color=cmap(i))
axes[1, 0].set_title('Low Organization: Overlapped LAWs\n(Very Different Shapes)', 
                      fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Sample', fontsize=10)
axes[1, 0].set_ylabel('Amplitude', fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# Distance matrix
im2 = axes[1, 1].imshow(dist_low, cmap='RdYlGn_r', vmin=0, vmax=np.pi)
axes[1, 1].set_title(f'Low Organization: Distance Matrix\nSI = {si_low:.3f} (Type III)', 
                      fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('LAW Index', fontsize=10)
axes[1, 1].set_ylabel('LAW Index', fontsize=10)
plt.colorbar(im2, ax=axes[1, 1], label='Distance (rad)')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'step6_si_interpretation.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: step6_si_interpretation.png")
