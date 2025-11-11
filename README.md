# Biomedical Signal Processing: AF Organization Analysis

This project implements the atrial fibrillation (AF) organization analysis algorithm proposed in the paper "A Method for Quantifying Atrial Fibrillation Organization Based on Wave-Morphology Similarity" (IEEE Transactions on Biomedical Engineering, 2002)[^1].

_This project is intended for educational purposes only_

## Algorithm Description

The algorithm quantifies the degree of organization in atrial fibrillation by analyzing the morphological similarity between Local Activation Waves (LAWs).

The key components are:
1. **Signal Preprocessing**: Bandpass filtering (40-250 Hz) and envelope detection
2. **Activation Detection**: Adaptive threshold detection of atrial activations
3. **LAW Extraction**: Extraction of 90ms windows centered on detected activations
4. **Morphological Analysis**: L2 normalization and geodesic distance computation on the unit sphere
5. **Regularity Index (SI)**: Calculation of the Similarity Index using the Heaviside function


[^1]: https://ieeexplore.ieee.org/document/1159144