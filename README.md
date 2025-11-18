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

## Project Structure

```bash
biomedical-signal-processing/
├── af_data/                           # PhysioNet AF Database records
│   ├── afdb_records/                  # AFDB: Surface ECG (250 Hz) - 21 records
│   └── iafdb_records/                 # IAFDB: Intracardiac (977 Hz) - 32 records
├── docs/
│   ├── presentation.pdf
│   └── Faes2002_similarity_index.pdf
├── results/                           # Analysis results and plots
│   ├── afdb/                          # AFDB analysis results
│   │   ├── analysis_results.csv
│   │   ├── analysis_plots.png
│   │   └── analysis_summary.txt
│   ├── iafdb/                         # IAFDB analysis results
│   │   ├── analysis_results.csv
│   │   ├── analysis_plots.png
│   │   └── analysis_summary.txt
│   └── comparison/                    # Cross-dataset comparison
│       └── comparison_report.txt
├── src/                               # Core implementation modules
│   ├── af_algorithm.py                # Algorithm classes
│   ├── af_data_loader.py              # Data loading utilities
│   └── README.md
├── LICENSE                            # License information
├── main.py                            # Main execution script
├── requirements.txt                   # Python dependencies
└── README.md   
```

## Analysis Results & Performance

### Dataset Used
- **Source**: PhysioNet MIT-BIH Atrial Fibrillation Database (*AFDB*) and Intracardiac Atrial Fibrillation Database (*IAFDB*)
- **AFDB Records**: 21 real patient surface ECG recordings (250 Hz sampling)
- **IAFDB Records**: 32 real patient intracardiac recordings (977 Hz sampling)
- **Total Records**: 53
- **Duration**: First 30 seconds analyzed per record

### Key Performance Metrics
- **Total Signals Analyzed**: 53 (21 AFDB surface ECG + 32 IAFDB intracardiac)
- **Sampling Rates**: 250 Hz (AFDB), 977 Hz (IAFDB)
- **Analysis Duration**: First 30 seconds per record
- **Classification Results**: Algorithm performs consistently across both recording modalities, demonstrating robustness to different signal types and sampling rates. Results align with literature expectations for AF organization classification.

### Validation Against Literature
Expected ranges from the original paper[^1]:
- **Flutter**: SI ~ 1.00
- **Type I**: SI ~ 0.75 ± 0.23
- **Type II**: SI ~ 0.35 ± 0.11
- **Type III**: SI ~ 0.15 ± 0.08

Results show good alignment with expected ranges and demonstrate the algorithm's robustness across surface ECG and intracardiac recordings.

## Generated Outputs

The analysis produces the following materials for each dataset:

1. **CSV Results** (`results/afdb/analysis_results.csv`, `results/iafdb/analysis_results.csv`): Detailed metrics for each record
2. **Statistical Plots** (`results/afdb/analysis_plots.png`, `results/iafdb/analysis_plots.png`): SI distributions, boxplots by AF type, correlations
3. **Summary Report** (`results/afdb/analysis_summary.txt`, `results/iafdb/analysis_summary.txt`): Complete statistical summary and methodology
4. **Cross-Dataset Comparison** (`results/comparison/comparison_report.txt`): Comparative analysis between AFDB and IAFDB results

## Run it locally

1. Is highly recommended to create a virtual environment:

```bash
python -m venv af_analysis_env
source af_analysis_env/bin/activate  # On Windows use `af_analysis_env\Scripts\activate`
```

I highly recommend using conda, as you can specify the python version:

```bash
conda create -n af_analysis python=3.11.4
conda activate af_analysis
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download the PhysioNet MIT-BIH Atrial Fibrillation Database (AFDB) and Intracardiac Atrial Fibrillation Database (IAFDB) records and place them in the `af_data/` directory:
   - AFDB: https://physionet.org/content/afdb/1.0.0/
   - IAFDB: https://physionet.org/content/iafdb/1.0.0/

> [!NOTE]
> The data is already included in the repository for convenience.
> Also, ONLY the `.dat` and `.hea` files are necessary for the analysis. The other files are not included.

1. Run the main to execute the complete analysis:
```bash
python main.py
```

This will:
- Load all PhysioNet AF records present in the `af_data/` directory (both AFDB and IAFDB)
- Analyze each signal with progress indicator
- Generate all plots and reports for each dataset
- Create a cross-dataset comparison report
- Display final statistics

## References
- **Paper**: L. Faes, G. Nollo, R. Antolini, F. Gaita and F. Ravelli, "A method for quantifying atrial fibrillation organization based on wave-morphology similarity," in IEEE Transactions on Biomedical Engineering, vol. 49, no. 12, pp. 1504-1513, Dec. 2002, doi: [10.1109/TBME.2002.805472](https://doi.org/10.1109/tbme.2002.805472).

- **Datasets**: 
  - AFDB: Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220. RRID:SCR_007345. (https://doi.org/10.13026/C2MW2D)
  - IAFDB: Petrutiu, S., Sahakian, A. V., Fisher, W. G., & Swiryn, S. (2007). Intracardiac atrial fibrillation electrograms. PhysioNet. (https://doi.org/10.13026/C2F305)

[^1]: https://ieeexplore.ieee.org/document/1159144