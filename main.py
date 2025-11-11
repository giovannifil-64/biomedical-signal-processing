import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from src.af_algorithm import AFOrganizationAnalyzer, AFPerformanceEvaluator
from src.af_data_loader import PhysioNetDataLoader


def save_analysis_results(evaluator, test_records, fs):
    """
    Save detailed analysis results, plots, and statistics for presentation.
    """
    import os
    os.makedirs("results", exist_ok=True)

    print("\nSaving analysis results and plots...")

    # Save detailed results to CSV
    df = pd.DataFrame(evaluator.results)
    csv_path = "results/analysis_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] Detailed results saved to: {csv_path}")

    # Create SI distribution plot
    plt.figure(figsize=(12, 8))

    # SI histogram
    plt.subplot(2, 2, 1)
    si_values = np.array(df['SI'].values)
    plt.hist(si_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(float(np.mean(si_values)), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(si_values):.3f}')
    plt.xlabel('Regularity Index (SI)')
    plt.ylabel('Frequency')
    plt.title('SI Distribution Across All Records')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # SI by predicted type (boxplot)
    plt.subplot(2, 2, 2)
    types = ["flutter", "type1", "type2", "type3"]
    si_by_type = []
    labels = []

    for af_type in types:
        subset = df[df["predicted_type"] == af_type]
        if len(subset) > 0:
            si_by_type.append(subset['SI'].values)
            labels.append(f'{af_type.upper()}\n(n={len(subset)})')

    if si_by_type:
        bp = plt.boxplot(si_by_type, labels=labels, patch_artist=True)
        plt.setp(bp['boxes'], facecolor='lightblue')
        plt.setp(bp['medians'], color='red', linewidth=2)

        # Add reference lines from paper
        plt.axhline(y=0.49, color='orange', linestyle='--', alpha=0.7, label='Type I/II threshold')
        plt.axhline(y=0.24, color='red', linestyle='--', alpha=0.7, label='Type II/III threshold')

    plt.ylabel('Regularity Index (SI)')
    plt.title('SI Distribution by AF Type')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Number of LAWs vs SI
    plt.subplot(2, 2, 3)
    plt.scatter(df['n_laws'], df['SI'], alpha=0.6, s=50, color='green')
    plt.xlabel('Number of LAWs')
    plt.ylabel('Regularity Index (SI)')
    plt.title('LAWs Count vs SI')
    plt.grid(True, alpha=0.3)

    # Activations vs LAWs
    plt.subplot(2, 2, 4)
    plt.scatter(df['n_activations'], df['n_laws'], alpha=0.6, s=50, color='purple')
    plt.xlabel('Number of Activations')
    plt.ylabel('Number of Valid LAWs')
    plt.title('Activations vs Valid LAWs')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "results/si_analysis_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Analysis plots saved to: {plot_path}")

    # Save example LAW plots for a representative record
    if test_records:
        # Select a representative record (middle of the sorted list for variety)
        rep_index = len(test_records) // 2  # Middle record
        rep_record = test_records[rep_index]

        print(f"\nGenerating detailed analysis plots for record {rep_record}...")

        try:
            # Load signal for representative record
            data_loader = PhysioNetDataLoader(data_dir="af_data")
            signal_data, fs = data_loader.read_mit_data(rep_record, signal_index=0)

            if len(signal_data) > 0:
                # Analyze first 10 seconds for detailed plotting
                segment_length = min(10 * fs, len(signal_data))
                signal_segment = signal_data[:segment_length]

                # Update analyzer
                analyzer = AFOrganizationAnalyzer(fs=fs, epsilon=np.pi/3)
                analyzer.law_samples = int((analyzer.law_duration_ms / 1000) * fs)
                analyzer.blanking_samples = int((analyzer.blanking_ms / 1000) * fs)

                # Get analysis results
                SI, activations, laws = analyzer.analyze_signal(signal_segment, plot=False)

                # Find the result for this record
                rep_result = None
                for result in evaluator.results:
                    if result['label'] == f"Record_{rep_record}":
                        rep_result = result
                        break

                # Plot 1: Signal with activations
                plt.figure(figsize=(14, 10))

                plt.subplot(3, 1, 1)
                time = np.arange(len(signal_segment)) / fs
                plt.plot(time, signal_segment, 'b-', linewidth=1, alpha=0.8)
                if len(activations) > 0:
                    plt.plot(time[activations], signal_segment[activations], 'r^',
                           markersize=10, label='Detected Activations')
                plt.xlabel('Time (s)')
                plt.ylabel('Atrial Signal (mV)')
                plt.title(f'PhysioNet Record {rep_record} - SI = {SI:.3f} ({rep_result["predicted_type"] if rep_result else "unknown"})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xlim(0, 10)  # Show first 10 seconds

                # Plot 2: Example LAWs
                plt.subplot(3, 1, 2)
                if laws:
                    law_time = np.arange(len(laws[0])) / fs * 1000  # Convert to ms
                    n_examples = min(8, len(laws))
                    # Create colors manually
                    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

                    for i in range(n_examples):
                        plt.plot(law_time, laws[i], color=colors[i],
                               label=f'LAW {i+1}', linewidth=1.5, alpha=0.8)

                    plt.xlabel('Time (ms)')
                    plt.ylabel('Amplitude (normalized)')
                    plt.title(f'Example Local Activation Waves (n={len(laws)} total)')
                    plt.legend(ncol=4, fontsize=8)
                    plt.grid(True, alpha=0.3)

                # Plot 3: LAW similarity matrix (if enough LAWs)
                plt.subplot(3, 1, 3)
                if len(laws) >= 5:
                    # Compute similarities for subset of LAWs
                    subset_laws = laws[:20]  # Limit for visualization
                    normalized = analyzer.normalize_laws(subset_laws)
                    similarities = np.dot(normalized, normalized.T)

                    plt.imshow(similarities, cmap='RdYlBu_r', aspect='equal',
                             vmin=-1, vmax=1)
                    plt.colorbar(label='Cosine Similarity')
                    plt.xlabel('LAW Index')
                    plt.ylabel('LAW Index')
                    plt.title(f'LAW Similarity Matrix (first {len(subset_laws)} LAWs)')
                    plt.grid(True, alpha=0.3)
                else:
                    plt.text(0.5, 0.5, f'Not enough LAWs for similarity matrix\n({len(laws)} available)',
                           ha='center', va='center', transform=plt.gca().transAxes,
                           fontsize=12)
                    plt.title('LAW Similarity Matrix')

                plt.tight_layout()
                individual_analysis_path = f"results/individual_analysis_{rep_record}.png"
                plt.savefig(individual_analysis_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"[OK] Individual analysis plots saved to: {individual_analysis_path}")

        except Exception as e:
            print(f"  Error generating individual plots: {e}")

    # Save summary statistics to text file
    summary_path = "results/analysis_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("AF Organization Analysis - Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Algorithm: Faes et al. 2002\n")
        f.write(f"Records Analyzed: {len(test_records)}\n")
        f.write(f"Sampling Frequency: {fs} Hz\n")
        f.write(f"Epsilon Threshold: {np.pi/3:.3f} radians\n\n")

        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total Signals: {len(df)}\n")
        f.write(f"Average SI: {df['SI'].mean():.3f} ± {df['SI'].std():.3f}\n")
        f.write(f"SI Range: [{df['SI'].min():.3f}, {df['SI'].max():.3f}]\n")
        f.write(f"Average Activations: {df['n_activations'].mean():.1f}\n")
        f.write(f"Average Valid LAWs: {df['n_laws'].mean():.1f}\n\n")

        f.write("CLASSIFICATION RESULTS:\n")
        f.write("-" * 25 + "\n")
        for af_type in ["flutter", "type1", "type2", "type3"]:
            subset = df[df["predicted_type"] == af_type]
            if len(subset) > 0:
                f.write(f"{af_type.upper()}: {len(subset)} records\n")
                f.write(f"  SI: {subset['SI'].mean():.3f} ± {subset['SI'].std():.3f}\n")
                f.write(f"  Range: [{subset['SI'].min():.3f}, {subset['SI'].max():.3f}]\n\n")

        f.write("EXPECTED RANGES (Faes et al. 2002):\n")
        f.write("-" * 35 + "\n")
        f.write("Flutter:  SI ~ 1.00\n")
        f.write("Type I:   SI ~ 0.75 ± 0.23\n")
        f.write("Type II:  SI ~ 0.35 ± 0.11\n")
        f.write("Type III: SI ~ 0.15 ± 0.08\n\n")

        f.write("RECORD-BY-RECORD RESULTS:\n")
        f.write("-" * 28 + "\n")
        for _, row in df.iterrows():
            f.write(f"{row['label']:12s}: SI={row['SI']:.3f}, Type={row['predicted_type']}, LAWs={row['n_laws']}\n")

    print(f"[OK] Summary report saved to: {summary_path}")
    print(f"\nAll results saved in: results/ directory")


def main():
    """
    Complete demonstration of the Faes et al. 2002 AF organization algorithm.
    """
    print("Biomedical Signal Processing - AF Organization Analysis")
    print("Complete Implementation of Faes et al. 2002 Algorithm")
    print("=" * 60)

    # Initialize components
    analyzer = AFOrganizationAnalyzer(fs=250, epsilon=np.pi/3)
    data_loader = PhysioNetDataLoader(data_dir="af_data")
    evaluator = AFPerformanceEvaluator(analyzer)

    # Get all available records from af_data directory
    if os.path.exists("af_data"):
        all_files = os.listdir("af_data")
        # Extract record IDs from .dat files
        test_records = sorted([f.split('.')[0] for f in all_files if f.endswith('.dat')])
        # Process ALL records (no limit)
    else:
        print("Error: af_data directory not found!")
        return

    print(f"Found {len(test_records)} PhysioNet AF records in af_data/")

    # Get sampling frequency (assume all records have same fs)
    fs = 250  # Default PhysioNet AF sampling frequency

    # Analyze all records with progress indicator
    for i, record_id in enumerate(test_records, 1):
        print(f"Analyzing record {record_id}... ({i}/{len(test_records)})")

        try:
            # Load signal
            signal_data, fs = data_loader.read_mit_data(record_id, signal_index=0)

            if len(signal_data) == 0:
                print(f"  Failed to read signal data")
                continue

            # Analyze first 30 seconds
            segment_length = min(30 * fs, len(signal_data))
            signal_segment = signal_data[:segment_length]

            # Update analyzer sampling frequency
            analyzer.fs = fs
            analyzer.law_samples = int((analyzer.law_duration_ms / 1000) * fs)
            analyzer.blanking_samples = int((analyzer.blanking_ms / 1000) * fs)

            # Analyze
            result = evaluator.evaluate_signal(
                signal_segment, label=f"Record_{record_id}", true_type=None
            )

            # Minimal output - just completion status
            print(f"  [OK] SI: {result['SI']:.3f}, Type: {result['predicted_type']}")

        except Exception as e:
            print(f"  [ERROR] {e}")

    # Print comprehensive summary
    evaluator.print_summary()

    # Save detailed results and plots
    save_analysis_results(evaluator, test_records, fs)

    # Expected ranges from paper
    print("\nExpected ranges from Faes et al. 2002:")
    print("  Flutter:  SI ~ 1.00")
    print("  Type I:   SI ~ 0.75 ± 0.23")
    print("  Type II:  SI ~ 0.35 ± 0.11")
    print("  Type III: SI ~ 0.15 ± 0.08")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()