import os
import pandas as pd
import numpy as np
import re

def analyze_lightning_logs(root_folder):
    """
    Analyzes Lightning logs for multiple versions, averages the final 2 test_f1 scores
    for each version, and then computes the mean and standard deviation across versions.

    Args:
        root_folder (str): The exact path to the lightning_logs directory.
    """
    all_version_averages = []
    lightning_logs_path = root_folder
    version_folders = [
        d for d in os.listdir(lightning_logs_path)
        if os.path.isdir(os.path.join(lightning_logs_path, d)) and d.startswith("version_")
    ]

    if not version_folders:
        print(f"No 'version_' folders found in '{lightning_logs_path}'.")
        return

    for version_folder in version_folders:
        version_path = os.path.join(lightning_logs_path, version_folder)
        metrics_file = os.path.join(version_path, "metrics.csv")

        if os.path.exists(metrics_file):
            try:
                df = pd.read_csv(metrics_file)
                if "test_f1" in df.columns:
                    test_f1_values = df["test_f1"].dropna()
                    num_test_f1 = len(test_f1_values)

                    if num_test_f1 > 0:
                        top_n = min(num_test_f1, 2)
                        last_n_f1 = test_f1_values.tail(top_n).values
                        average_f1 = np.mean(last_n_f1)
                        all_version_averages.append(average_f1)
                        print(f"Processed version: {version_folder}, Averaged last {top_n} test_f1 = {average_f1:.4f}")
                    else:
                        print(f"No 'test_f1' values found in {version_path}/metrics.csv")
                else:
                    print(f"'test_f1' column not found in {version_path}/metrics.csv")

            except Exception as e:
                print(f"Error reading {metrics_file}: {e}")
        else:
            print(f"metrics.csv not found in {version_path}")

    if all_version_averages:
        mean_avg_f1 = np.mean(all_version_averages)
        std_avg_f1 = np.std(all_version_averages)
        print("\n--- Summary Across All Processed Versions ---")
        print(f"Mean of average last 2 test_f1 scores: {mean_avg_f1:.4f}")
        print(f"Standard deviation of average last 2 test_f1 scores: {std_avg_f1:.4f}")
    else:
        print("\nNo valid 'test_f1' scores found across any processed versions.")

if __name__ == "__main__":
    target_logs_folder_path = "lightning_logs"
    analyze_lightning_logs(root_folder=target_logs_folder_path)