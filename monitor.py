import pandas as pd
from scipy.stats import ks_2samp

# Load reference/training data
reference_data = pd.read_csv("clean_dataset.csv")

# Load production traffic
production_data = pd.read_csv("prediction_logs.csv")

# Features used by the model
features = [
    col for col in reference_data.columns
    if col != "Label"
]

print("Reference samples:", len(reference_data))
print("Production samples:", len(production_data))

print("\nDRIFT REPORT \n")

drifted_features = []

for feature in features:

    reference_values = reference_data[feature].dropna()
    production_values = production_data[feature].dropna()

    ks_statistic, p_value = ks_2samp(
        reference_values,
        production_values
    )

    if p_value < 0.05:
        status = "DRIFT DETECTED"
        drifted_features.append(feature)
    else:
        status = "No Drift"

    print(
        f"{feature:30} | "
        f"KS: {ks_statistic:.4f} | "
        f"P-value: {p_value:.4f} | "
        f"{status}"
    )

print("\n==================================")

print(f"Total features checked: {len(features)}")
print(f"Drifted features: {len(drifted_features)}")

if len(drifted_features) > 0:
    print("\nDrift detected in:")
    for feature in drifted_features:
        print("-", feature)
else:
    print("\nNo significant data drift detected.")


MIN_PRODUCTION_SAMPLES = 500
DRIFT_THRESHOLD = 0.30  

total_features = len(features)
drifted_count = len(drifted_features)

drift_ratio = drifted_count / total_features

print("\n========== RETRAINING CHECK ==========")

print(f"Production samples: {len(production_data)}")
print(f"Drifted features: {drifted_count}/{total_features}")
print(f"Drift ratio: {drift_ratio:.2%}")
print(f"Trigger threshold: {DRIFT_THRESHOLD:.0%}")

if len(production_data) < MIN_PRODUCTION_SAMPLES:

    print("\nNot enough production data.")
    print("Retraining NOT triggered.")

elif drift_ratio >= DRIFT_THRESHOLD:

    print("\n🚨 SIGNIFICANT DATA DRIFT DETECTED")
    print("Retraining workflow TRIGGERED.")

else:

    print("\nData drift is within acceptable range.")
    print("Retraining NOT required.")