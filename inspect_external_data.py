import pandas as pd

file_path = (
    "Balanced_IEC104_Train_Test_CSV_Files/"
    "iec104_train_test_csvs/"
    "tests_cic_120/"
    "iec104_CICFLOWMETER_test_final.csv"
)

# Load external dataset
df = pd.read_csv(file_path)

print("=" * 60)
print("EXTERNAL DATASET INSPECTION")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {repr(column)}")

print("\nFirst 3 rows:")
print(df.head(3).to_string())

print("\nData types:")
print(df.dtypes)

print("\n========== LABEL DISTRIBUTION ==========")

print(df["Label"].value_counts())

print("\nUnique labels:")
print(df["Label"].unique())

print("\nTotal unique labels:")
print(df["Label"].nunique())