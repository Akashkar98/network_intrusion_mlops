import pandas as pd
import requests
import time

# External IEC-104 dataset
file_path = (
    "Balanced_IEC104_Train_Test_CSV_Files/"
    "iec104_train_test_csvs/"
    "tests_cic_120/"
    "iec104_CICFLOWMETER_test_final.csv"
)

df = pd.read_csv(file_path)

# Take 500 external network flows
samples = df.sample(n=500, random_state=42)

API_URL = "http://127.0.0.1:8000/predict"

print(f"Sending {len(samples)} EXTERNAL network flows...\n")

for i, (_, row) in enumerate(samples.iterrows(), start=1):

    payload = {
        "Src_Port": float(row["Src Port"]),
        "Dst_Port": float(row["Dst Port"]),
        "Protocol": int(row["Protocol"]),
        "Flow_Duration": float(row["Flow Duration"]),

        "Total_Fwd_Packet": float(row["Tot Fwd Pkts"]),
        "Total_Bwd_packets": float(row["Tot Bwd Pkts"]),
        "Total_Length_of_Fwd_Packet": float(row["TotLen Fwd Pkts"]),

        "Fwd_Packet_Length_Min": float(row["Fwd Pkt Len Min"]),
        "Bwd_Packet_Length_Max": float(row["Bwd Pkt Len Max"]),

        "Flow_IAT_Min": float(row["Flow IAT Min"]),
        "Fwd_IAT_Min": float(row["Fwd IAT Min"]),
        "Fwd_Header_Length": float(row["Fwd Header Len"]),

        "Bwd_Packets_s": float(row["Bwd Pkts/s"]),

        "Packet_Length_Max": float(row["Pkt Len Max"]),
        "Packet_Length_Std": float(row["Pkt Len Std"]),

        "RST_Flag_Count": float(row["RST Flag Cnt"]),

        "FWD_Init_Win_Bytes": float(row["Init Fwd Win Byts"]),
        "Bwd_Init_Win_Bytes": float(row["Init Bwd Win Byts"]),

        "FIN_Flag_Count": float(row["FIN Flag Cnt"]),
        "SYN_Flag_Count": float(row["SYN Flag Cnt"]),
        "ACK_Flag_Count": float(row["ACK Flag Cnt"])
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            print(
                f"{i}/500 | "
                f"Prediction: {result['result']} | "
                f"Attack probability: "
                f"{result['attack_probability']:.2f}"
            )

        else:
            print(
                f"{i}/500 | "
                f"ERROR {response.status_code} | "
                f"{response.text}"
            )

    except Exception as error:
        print(f"{i}/500 | ERROR: {error}")

    time.sleep(0.02)

print("\nExternal traffic simulation completed.")