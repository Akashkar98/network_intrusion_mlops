import pandas as pd
import requests
import time

# Load cleaned dataset
dataset = pd.read_csv("clean_dataset.csv")

# Take 100 random network flows
samples = dataset.sample(n=100, random_state=42)

API_URL = "http://127.0.0.1:8000/predict"

for i, (_, row) in enumerate(samples.iterrows(), start=1):

    payload = {
        "Src_Port": float(row["Src Port"]),
        "Dst_Port": float(row["Dst Port"]),
        "Protocol": int(row["Protocol"]),
        "Flow_Duration": float(row["Flow Duration"]),
        "Total_Fwd_Packet": float(row["Total Fwd Packet"]),
        "Total_Bwd_packets": float(row["Total Bwd packets"]),
        "Total_Length_of_Fwd_Packet": float(row["Total Length of Fwd Packet"]),
        "Fwd_Packet_Length_Min": float(row["Fwd Packet Length Min"]),
        "Bwd_Packet_Length_Max": float(row["Bwd Packet Length Max"]),
        "Flow_IAT_Min": float(row["Flow IAT Min"]),
        "Fwd_IAT_Min": float(row["Fwd IAT Min"]),
        "Fwd_Header_Length": float(row["Fwd Header Length"]),
        "Bwd_Packets_s": float(row["Bwd Packets/s"]),
        "Packet_Length_Max": float(row["Packet Length Max"]),
        "Packet_Length_Std": float(row["Packet Length Std"]),
        "RST_Flag_Count": float(row["RST Flag Count"]),
        "FWD_Init_Win_Bytes": float(row["FWD Init Win Bytes"]),
        "Bwd_Init_Win_Bytes": float(row["Bwd Init Win Bytes"]),
        "FIN_Flag_Count": float(row["FIN Flag Count"]),
        "SYN_Flag_Count": float(row["SYN Flag Count"]),
        "ACK_Flag_Count": float(row["ACK Flag Count"])
    }

    response = requests.post(API_URL, json=payload)

    print(
        f"{i}/100 | "
        f"Status: {response.status_code} | "
        f"Prediction: {response.json().get('result')}"
    )

    time.sleep(0.05)

print("\nTraffic simulation completed.")