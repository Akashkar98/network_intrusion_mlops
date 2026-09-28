from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import datetime

app = FastAPI(
    title="Network Intrusion Detection API"
)


model = joblib.load(
    "models/intrusion_detection_pipeline.joblib"
)


class NetworkFlow(BaseModel):

    Src_Port: float
    Dst_Port: float
    Protocol: int
    Flow_Duration: float
    Total_Fwd_Packet: float
    Total_Bwd_packets: float
    Total_Length_of_Fwd_Packet: float
    Fwd_Packet_Length_Min: float
    Bwd_Packet_Length_Max: float
    Flow_IAT_Min: float
    Fwd_IAT_Min: float
    Fwd_Header_Length: float
    Bwd_Packets_s: float
    Packet_Length_Max: float
    Packet_Length_Std: float
    RST_Flag_Count: float
    FWD_Init_Win_Bytes: float
    Bwd_Init_Win_Bytes: float
    FIN_Flag_Count: float
    SYN_Flag_Count: float
    ACK_Flag_Count: float


@app.get("/")
def home():

    return {
        "message": "Network Intrusion Detection API is running"
    }


@app.post("/predict")
def predict(data: NetworkFlow):

    input_data = data.model_dump()

    input_data = {
        "Src Port": input_data["Src_Port"],
        "Dst Port": input_data["Dst_Port"],
        "Protocol": input_data["Protocol"],
        "Flow Duration": input_data["Flow_Duration"],
        "Total Fwd Packet": input_data["Total_Fwd_Packet"],
        "Total Bwd packets": input_data["Total_Bwd_packets"],
        "Total Length of Fwd Packet": input_data["Total_Length_of_Fwd_Packet"],
        "Fwd Packet Length Min": input_data["Fwd_Packet_Length_Min"],
        "Bwd Packet Length Max": input_data["Bwd_Packet_Length_Max"],
        "Flow IAT Min": input_data["Flow_IAT_Min"],
        "Fwd IAT Min": input_data["Fwd_IAT_Min"],
        "Fwd Header Length": input_data["Fwd_Header_Length"],
        "Bwd Packets/s": input_data["Bwd_Packets_s"],
        "Packet Length Max": input_data["Packet_Length_Max"],
        "Packet Length Std": input_data["Packet_Length_Std"],
        "RST Flag Count": input_data["RST_Flag_Count"],
        "FWD Init Win Bytes": input_data["FWD_Init_Win_Bytes"],
        "Bwd Init Win Bytes": input_data["Bwd_Init_Win_Bytes"],
        "FIN Flag Count": input_data["FIN_Flag_Count"],
        "SYN Flag Count": input_data["SYN_Flag_Count"],
        "ACK Flag Count": input_data["ACK_Flag_Count"]
    }

    input_df = pd.DataFrame([input_data])


    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    normal_probability = float(probabilities[0])
    attack_probability = float(probabilities[1])

    log_data = {
    "timestamp": datetime.now().isoformat(),
    **input_data,
    "prediction": int(prediction),
    "result": "Attack" if prediction == 1 else "Normal",
    "normal_probability": normal_probability,
    "attack_probability": attack_probability
}

    log_df = pd.DataFrame([log_data])

    log_file = "prediction_logs.csv"

    log_df.to_csv(
        log_file,
        mode="a",
        header=not os.path.exists(log_file),
        index=False
    )

    return {
        "prediction": int(prediction),
        "result": "Attack" if prediction == 1 else "Normal",
        "normal_probability": normal_probability,
        "attack_probability": attack_probability
    }