# Network Intrusion Detection MLOps System

An end-to-end Machine Learning and MLOps project for detecting malicious network traffic using network-flow features.

The project covers the complete lifecycle from data preprocessing and model experimentation to API deployment, Docker containerization, prediction logging, data-drift monitoring, and automated retraining triggers.

---

## Project Overview

The system classifies network flows into:

- `0` — Normal Traffic
- `1` — Attack Traffic

A Random Forest classifier was selected after comparison with baseline and Logistic Regression models.

The final model achieved approximately **99.64% accuracy on the held-out test split**.

> Note: The dataset does not provide reliable session/device identifiers or a suitable temporal field for a strict grouped/time-based split. Therefore, a stratified random split was used. Test performance should not be interpreted as real-world production IDS performance.

---

## System Architecture

```text
Network Traffic
       |
       v
    FastAPI
       |
       v
Saved ML Pipeline
       |
       v
Normal / Attack Prediction
       |
       v
Prediction Logging
       |
       v
Data Drift Monitoring
       |
       v
Retraining Trigger