# Multi-Task-Spatio-Temporal-Graph-Attention-Networks-for-Lightweight-Urban-EV-Grid-Management

## Overview

As Electric Vehicle (EV) adoption continues to grow, urban charging infrastructure faces increasing challenges due to fluctuating charging demand, station congestion, and grid instability. This project introduces a **Multi-Task Spatio-Temporal Graph Attention Network (MT-ST-GAT)** that simultaneously predicts:

* EV charging station occupancy
* Charging duration
* Energy consumption volume

By combining Transformer-based temporal modeling with Graph Attention Networks (GATs), the framework captures both temporal trends and spatial dependencies across urban charging stations.

## Key Features

* Multi-task forecasting of occupancy, duration, and energy volume
* Transformer encoder for temporal pattern learning
* Masked Graph Attention Network for spatial dependency modeling
* Leakage-free preprocessing pipeline with chronological data splitting
* Lightweight architecture (~124K parameters)
* State-of-the-art performance on the UrbanEV benchmark dataset

## Dataset

The project uses the **UrbanEV Dataset**, consisting of:

* 275 charging zones
* 4,344 hourly observations
* Occupancy (%)
* Charging Duration (hours)
* Energy Volume (kWh)

## Architecture

### Temporal Module

* Transformer Encoder
* Multi-Head Self-Attention
* Positional Encoding
* 24-hour lookback window

### Spatial Module

* Masked Graph Attention Network (GAT)
* Binary adjacency matrix based on geographic connectivity
* Oversmoothing prevention through boundary masking

### Multi-Task Prediction Heads

* Occupancy Prediction
* Charging Duration Prediction
* Energy Volume Prediction

## Model Configuration

| Parameter                | Value    |
| ------------------------ | -------- |
| Lookback Window          | 24 Hours |
| Prediction Horizon       | 1 Hour   |
| Transformer Layers       | 2        |
| Temporal Attention Heads | 4        |
| Spatial Attention Heads  | 4        |
| Embedding Dimension      | 64       |
| Batch Size               | 32       |
| Dropout                  | 0.2      |
| Optimizer                | AdamW    |
| Learning Rate            | 0.001    |

## Results

The proposed MT-ST-GAT model demonstrates strong predictive performance while maintaining computational efficiency.

### Highlights

* ~22% improvement in occupancy prediction accuracy compared to baseline approaches
* Charging Duration RMSE ≈ 1.72 hours
* Lightweight deployment-ready architecture
* Improved generalization through multi-task learning


## Future Work

* Real-time deployment on smart grid infrastructure
* Integration of weather and traffic data
* Dynamic pricing prediction
* Reinforcement learning-based charging optimization
* City-scale deployment across multiple urban regions
