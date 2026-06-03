"""
=============================================================================
Multi-Task ST-GAT — Grid Management System
=============================================================================
Dataset:  UrbanEV (occupancy.csv + duration.csv + volume.csv + adj.csv)
Task:     24-hour lookback → 1-hour ahead prediction for THREE targets:
            1. Occupancy (%)
            2. Charging Duration (hours)
            3. Power Volume (kWh)

Architecture:
  Shared Backbone:
    ✅ Temporal Transformer with positional embeddings (per station, per variable)
    ✅ Spatial MultiheadAttention with official adj.csv mask (post-temporal)
  Multi-Task Heads:
    ✅ Occupancy Head: FC(embed_dim → 32 → 1)
    ✅ Duration Head:  FC(embed_dim → 32 → 1)
    ✅ Volume Head:    FC(embed_dim → 32 → 1)

Pipeline:
  ✅ Chronological split BEFORE normalization (70/10/20)
  ✅ Independent StandardScaler per variable (fit on TRAIN only)
  ✅ Weighted multi-task loss: L = w1*MSE_occ + w2*MSE_dur + w3*MSE_vol
  ✅ Metrics in both normalized and original scale per task
  ✅ Official adj.csv (boundary-based) → boolean mask

Important Data Note (from UrbanEV docs):
  "Occupancy data records all unavailable/busy charging piles.
   Duration and volume only account for piles actively providing electricity."
  These are complementary views of grid utilization.
=============================================================================
"""

# ── Cell 1: Imports ──────────────────────────────────────────────────────────

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device: {device}")

# ── Cell 2: Load ALL Raw Data ────────────────────────────────────────────────

INPUT_DIR = "./data"
DATASET_PATH = None
for root, dirs, files in os.walk(INPUT_DIR):
    if "occupancy.csv" in files and "duration.csv" in files and "volume.csv" in files:
        DATASET_PATH = root
        break

if DATASET_PATH is None:
    raise FileNotFoundError("Could not find occupancy.csv, duration.csv, and volume.csv in /kaggle/input/.")

print(f"📂 Dataset found at: {DATASET_PATH}")

# ── Load occupancy (Unit: %) ──
df_occ = pd.read_csv(os.path.join(DATASET_PATH, "occupancy.csv"))
time_col = df_occ.columns[0]
df_occ = df_occ.drop(columns=[time_col])
station_names = df_occ.columns.tolist()
NUM_STATIONS = len(station_names)

# ── Load duration (Unit: hours) ──
df_dur = pd.read_csv(os.path.join(DATASET_PATH, "duration.csv"))
df_dur = df_dur.drop(columns=[df_dur.columns[0]])

# ── Load volume (Unit: kWh) ──
df_vol = pd.read_csv(os.path.join(DATASET_PATH, "volume.csv"))
df_vol = df_vol.drop(columns=[df_vol.columns[0]])

raw_occ = df_occ.values.astype(np.float32)
raw_dur = df_dur.values.astype(np.float32)
raw_vol = df_vol.values.astype(np.float32)

TOTAL_HOURS = raw_occ.shape[0]

print(f"📊 Occupancy shape: {raw_occ.shape}")
print(f"📊 Duration shape:  {raw_dur.shape}")
print(f"📊 Volume shape:    {raw_vol.shape}")
print(f"🏢 Number of stations: {NUM_STATIONS}")
print(f"⏱️  Total hours: {TOTAL_HOURS}")

# Verify all datasets have same shape
assert raw_occ.shape == raw_dur.shape == raw_vol.shape, \
    f"Shape mismatch! occ={raw_occ.shape}, dur={raw_dur.shape}, vol={raw_vol.shape}"

# ── Cell 3: Data Statistics ──────────────────────────────────────────────────

print(f"\n{'='*60}")
print("RAW DATA STATISTICS")
print(f"{'='*60}")
for name, data in [("Occupancy (%)", raw_occ), ("Duration (hrs)", raw_dur), ("Volume (kWh)", raw_vol)]:
    print(f"  {name}:")
    print(f"    Range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"    Mean:  {data.mean():.2f} | Std: {data.std():.2f}")
    print(f"    Zeros: {(data == 0).sum()} ({(data == 0).sum() / data.size * 100:.1f}%)")

# ── Cell 4: Chronological Split (BEFORE normalization) ───────────────────────

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10

train_end = int(TOTAL_HOURS * TRAIN_RATIO)
val_end   = int(TOTAL_HOURS * (TRAIN_RATIO + VAL_RATIO))

# Split each variable independently
raw_train_occ = raw_occ[:train_end]
raw_val_occ   = raw_occ[train_end:val_end]
raw_test_occ  = raw_occ[val_end:]

raw_train_dur = raw_dur[:train_end]
raw_val_dur   = raw_dur[train_end:val_end]
raw_test_dur  = raw_dur[val_end:]

raw_train_vol = raw_vol[:train_end]
raw_val_vol   = raw_vol[train_end:val_end]
raw_test_vol  = raw_vol[val_end:]

print(f"\n{'='*60}")
print(f"CHRONOLOGICAL SPLIT (Leakage-Free)")
print(f"{'='*60}")
print(f"  Train: hours 0 → {train_end}    ({raw_train_occ.shape[0]} samples)")
print(f"  Val:   hours {train_end} → {val_end}  ({raw_val_occ.shape[0]} samples)")
print(f"  Test:  hours {val_end} → {TOTAL_HOURS} ({raw_test_occ.shape[0]} samples)")

# ── Cell 5: Leakage-Free Normalization (Independent StandardScaler per var) ──
#
# CRITICAL: Each variable gets its OWN scaler, fit on TRAINING block only
# This prevents cross-variable information leakage
# ─────────────────────────────────────────────────────────────────────────────

scaler_occ = StandardScaler()
scaler_dur = StandardScaler()
scaler_vol = StandardScaler()

# FIT on train ONLY
scaler_occ.fit(raw_train_occ)
scaler_dur.fit(raw_train_dur)
scaler_vol.fit(raw_train_vol)

# Transform all splits
norm_train_occ = scaler_occ.transform(raw_train_occ).astype(np.float32)
norm_val_occ   = scaler_occ.transform(raw_val_occ).astype(np.float32)
norm_test_occ  = scaler_occ.transform(raw_test_occ).astype(np.float32)

norm_train_dur = scaler_dur.transform(raw_train_dur).astype(np.float32)
norm_val_dur   = scaler_dur.transform(raw_val_dur).astype(np.float32)
norm_test_dur  = scaler_dur.transform(raw_test_dur).astype(np.float32)

norm_train_vol = scaler_vol.transform(raw_train_vol).astype(np.float32)
norm_val_vol   = scaler_vol.transform(raw_val_vol).astype(np.float32)
norm_test_vol  = scaler_vol.transform(raw_test_vol).astype(np.float32)

print(f"\n{'='*60}")
print(f"LEAKAGE-FREE NORMALIZATION (Independent StandardScaler)")
print(f"{'='*60}")
print(f"  Formula: X_scaled = (X - μ_train) / σ_train")
print(f"  Each variable has its OWN scaler fit on TRAINING only")
for name, scaler, norm_tr in [
    ("Occupancy", scaler_occ, norm_train_occ),
    ("Duration",  scaler_dur, norm_train_dur),
    ("Volume",    scaler_vol, norm_train_vol)]:
    print(f"\n  {name}:")
    print(f"    μ range: [{scaler.mean_.min():.2f}, {scaler.mean_.max():.2f}]")
    print(f"    σ range: [{scaler.scale_.min():.2f}, {scaler.scale_.max():.2f}]")
    print(f"    Train norm range: [{norm_tr.min():.4f}, {norm_tr.max():.4f}]")

# ── Cell 6: Multi-Variate Sliding Window Sequence Generation ─────────────────
# Stack all 3 variables into a single input tensor: [T, N, 3]
# The model sees all three variables simultaneously at each timestep

LOOKBACK = 24
HORIZON  = 1

def create_multivar_sequences(occ, dur, vol, lookback=24, horizon=1):
    """Create sequences with 3-channel input and 3-channel output.

    Input:  [samples, lookback, stations, 3] — all 3 vars as features
    Output: [samples, stations, 3] — predict all 3 vars at next timestep
    """
    # Stack along feature dim: [T, N, 3]
    stacked = np.stack([occ, dur, vol], axis=-1)  # [T, N, 3]

    X, Y = [], []
    for i in range(len(occ) - lookback - horizon + 1):
        X.append(stacked[i : i + lookback])        # [lookback, N, 3]
        Y.append(stacked[i + lookback])             # [N, 3]

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X_train, Y_train = create_multivar_sequences(norm_train_occ, norm_train_dur, norm_train_vol, LOOKBACK, HORIZON)
X_val,   Y_val   = create_multivar_sequences(norm_val_occ,   norm_val_dur,   norm_val_vol,   LOOKBACK, HORIZON)
X_test,  Y_test  = create_multivar_sequences(norm_test_occ,  norm_test_dur,  norm_test_vol,  LOOKBACK, HORIZON)

print(f"\n📦 Multi-variate Sequence shapes:")
print(f"   X_train: {X_train.shape}  Y_train: {Y_train.shape}")
print(f"   X_val:   {X_val.shape}  Y_val:   {Y_val.shape}")
print(f"   X_test:  {X_test.shape}  Y_test:  {Y_test.shape}")
print(f"   Input channels: 3 (occupancy, duration, volume)")

X_train_t = torch.tensor(X_train)
Y_train_t = torch.tensor(Y_train)
X_val_t   = torch.tensor(X_val)
Y_val_t   = torch.tensor(Y_val)
X_test_t  = torch.tensor(X_test)
Y_test_t  = torch.tensor(Y_test)

BATCH_SIZE = 32

train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, Y_val_t),
                          batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, Y_test_t),
                          batch_size=BATCH_SIZE, shuffle=False)

# ── Cell 7: Load Official Adjacency Matrix (adj.csv) → Boolean Mask ─────────
# Using the paper's official adj.csv (binary boundary-based adjacency)
# Paper: "a value of 1 indicates that two traffic zones are adjacent, otherwise 0"

print(f"\n{'='*60}")
print("LOADING SPATIAL ATTENTION MASK (Official adj.csv)")
print(f"{'='*60}")

# adj.csv has a header row with zone IDs — skip it, read data only
df_adj = pd.read_csv(os.path.join(DATASET_PATH, "adj.csv"), header=None, skiprows=1)
adj_values = df_adj.values.astype(np.float32)

print(f"  Adjacency matrix shape: {adj_values.shape}")
print(f"  Value range: [{adj_values.min():.4f}, {adj_values.max():.4f}]")
print(f"  Source: Official adj.csv (boundary-based, from paper)")

# Boolean mask: True = BLOCKED (adjacency == 0 means no spatial link)
attn_mask_np = (adj_values == 0.0)

num_connected = np.count_nonzero(~attn_mask_np)
total_pairs = attn_mask_np.size
sparsity = np.count_nonzero(attn_mask_np) / total_pairs

print(f"  Connected pairs: {num_connected} / {total_pairs}")
print(f"  Graph sparsity:  {sparsity*100:.1f}% (edges blocked)")
print(f"  Avg neighbors per station: {num_connected / NUM_STATIONS:.1f}")

attn_mask = torch.tensor(attn_mask_np, dtype=torch.bool).to(device)

# ── Cell 8: Multi-Task ST-GAT Architecture ──────────────────────────────────
#
# Architecture: Shared Backbone + Three Independent Heads
#
# Flow:
#   [B, 24, N, 3]
#     → feature_embed: [B, 24, N, 3] → [B*N, 24, embed_dim]
#     → TemporalTransformer: per station, 24h history → [B, N, embed_dim]
#     → SpatialGraphAttention: masked cross-station → [B, N, embed_dim]
#     → Head_occ: FC(embed_dim→32→1) → [B, N]
#     → Head_dur: FC(embed_dim→32→1) → [B, N]
#     → Head_vol: FC(embed_dim→32→1) → [B, N]

class TemporalTransformerMV(nn.Module):
    """Multi-variate Temporal Transformer.

    Processes each station's 24-hour history with 3-channel input
    (occupancy, duration, volume) using a Transformer Encoder.

    Input:  [B, T, N, 3]
    Output: [B, N, embed_dim]
    """
    def __init__(self, seq_len=24, num_features=3, embed_dim=64,
                 num_heads=4, num_layers=2, dropout=0.2):
        super().__init__()

        # Project 3 input features → embed_dim
        self.feature_embed = nn.Linear(num_features, embed_dim)

        # Learned positional embedding for the 24-hour sequence
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        B, T, N, F = x.shape

        # [B, T, N, F] → [B, N, T, F] → [B*N, T, F]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)

        # Project features → embed_dim, add positional encoding
        x = self.feature_embed(x) + self.pos_embed  # [B*N, T, embed_dim]

        # Transformer self-attention over time dimension
        x = self.transformer(x)

        # Take the LAST timestep's embedding
        # [B*N, T, embed_dim] → [B, N, embed_dim]
        x = x[:, -1, :]
        x = x.reshape(B, N, -1)
        return x


class SpatialGraphAttention(nn.Module):
    """Apply masked multi-head attention over the spatial dimension (stations).

    Each station's temporal embedding attends ONLY to its physical neighbors
    as defined by the adjacency matrix mask.

    Input:  [B, N, embed_dim] + mask [N, N]
    Output: [B, N, embed_dim], attention_weights
    """
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.2):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU()

    def forward(self, x, mask):
        attn_output, attn_weights = self.multihead_attn(
            query=x, key=x, value=x,
            attn_mask=mask  # True = blocked
        )
        out = self.layer_norm(x + self.activation(attn_output))
        return out, attn_weights


class MultiTaskSTGAT(nn.Module):
    """Multi-Task ST-GAT: Shared Backbone + 3 Prediction Heads.

    Shared Backbone:
        TemporalTransformerMV → SpatialGraphAttention

    Heads (independent):
        Occupancy:  FC(embed_dim → 32 → 1) → [B, N]
        Duration:   FC(embed_dim → 32 → 1) → [B, N]
        Volume:     FC(embed_dim → 32 → 1) → [B, N]

    Flow:
        [B, 24, 275, 3]
          → TemporalTransformerMV: each station's 24h × 3var → [B, 275, 64]
          → SpatialGraphAttention: masked cross-station → [B, 275, 64]
          → 3 × PredictionHead: per-station FC → [B, 275] each
    """
    def __init__(self, seq_len=24, num_stations=275, num_features=3,
                 embed_dim=64, num_heads=4, num_temporal_layers=2, dropout=0.2):
        super().__init__()

        # ── Shared Backbone ──
        self.temporal_module = TemporalTransformerMV(
            seq_len=seq_len,
            num_features=num_features,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_temporal_layers,
            dropout=dropout
        )

        self.spatial_gat = SpatialGraphAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # ── Task-Specific Prediction Heads ──
        def make_head():
            return nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            )

        self.head_occ = make_head()
        self.head_dur = make_head()
        self.head_vol = make_head()

    def forward(self, x, mask):
        # x: [B, T, N, 3]

        # Step 1: Temporal encoding (shared backbone)
        temp_features = self.temporal_module(x)       # [B, N, embed_dim]

        # Step 2: Spatial graph attention (shared backbone)
        spat_features, attn_weights = self.spatial_gat(temp_features, mask)
        # spat_features: [B, N, embed_dim]

        # Step 3: Task-specific predictions
        pred_occ = self.head_occ(spat_features).squeeze(-1)   # [B, N]
        pred_dur = self.head_dur(spat_features).squeeze(-1)   # [B, N]
        pred_vol = self.head_vol(spat_features).squeeze(-1)   # [B, N]

        return pred_occ, pred_dur, pred_vol, attn_weights


# ── Cell 9: Multi-Task Training Loop ────────────────────────────────────────

def train_multitask(model, train_loader, val_loader, epochs, lr, model_name,
                    attn_mask=None, patience=10,
                    w_occ=1.0, w_dur=1.0, w_vol=1.0):
    """Train with weighted multi-task loss.

    Loss = w_occ * MSE(occ) + w_dur * MSE(dur) + w_vol * MSE(vol)
    """

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_losses = []

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Optimizer: AdamW (lr={lr}, weight_decay=1e-4)")
    print(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
    print(f"  Epochs: {epochs} | Early Stop Patience: {patience}")
    print(f"  Loss weights: occ={w_occ}, dur={w_dur}, vol={w_vol}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training phase ──
        model.train()
        epoch_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)  # [B, T, N, 3]
            batch_y = batch_y.to(device)  # [B, N, 3]

            optimizer.zero_grad()

            pred_occ, pred_dur, pred_vol, _ = model(batch_x, attn_mask)

            # Multi-task loss
            loss_occ = criterion(pred_occ, batch_y[:, :, 0])
            loss_dur = criterion(pred_dur, batch_y[:, :, 1])
            loss_vol = criterion(pred_vol, batch_y[:, :, 2])

            loss = w_occ * loss_occ + w_dur * loss_dur + w_vol * loss_vol

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # ── Validation phase ──
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                pred_occ, pred_dur, pred_vol, _ = model(batch_x, attn_mask)

                loss_occ = criterion(pred_occ, batch_y[:, :, 0])
                loss_dur = criterion(pred_dur, batch_y[:, :, 1])
                loss_vol = criterion(pred_vol, batch_y[:, :, 2])

                loss = w_occ * loss_occ + w_dur * loss_dur + w_vol * loss_vol

                epoch_val_loss += loss.item() * batch_x.size(0)

        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " ✅ (best)"
        else:
            patience_counter += 1
            marker = ""

        if epoch % 5 == 0 or epoch == 1 or patience_counter == 0:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train: {epoch_train_loss:.6f} | "
                  f"Val: {epoch_val_loss:.6f} | "
                  f"LR: {current_lr:.2e}{marker}")

        if patience_counter >= patience:
            print(f"  ⏹️  Early stopping at epoch {epoch} (patience={patience})")
            break

    elapsed = time.time() - start_time
    print(f"  ⏱️  Training time: {elapsed:.1f}s")

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses


# ── Cell 10: Multi-Task Evaluation ──────────────────────────────────────────

def evaluate_multitask(model, test_loader, scaler_occ, scaler_dur, scaler_vol,
                       model_name, attn_mask=None):
    """Evaluate all three tasks. Reports RMSE/MAE in normalized and original scale."""

    model.eval()
    all_pred_occ, all_pred_dur, all_pred_vol = [], [], []
    all_tgt_occ,  all_tgt_dur,  all_tgt_vol  = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)

            pred_occ, pred_dur, pred_vol, _ = model(batch_x, attn_mask)

            all_pred_occ.append(pred_occ.cpu().numpy())
            all_pred_dur.append(pred_dur.cpu().numpy())
            all_pred_vol.append(pred_vol.cpu().numpy())

            all_tgt_occ.append(batch_y[:, :, 0].numpy())
            all_tgt_dur.append(batch_y[:, :, 1].numpy())
            all_tgt_vol.append(batch_y[:, :, 2].numpy())

    preds = {
        'occ': np.concatenate(all_pred_occ, axis=0),
        'dur': np.concatenate(all_pred_dur, axis=0),
        'vol': np.concatenate(all_pred_vol, axis=0),
    }
    tgts = {
        'occ': np.concatenate(all_tgt_occ, axis=0),
        'dur': np.concatenate(all_tgt_dur, axis=0),
        'vol': np.concatenate(all_tgt_vol, axis=0),
    }

    scalers = {'occ': scaler_occ, 'dur': scaler_dur, 'vol': scaler_vol}
    labels  = {'occ': 'Occupancy (%)', 'dur': 'Duration (hrs)', 'vol': 'Volume (kWh)'}

    results = {}

    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {model_name}")
    print(f"{'='*60}")

    for key in ['occ', 'dur', 'vol']:
        p_norm = preds[key]
        t_norm = tgts[key]

        rmse_norm = np.sqrt(mean_squared_error(t_norm.flatten(), p_norm.flatten()))
        mae_norm  = mean_absolute_error(t_norm.flatten(), p_norm.flatten())

        # Inverse transform to original scale
        p_orig = scalers[key].inverse_transform(p_norm)
        t_orig = scalers[key].inverse_transform(t_norm)

        rmse_orig = np.sqrt(mean_squared_error(t_orig.flatten(), p_orig.flatten()))
        mae_orig  = mean_absolute_error(t_orig.flatten(), p_orig.flatten())

        results[key] = {
            'rmse_norm': rmse_norm, 'mae_norm': mae_norm,
            'rmse_orig': rmse_orig, 'mae_orig': mae_orig
        }

        print(f"\n  📊 {labels[key]}:")
        print(f"    Normalized:  RMSE={rmse_norm:.6f}  MAE={mae_norm:.6f}")
        print(f"    Original:    RMSE={rmse_orig:.4f}  MAE={mae_orig:.4f}")

    print(f"\n{'='*60}")

    return results


# ── Cell 11: Train & Evaluate ────────────────────────────────────────────────

print("\n" + "🔷" * 30)
print("Multi-Task ST-GAT — Grid Management System")
print("🔷" * 30)

model = MultiTaskSTGAT(
    seq_len=LOOKBACK,
    num_stations=NUM_STATIONS,
    num_features=3,            # occupancy + duration + volume
    embed_dim=64,
    num_heads=4,
    num_temporal_layers=2,
    dropout=0.2
).to(device)

# ── Train ──
model, train_losses, val_losses = train_multitask(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=60,
    lr=1e-3,
    model_name="Multi-Task ST-GAT",
    attn_mask=attn_mask,
    patience=10,
    # Equal weights — all three tasks weighted the same
    w_occ=1.0,
    w_dur=1.0,
    w_vol=1.0
)

# ── Evaluate ──
results = evaluate_multitask(
    model=model,
    test_loader=test_loader,
    scaler_occ=scaler_occ,
    scaler_dur=scaler_dur,
    scaler_vol=scaler_vol,
    model_name="Multi-Task ST-GAT",
    attn_mask=attn_mask
)


# ── Cell 12: Final Summary ──────────────────────────────────────────────────

total_params = sum(p.numel() for p in model.parameters())

print("\n")
print("╔" + "═"*62 + "╗")
print("║" + "  Multi-Task ST-GAT — GRID MANAGEMENT SUMMARY".center(62) + "║")
print("╠" + "═"*62 + "╣")
print("║" + f"  Total Parameters: {total_params:,}".ljust(62) + "║")
print("╠" + "═"*62 + "╣")

for key, label, unit in [
    ('occ', 'OCCUPANCY', '%'),
    ('dur', 'DURATION', 'hrs'),
    ('vol', 'VOLUME', 'kWh')]:

    r = results[key]
    print("║" + f"  📊 {label} ({unit})".ljust(62) + "║")
    print("║" + f"    Norm  → RMSE: {r['rmse_norm']:.6f}  MAE: {r['mae_norm']:.6f}".ljust(62) + "║")
    print("║" + f"    Orig  → RMSE: {r['rmse_orig']:.4f}  MAE: {r['mae_orig']:.4f}".ljust(62) + "║")
    if key != 'vol':
        print("╠" + "═"*62 + "╣")

print("╚" + "═"*62 + "╝")

print(f"\n✅ Multi-Task ST-GAT complete!")
print(f"")
print(f"   🔒 DATA INTEGRITY:")
print(f"   ├── 3 variables: occupancy(%), duration(hrs), volume(kWh)")
print(f"   ├── Independent StandardScaler per variable")
print(f"   ├── All scalers fit on Training block only ({raw_train_occ.shape[0]} hours)")
print(f"   └── Val/Test NEVER seen by ANY scaler during fitting")
print(f"")
print(f"   🧠 ARCHITECTURE:")
print(f"   ├── Input: 3-channel (occ, dur, vol) per station per timestep")
print(f"   ├── Shared: Temporal Transformer (2-layer) → Spatial GAT")
print(f"   ├── Heads: 3 independent FC(64→32→1) prediction heads")
print(f"   ├── Spatial: Official adj.csv → boolean mask")
print(f"   └── Optimizer: AdamW + ReduceLROnPlateau")
print(f"")
print(f"   ⚖️  LOSS FUNCTION:")
print(f"   └── L = 1.0×MSE(occ) + 1.0×MSE(dur) + 1.0×MSE(vol)")
print(f"")
print(f"   📋 DATA NOTE:")
print(f"   ├── Occupancy: all unavailable/busy charging piles")
print(f"   └── Duration/Volume: only piles actively providing electricity")

torch.save(model.state_dict(), "model.pth")
