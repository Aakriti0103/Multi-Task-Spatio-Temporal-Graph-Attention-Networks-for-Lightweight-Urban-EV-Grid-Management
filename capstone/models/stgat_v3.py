"""
=============================================================================
ST-GAT v3 — Best of Both Worlds
=============================================================================
Dataset:  UrbanEV (occupancy.csv + adj.csv)
Task:     24-hour lookback → 1-hour ahead occupancy prediction (275 stations)

Architecture from old model (Temporal Transformer → Spatial GAT):
  ✅ Temporal Transformer with positional embeddings (per-station)
  ✅ Spatial MultiheadAttention with adjacency mask (post-temporal)
  ✅ embed_dim=64, proper capacity

Pipeline from v2 (leakage-free):
  ✅ Chronological split BEFORE normalization
  ✅ StandardScaler.fit() on TRAIN only
  ✅ Official adj.csv (boundary-based) → boolean mask
  ✅ Metrics in both normalized and original scale
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

# ── Cell 2: Load Raw Data ────────────────────────────────────────────────────

INPUT_DIR = "./data"
DATASET_PATH = None
for root, dirs, files in os.walk(INPUT_DIR):
    if "occupancy.csv" in files:
        DATASET_PATH = root
        break

if DATASET_PATH is None:
    raise FileNotFoundError("Could not find occupancy.csv in /kaggle/input/.")

print(f"📂 Dataset found at: {DATASET_PATH}")

df_occ = pd.read_csv(os.path.join(DATASET_PATH, "occupancy.csv"))
print(f"📊 Occupancy shape: {df_occ.shape}")

time_col = df_occ.columns[0]
df_occ = df_occ.drop(columns=[time_col])
station_names = df_occ.columns.tolist()
NUM_STATIONS = len(station_names)
print(f"🏢 Number of stations: {NUM_STATIONS}")

raw_data = df_occ.values.astype(np.float32)
TOTAL_HOURS = raw_data.shape[0]
print(f"⏱️  Total hours: {TOTAL_HOURS}")

# ── Cell 3: Chronological Split (BEFORE normalization) ───────────────────────

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10

train_end = int(TOTAL_HOURS * TRAIN_RATIO)
val_end   = int(TOTAL_HOURS * (TRAIN_RATIO + VAL_RATIO))

raw_train = raw_data[:train_end]
raw_val   = raw_data[train_end:val_end]
raw_test  = raw_data[val_end:]

print(f"\n{'='*60}")
print(f"CHRONOLOGICAL SPLIT (Leakage-Free)")
print(f"{'='*60}")
print(f"  Train: hours 0 → {train_end}    ({raw_train.shape[0]} samples)")
print(f"  Val:   hours {train_end} → {val_end}  ({raw_val.shape[0]} samples)")
print(f"  Test:  hours {val_end} → {TOTAL_HOURS} ({raw_test.shape[0]} samples)")

# ── Cell 4: Leakage-Free Normalization (StandardScaler) ──────────────────────
#
# StandardScaler formula (per station):
#   X_scaled = (X - μ_train) / σ_train
#
# CRITICAL: fit() on TRAINING block only, then transform() all splits
# ─────────────────────────────────────────────────────────────────────────────

scaler = StandardScaler()
scaler.fit(raw_train)  # FIT on train ONLY

norm_train = scaler.transform(raw_train).astype(np.float32)
norm_val   = scaler.transform(raw_val).astype(np.float32)
norm_test  = scaler.transform(raw_test).astype(np.float32)

print(f"\n{'='*60}")
print(f"LEAKAGE-FREE NORMALIZATION (StandardScaler / Z-Score)")
print(f"{'='*60}")
print(f"  Formula: X_scaled = (X - μ_train) / σ_train")
print(f"  Scaler fit on: TRAINING block only ({raw_train.shape[0]} hours)")
print(f"  Per-station mean (from train): [{scaler.mean_.min():.2f}, {scaler.mean_.max():.2f}]")
print(f"  Per-station std  (from train): [{scaler.scale_.min():.2f}, {scaler.scale_.max():.2f}]")
print(f"")
print(f"  Train normalized range: [{norm_train.min():.4f}, {norm_train.max():.4f}]")
print(f"  Val   normalized range: [{norm_val.min():.4f}, {norm_val.max():.4f}]")
print(f"  Test  normalized range: [{norm_test.min():.4f}, {norm_test.max():.4f}]")

# ── Cell 5: Sliding Window Sequence Generation ──────────────────────────────

LOOKBACK = 24
HORIZON  = 1

def create_sequences(data, lookback=24, horizon=1):
    X, Y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i : i + lookback])
        Y.append(data[i + lookback])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

X_train, Y_train = create_sequences(norm_train, LOOKBACK, HORIZON)
X_val,   Y_val   = create_sequences(norm_val,   LOOKBACK, HORIZON)
X_test,  Y_test  = create_sequences(norm_test,  LOOKBACK, HORIZON)

print(f"\n📦 Sequence shapes:")
print(f"   X_train: {X_train.shape}  Y_train: {Y_train.shape}")
print(f"   X_val:   {X_val.shape}  Y_val:   {Y_val.shape}")
print(f"   X_test:  {X_test.shape}  Y_test:  {Y_test.shape}")

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

# ── Cell 6: Load Official Adjacency Matrix (adj.csv) → Boolean Mask ──────────
# Using the paper's official adj.csv (binary boundary-based adjacency)
# instead of our Gaussian kernel adjacency_matrix.csv
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

# ── Cell 7: ST-GAT v3 Architecture ──────────────────────────────────────────
# This is your old architecture that achieved RMSE 0.057, kept intact.
# Flow: Temporal Transformer (per station) → Spatial GAT → Prediction Head

class TemporalTransformer(nn.Module):
    """Process each station's 24-hour history independently using a
    Transformer Encoder with learned positional embeddings.

    Input:  [B, T, N] → reshape to [B*N, T, 1] (each station = sequence)
    Output: [B, N, embed_dim] (last timestep embedding per station)
    """
    def __init__(self, seq_len=24, feature_dim=1, embed_dim=64,
                 num_heads=4, num_layers=2, dropout=0.2):
        super(TemporalTransformer, self).__init__()

        self.feature_embed = nn.Linear(feature_dim, embed_dim)

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
        B, T, N = x.shape

        # Treat each station as an independent sequence
        # [B, T, N] → [B, N, T] → [B*N, T, 1]
        x = x.transpose(1, 2).reshape(B * N, T, 1)

        # Project scalar → embed_dim, add positional encoding
        x = self.feature_embed(x) + self.pos_embed

        # Transformer self-attention over time dimension
        x = self.transformer(x)

        # Reshape back and take the LAST timestep's embedding
        # [B*N, T, embed_dim] → [B, N, T, embed_dim] → [B, N, embed_dim]
        x = x.reshape(B, N, T, -1)
        return x[:, :, -1, :]


class SpatialGraphAttention(nn.Module):
    """Apply masked multi-head attention over the spatial dimension (stations).

    Each station's temporal embedding attends ONLY to its physical neighbors
    as defined by the adjacency matrix mask.

    Input:  [B, N, embed_dim] + mask [N, N]
    Output: [B, N, embed_dim], attention_weights
    """
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.2):
        super(SpatialGraphAttention, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.activation = nn.ReLU()

    def forward(self, x, mask):
        # Self-attention where Q, K, V are all the temporal embeddings
        attn_output, attn_weights = self.multihead_attn(
            query=x, key=x, value=x,
            attn_mask=mask  # True = blocked (no attention)
        )
        # Residual connection + LayerNorm
        out = self.layer_norm(x + self.activation(attn_output))
        return out, attn_weights


class SpatioTemporalGAT_v3(nn.Module):
    """ST-GAT v3: Temporal Transformer → Spatial GAT → Prediction Head.

    This is the architecture that achieved RMSE 0.057, now paired with
    a leakage-free data pipeline.

    Flow:
        [B, 24, 275]
          → TemporalTransformer: each station's 24h history → [B, 275, 64]
          → SpatialGraphAttention: masked cross-station attention → [B, 275, 64]
          → PredictionHead: per-station FC → [B, 1, 275]
    """
    def __init__(self, seq_len=24, num_stations=275, embed_dim=64,
                 num_heads=4, num_temporal_layers=2, dropout=0.2):
        super(SpatioTemporalGAT_v3, self).__init__()

        self.temporal_module = TemporalTransformer(
            seq_len=seq_len,
            feature_dim=1,
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

        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, mask):
        # x: [B, T, N]

        # Step 1: Temporal encoding (per station, independently)
        temp_features = self.temporal_module(x)     # [B, N, embed_dim]

        # Step 2: Spatial graph attention (masked cross-station)
        spat_features, attn_weights = self.spatial_gat(temp_features, mask)
        # spat_features: [B, N, embed_dim]

        # Step 3: Per-station prediction
        pred = self.prediction_head(spat_features)  # [B, N, 1]
        pred = pred.squeeze(-1)                      # [B, N]

        return pred, attn_weights


# ── Cell 8: Training Loop ───────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs, lr, model_name,
                attn_mask=None, patience=10):
    """Train with AdamW + ReduceLROnPlateau + early stopping."""

    # AdamW with weight decay (same as your original model)
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
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training phase ──
        model.train()
        epoch_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds, _ = model(batch_x, attn_mask)
            loss = criterion(preds, batch_y)
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

                preds, _ = model(batch_x, attn_mask)
                loss = criterion(preds, batch_y)
                epoch_val_loss += loss.item() * batch_x.size(0)

        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # Step the LR scheduler
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # ── Early stopping ──
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

# ── Cell 9: Evaluation (Both Scales) ────────────────────────────────────────

def evaluate_model(model, test_loader, scaler, model_name, attn_mask=None):
    """Evaluate on test set. Reports RMSE/MAE in both normalized and original scale."""

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)

            preds, _ = model(batch_x, attn_mask)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.numpy())

    preds_norm = np.concatenate(all_preds, axis=0)
    targets_norm = np.concatenate(all_targets, axis=0)

    # ── Normalized scale ──
    rmse_norm = np.sqrt(mean_squared_error(targets_norm.flatten(), preds_norm.flatten()))
    mae_norm  = mean_absolute_error(targets_norm.flatten(), preds_norm.flatten())

    # ── Original scale (occupancy %) ──
    preds_original   = scaler.inverse_transform(preds_norm)
    targets_original = scaler.inverse_transform(targets_norm)

    rmse_orig = np.sqrt(mean_squared_error(targets_original.flatten(), preds_original.flatten()))
    mae_orig  = mean_absolute_error(targets_original.flatten(), preds_original.flatten())

    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"  Normalized Scale (Z-score):")
    print(f"    RMSE: {rmse_norm:.6f}")
    print(f"    MAE:  {mae_norm:.6f}")
    print(f"  Original Scale (occupancy %):")
    print(f"    RMSE: {rmse_orig:.4f} pp")
    print(f"    MAE:  {mae_orig:.4f} pp")
    print(f"{'='*60}")

    return rmse_norm, mae_norm, rmse_orig, mae_orig

# ── Cell 10: Train & Evaluate ────────────────────────────────────────────────

print("\n" + "🔷" * 30)
print("ST-GAT v3 (Temporal Transformer → Spatial GAT)")
print("🔷" * 30)

model = SpatioTemporalGAT_v3(
    seq_len=LOOKBACK,
    num_stations=NUM_STATIONS,
    embed_dim=64,              # same as your original model
    num_heads=4,               # same as your original model
    num_temporal_layers=2,     # 2-layer Transformer encoder
    dropout=0.2
).to(device)

model, train_losses, val_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=60,                 # same as your original
    lr=1e-3,
    model_name="ST-GAT v3",
    attn_mask=attn_mask,
    patience=10
)

rmse_norm, mae_norm, rmse_orig, mae_orig = evaluate_model(
    model=model,
    test_loader=test_loader,
    scaler=scaler,
    model_name="ST-GAT v3",
    attn_mask=attn_mask
)

# ── Cell 11: Final Summary ──────────────────────────────────────────────────

total_params = sum(p.numel() for p in model.parameters())

print("\n")
print("╔" + "═"*62 + "╗")
print("║" + "  ST-GAT v3 — FINAL SUMMARY".center(62) + "║")
print("╠" + "═"*62 + "╣")
print("║" + f"  Total Parameters: {total_params:,}".ljust(62) + "║")
print("╠" + "═"*62 + "╣")
print("║" + "  📏 NORMALIZED SCALE (Z-score)".ljust(62) + "║")
print("║" + f"    RMSE: {rmse_norm:.6f}".ljust(62) + "║")
print("║" + f"    MAE:  {mae_norm:.6f}".ljust(62) + "║")
print("╠" + "═"*62 + "╣")
print("║" + "  📊 ORIGINAL SCALE (occupancy %)".ljust(62) + "║")
print("║" + f"    RMSE: {rmse_orig:.4f} pp".ljust(62) + "║")
print("║" + f"    MAE:  {mae_orig:.4f} pp".ljust(62) + "║")
print("╚" + "═"*62 + "╝")

print(f"\n✅ ST-GAT v3 complete!")
print(f"")
print(f"   🔒 DATA LEAKAGE PREVENTION:")
print(f"   ├── Raw data split chronologically BEFORE any normalization")
print(f"   ├── StandardScaler.fit() called on Training block only ({raw_train.shape[0]} hours)")
print(f"   ├── Formula: X_norm = (X - μ_train) / σ_train")
print(f"   └── Val/Test NEVER seen by scaler during fitting")
print(f"")
print(f"   🧠 ARCHITECTURE (same as your best model):")
print(f"   ├── Temporal: 2-layer Transformer Encoder (per station, with pos embeddings)")
print(f"   ├── Spatial: Masked MultiheadAttention (adj.csv → boolean mask)")
print(f"   ├── Decoder: FC(64→32→1) per station")
print(f"   └── Optimizer: AdamW + ReduceLROnPlateau")
print(f"")
print(f"   📊 WHAT'S DIFFERENT FROM YOUR OLD 0.057 MODEL:")
print(f"   ├── Normalization: MinMaxScaler (leaky) → StandardScaler (leakage-free)")
print(f"   ├── Split order: normalize→split (wrong) → split→normalize (correct)")
print(f"   └── Everything else: IDENTICAL architecture")
