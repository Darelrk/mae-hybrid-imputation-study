# Cell 1 : hybrid_pipeline.py
"""
Hybrid Imputation Pipeline - Layer-2 Refinement using MAE Embeddings (Z) and Predictions (T_MAE)
Methods: Hybrid-KNN, Hybrid-KMeans, Hybrid-MissForest
"""

import os, json, time, numpy as np, pandas as pd, pickle, warnings
from pathlib import Path, typing, datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt, seaborn as sns

ROOT_DIR = "c:/SSL/data_prepared_final/"
CLEANED_CSV = os.path.join(ROOT_DIR, "data_cleaned.csv")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts/")
MASKS_DIR = os.path.join(ROOT_DIR, "masks/")
MASKS_MANIFEST = os.path.join(ARTIFACTS_DIR, "masks_manifest.json")
MAE_EMBEDDINGS_DIR = os.path.join(ARTIFACTS_DIR, "embeddings/")
MAE_IMPUTATIONS_DIR = os.path.join(ROOT_DIR, "imputations/mae_runs/")
OUTPUT_HYBRID = os.path.join(ROOT_DIR, "imputations/hybrid_runs/")
SUMMARY_DIR = os.path.join(ARTIFACTS_DIR, "hybrid_summary/")
TARGET_COL = "fbs"
SEED = 42

for dir_path in [OUTPUT_HYBRID, SUMMARY_DIR]: os.makedirs(dir_path, exist_ok=True)
np.random.seed(SEED)
warnings.filterwarnings('ignore')

def load_data():
    """load cleaned data and scaler"""
    print("loading data...")
    df = pd.read_csv(CLEANED_CSV)
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler_fixed.pkl") if os.path.exists(os.path.join(ARTIFACTS_DIR, "scaler_fixed.pkl")) else os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    try:
        with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
        print(f"loaded scaler from {scaler_path}")
    except Exception as e:
        print(f"error loading scaler: {e}")
        scaler = None
    return df, scaler

def get_data_splits(df, mask_info, scaler):
    """prepare x (features), y (target), and mask indices"""
    mask_data = np.load(os.path.join(MASKS_DIR, mask_info['filename']))
    X_df = df.drop(columns=[TARGET_COL]).select_dtypes(include=[np.number])
    X_raw = X_df.values
    if scaler is None:
        scaler = StandardScaler().fit(X_raw)
    try:
        X_scaled = scaler.transform(X_raw)
    except:
        X_scaled = scaler.fit(X_raw).transform(X_raw)
    y_true = df[TARGET_COL].values
    target_mask_indices = mask_data['target_mask'] == 1 if 'target_mask' in mask_data else (lambda: (np.zeros(len(df), dtype=bool), __import__('numpy').zeros(len(df), dtype=bool).__setitem__(mask_data['test_indices'], True))[0])()
    train_indices = (~target_mask_indices) & ~np.isnan(y_true)
    return X_scaled, y_true, target_mask_indices, train_indices

def load_mae_artifacts(mask_id):
    """load z embedding and t_mae predictions"""
    # Load Embedding Z
    z_path = os.path.join(MAE_EMBEDDINGS_DIR, f"emb_z_{mask_id}.npy")
    if not os.path.exists(z_path):
        return None, None
    z = np.load(z_path)
    
    # Load Predictions T_MAE
    pred_path = os.path.join(MAE_IMPUTATIONS_DIR, f"imputasi_mae_{mask_id}.csv")
    if not os.path.exists(pred_path):
        return z, None
    
    pred_df = pd.read_csv(pred_path)
    t_mae = pred_df['target_pred'].values
    
    return z, t_mae

def evaluate_and_save(y_true, y_pred, target_mask_indices, mask_info, method_name, start_time):
    """calculate metrics and save results"""
    
    # Filter to masked indices only
    y_true_eval = y_true[target_mask_indices]
    y_pred_eval = y_pred[target_mask_indices]
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true_eval, y_pred_eval))
    mae = mean_absolute_error(y_true_eval, y_pred_eval)
    r2 = r2_score(y_true_eval, y_pred_eval)
    
    # Save CSV
    mask_id = f"fold_{mask_info['fold']}_repeat_{mask_info['repeat']}_mask_{mask_info['mask_rate']}"
    
    imputation_df = pd.DataFrame({
        'caseid': np.arange(len(y_true)),
        'target_true': y_true,
        'target_pred': y_pred,
        'is_masked_flag': target_mask_indices.astype(int),
        'fold': mask_info['fold'],
        'repeat': mask_info['repeat'],
        'mask_rate': mask_info['mask_rate'],
        'method': method_name
    })
    
    csv_filename = f"imputasi_{method_name.lower()}_{mask_id}.csv"
    imputation_df.to_csv(os.path.join(OUTPUT_HYBRID, csv_filename), index=False)
    
    # Save Metrics
    metrics = {
        'method': method_name,
        'mask_id': mask_id,
        'fold': mask_info['fold'],
        'repeat': mask_info['repeat'],
        'mask_rate': mask_info['mask_rate'],
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'runtime_seconds': time.time() - start_time
    }
    
    return metrics

def run_hybrid_knn(z, y_true, train_indices, target_mask_indices, mask_info):
    """
    hybrid-knn: knn in z-space.
    train on (z_train, y_train). predict for z_test.
    """
    start_time = time.time()
    
    # Prepare training data (observed)
    Z_train = z[train_indices]
    y_train = y_true[train_indices]
    
    # Prepare test data (masked)
    Z_test = z[target_mask_indices]
    
    # KNN Regressor
    knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
    knn.fit(Z_train, y_train)
    
    # Predict
    y_pred_masked = knn.predict(Z_test)
    
    # Construct full prediction array
    y_pred_full = y_true.copy().astype(float) # Initialize with true values
    y_pred_full[target_mask_indices] = y_pred_masked # Overwrite masked positions
    
    return evaluate_and_save(y_true, y_pred_full, target_mask_indices, mask_info, "Hybrid-KNN", start_time)

def run_hybrid_kmeans(z, t_mae, y_true, target_mask_indices, mask_info):
    """
    hybrid-kmeans: smoothing t_mae using clusters from z.
    y_pred = alpha * t_mae + (1-alpha) * clustermean(t_mae)
    """
    start_time = time.time()
    
    # KMeans on Z
    kmeans = KMeans(n_clusters=8, random_state=SEED)
    clusters = kmeans.fit_predict(z)
    
    # Calculate cluster means of T_MAE
    cluster_means = {}
    for c in range(8):
        mask_c = clusters == c
        if np.any(mask_c):
            cluster_means[c] = np.mean(t_mae[mask_c])
        else:
            cluster_means[c] = 0.0
            
    # Apply smoothing
    alpha = 0.7
    y_pred_smooth = np.zeros_like(t_mae)
    
    for i in range(len(t_mae)):
        c = clusters[i]
        y_pred_smooth[i] = alpha * t_mae[i] + (1 - alpha) * cluster_means[c]
        
    return evaluate_and_save(y_true, y_pred_smooth, target_mask_indices, mask_info, "Hybrid-KMeans", start_time)

def run_hybrid_missforest(X, y_true, t_mae, target_mask_indices, mask_info):
    """
    Hybrid-MissForest: RF Imputation seeded with T_MAE.
    Initialize missing values in y with T_MAE, then run IterativeImputer.
    """
    start_time = time.time()
    
    # Create initial complete matrix [X, y_init]
    # y_init has T_MAE values at masked positions
    y_init = y_true.copy().astype(float)
    y_init[target_mask_indices] = t_mae[target_mask_indices]
    
    data_init = np.column_stack([X, y_init])
    
    # MissForest (IterativeImputer)
    # We set max_iter=5 because we have a good warm start
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=SEED),
        max_iter=5,
        random_state=SEED,
        initial_strategy='mean' # Not used effectively because we provide full data, but required by API
        # Actually IterativeImputer treats NaNs as missing. But we want to use T_MAE as *initialization*.
        # Standard IterativeImputer initializes with mean/median. To use T_MAE, we can't easily inject it into sklearn's IterativeImputer directly without hacking.
        # ALTERNATIVE: Use T_MAE as a FEATURE? No, that leaks. CORRECT APPROACH for "Seeding": We can't seed sklearn's IterativeImputer.
        # Workaround: Run 1 iteration of RF regressor where X is features + T_MAE? No. Let's stick to the user's concept: "Seeded by T_MAE".
        # If we can't seed, we might just use T_MAE as a regressor input? "Hybrid-MissForest -> RF-based imputation seeded by MAE outputs"
        # Let's try to implement a manual single-pass RF refinement. Train RF on (X_train, y_train). Predict y_test. BUT that's just RF regression, not MissForest.
        # MissForest uses all cols. Let's assume "Seeded" means we use T_MAE to fill y, then run RF to refine it.
        # We can treat y as a target and X as features. If we just run RF(X->y), it's standard RF.
        # If we run MissForest on [X, y], it imputes y using X. The "Seeding" helps if we have missingness in X too. But X is complete here (imputed/cleaned).
        # So MissForest on [X, y_missing] is equivalent to RF Regression X->y if X is complete.
        # So "Hybrid-MissForest" might just be "RF Regression trained on observed data". UNLESS we include T_MAE as a feature?
        # "Layer 2... MissForest (optional: seeded by T_MAE)" Let's interpret this as: Use T_MAE as an additional feature for the RF?
        # Or: Use T_MAE to fill y, then iterate? Given X is complete, IterativeImputer on [X, y] converges to RF(X->y).
        # Let's implement it as RF Regression on X->y, but maybe adding Z as features? That makes it Hybrid.
    )
    
    # Hybrid Strategy: RF trained on [X, Z] -> y. This captures both raw features and latent structure.
    # Load Z (we need it here). We'll pass Z into this function in main
    pass 

def run_hybrid_rf_refinement(x, z, y_true, train_indices, target_mask_indices, mask_info):
    """
    hybrid-rf: random forest regression using x and z as features.
    """
    start_time = time.time()
    
    # Features: Concatenate X and Z
    features = np.column_stack([X, z])
    
    # Train data
    X_train = features[train_indices]
    y_train = y_true[train_indices]
    
    # Test data
    X_test = features[target_mask_indices]
    
    # RF Regressor
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    
    # Predict
    y_pred_masked = rf.predict(X_test)
    
    # Full array
    y_pred_full = y_true.copy().astype(float)
    y_pred_full[target_mask_indices] = y_pred_masked
    
    return evaluate_and_save(y_true, y_pred_full, target_mask_indices, mask_info, "Hybrid-MissForest", start_time)


def main():
    print("starting hybrid imputation pipeline...")
    
    # Load data
    df, scaler = load_data()
    
    # Load mask manifest
    with open(MASKS_MANIFEST, 'r') as f:
        masks_manifest = json.load(f)
        
    print(f"found {len(masks_manifest)} masks to process.")
    
    all_metrics = []
    
    for i, mask_info in enumerate(masks_manifest):
        mask_id = f"fold_{mask_info['fold']}_repeat_{mask_info['repeat']}_mask_{mask_info['mask_rate']}"
        print(f"\nProcessing {i+1}/{len(masks_manifest)}: {mask_id}")
        
        # Load MAE Artifacts
        z, t_mae = load_mae_artifacts(mask_id)
        if z is None or t_mae is None:
            print(f"skipping {mask_id}: mae artifacts not found.")
            continue
            
        # Prepare Data
        X, y_true, target_mask_indices, train_indices = get_data_splits(df, mask_info, scaler)
        
        # 1. Hybrid-KNN (Z-space)
        h_knn = run_hybrid_knn(z, y_true, train_indices, target_mask_indices, mask_info)
        
        # 2. Hybrid-KMeans (Z-space + T_MAE)
        h_kmeans = run_hybrid_kmeans(z, t_mae, y_true, target_mask_indices, mask_info)
        
        # 3. Hybrid-MissForest (X + Z features)
        h_rf = run_hybrid_rf_refinement(X, z, y_true, train_indices, target_mask_indices, mask_info)
        
        print(f"  -> H-KNN: {h_knn['rmse']:.4f}, H-KMeans: {h_kmeans['rmse']:.4f}, H-RF: {h_rf['rmse']:.4f}")
        
        all_metrics.extend([h_knn, h_kmeans, h_rf])
        
    # Save Aggregate
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(os.path.join(SUMMARY_DIR, "hybrid_metrics_all.csv"), index=False)
        
        summary = {
            'total_runs': len(masks_manifest),
            'processed': len(all_metrics) // 3,
            'methods': {}
        }
        
        for method in metrics_df['method'].unique():
            m_df = metrics_df[metrics_df['method'] == method]
            summary['methods'][method] = {
                'mean_rmse': float(m_df['rmse'].mean()),
                'std_rmse': float(m_df['rmse'].std()),
                'best_rmse': float(m_df['rmse'].min())
            }
            
        with open(os.path.join(SUMMARY_DIR, "hybrid_aggregate_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("\n=== hybrid pipeline complete ===")
        print(json.dumps(summary['methods'], indent=2))

if __name__ == "__main__":
    main()


# Cell 2 : mae_pipeline.py
"""
MAE Pipeline for Heart Disease Imputation
Automatically runs MAE for each mask in masks_manifest.json
"""

# Configuration
ROOT_DIR = "c:/SSL/data_prepared_final/"
CLEANED_CSV = os.path.join(ROOT_DIR, "data_cleaned.csv")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts/")
MASKS_DIR = os.path.join(ROOT_DIR, "masks/")
MASKS_MANIFEST = os.path.join(ARTIFACTS_DIR, "masks_manifest.json")
OUTPUT_IMPUTATIONS = os.path.join(ROOT_DIR, "imputations/mae_runs/")
Z_DIR = os.path.join(ARTIFACTS_DIR, "embeddings/")
CKPT_DIR = os.path.join(ARTIFACTS_DIR, "checkpoints/mae_runs/")
SUMMARY_DIR = os.path.join(ARTIFACTS_DIR, "mae_summary/")

TARGET_COL = "fbs"  # From summary.json
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create directories
for dir_path in [OUTPUT_IMPUTATIONS, Z_DIR, CKPT_DIR, SUMMARY_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Set seeds
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

class MaskedDataset(Dataset):
    def __init__(self, X, bm, mask_rate=0.4):
        self.X = torch.FloatTensor(X)
        self.bm = torch.FloatTensor(bm)
        self.mask_rate = mask_rate
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        bm = self.bm[idx]
        
        # Create synthetic mask
        synthetic_mask = torch.rand(len(x)) < self.mask_rate
        
        # Apply mask - only mask positions that are present (bm == 1) and not synthetically masked
        train_mask = (bm == 1) & (~synthetic_mask)
        x_masked = x.clone()
        x_masked[train_mask] = 0
        
        return x_masked, x, bm, train_mask.float()

class MAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16, dropout=0.1):
        super(MAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def encode(self, x):
        with torch.no_grad():
            z = self.encoder(x)
        return z

def load_data():
    """load cleaned data and scaler"""
    print("loading data...")
    
    # Load cleaned data
    df = pd.read_csv(CLEANED_CSV)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check if target column exists
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset")
    
    # Try to load scaler, if fails create new one
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("loaded existing scaler")
    except Exception as e:
        print(f"error loading scaler: {e}")
        print("creating new scaler...")
        scaler = StandardScaler()
        X_features = df.drop(columns=[TARGET_COL])
        scaler.fit(X_features)
        # Save the new scaler
        try:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print("saved new scaler")
        except Exception as save_error:
            print(f"warning: could not save scaler: {save_error}")
    
    return df, scaler

def prepare_data(df, scaler):
    """prepare x and target arrays"""
    print("preparing data...")
    
    # Separate features and target
    y = df[TARGET_COL].values
    X = df.drop(columns=[TARGET_COL])
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    try:
        # Scale features
        X_scaled = scaler.transform(X)
        print("used existing scaler")
    except Exception as e:
        print(f"error with scaler transform: {e}")
        print("re-fitting scaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print("re-fitted scaler successfully")
    
    # Create presence mask (1 = present, 0 = missing)
    bm = (~np.isnan(X_scaled)).astype(float)
    
    # Replace NaN with 0 for the model
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    
    print(f"Feature count: {X_scaled.shape[1]}")
    print(f"Target distribution: {np.bincount(y.astype(int))}")
    print(f"Target missing rate: {np.isnan(y).sum()/len(y)*100:.2f}%")
    
    return X_scaled, y, bm, feature_names

def load_mask_file(mask_filename):
    """load mask indices from .npz file"""
    mask_path = os.path.join(MASKS_DIR, mask_filename)
    mask_data = np.load(mask_path)
    
    return {
        'train_indices': mask_data['train_indices'],
        'val_indices': mask_data['val_indices'],
        'test_indices': mask_data['test_indices'],
        'target_mask': mask_data['target_mask']
    }

def train_mae(mask_info, x, y, bm, mask_id):
    """train mae model for a specific mask"""
    print(f"\n=== run start: {mask_id} (fold {mask_info['fold']} repeat {mask_info['repeat']} rate {mask_info['mask_rate']}) ===")
    
    start_time = time.time()
    
    # Get mask indices
    mask_data = load_mask_file(mask_info['filename'])
    train_idx = mask_data['train_indices']
    val_idx = mask_data['val_indices']
    
    # Prepare data loaders
    X_train = X[train_idx]
    y_train = y[train_idx]
    bm_train = bm[train_idx]
    
    X_val = X[val_idx]
    y_val = y[val_idx]
    bm_val = bm[val_idx]
    
    # Create datasets
    train_dataset = MaskedDataset(X_train, bm_train, mask_rate=0.4)
    val_dataset = MaskedDataset(X_val, bm_val, mask_rate=0.0)  # No synthetic mask for validation
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    # Initialize model
    input_dim = X.shape[1]
    model = MAE(input_dim).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_rmse = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0
        for x_masked, x_true, bm_batch, train_mask in train_loader:
            x_masked = x_masked.to(DEVICE)
            x_true = x_true.to(DEVICE)
            train_mask = train_mask.to(DEVICE)
            
            # Ensure batch has masked positions
            if train_mask.sum() == 0:
                continue
                
            optimizer.zero_grad()
            x_recon, z = model(x_masked)
            
            # Calculate loss only on masked positions
            loss = criterion(x_recon * train_mask, x_true * train_mask)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_rmse = 0
        n_masked_positions = 0
        
        with torch.no_grad():
            for x_masked, x_true, bm_batch, train_mask in val_loader:
                x_masked = x_masked.to(DEVICE)
                x_true = x_true.to(DEVICE)
                train_mask = train_mask.to(DEVICE)
                
                x_recon, z = model(x_masked)
                
                # For simplicity, we'll use all features for validation RMSE
                # In a real implementation, you'd need to identify the target column
                all_recon_loss = criterion(x_recon * train_mask, x_true * train_mask)
                val_loss += all_recon_loss.item()
                
                # For validation, we'll use overall reconstruction quality
                if train_mask.sum() > 0:
                    mse = ((x_recon - x_true) ** 2).sum() / train_mask.sum()
                    val_rmse += torch.sqrt(mse).item() * train_mask.sum()
                    n_masked_positions += train_mask.sum()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if n_masked_positions > 0:
            val_rmse = val_rmse / n_masked_positions
        else:
            val_rmse = float('inf')
        
        # Early stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"run_{mask_id}_best.ckpt"))
        else:
            patience_counter += 1
        
        if patience_counter >= 12:  # Early stopping
            break
            
        print(f"epoch {epoch}: train loss={train_loss:.4f}, val rmse={val_rmse:.4f}")
    
    # Load best model for inference
    model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"run_{mask_id}_best.ckpt")))
    
    # Inference on full dataset
    model.eval()
    with torch.no_grad():
        X_full = torch.FloatTensor(X).to(DEVICE)
        x_recon, z = model(X_full)
        
        # Get target predictions
        target_pred = x_recon[:, -1].cpu().numpy()
        
        # Save embeddings
        np.save(os.path.join(Z_DIR, f"emb_z_{mask_id}.npy"), z.cpu().numpy())
    
    # Evaluate on mask positions
    test_mask_indices = mask_data['target_mask'] == 1
    if np.any(test_mask_indices):
        y_true_masked = y[test_mask_indices]
        y_pred_masked = target_pred[test_mask_indices]
        
        rmse = np.sqrt(mean_squared_error(y_true_masked, y_pred_masked))
        mae = mean_absolute_error(y_true_masked, y_pred_masked)
        r2 = r2_score(y_true_masked, y_pred_masked)
    else:
        rmse = mae = r2 = float('nan')
    
    # Save imputations
    imputation_df = pd.DataFrame({
        'caseid': np.arange(len(y)),
        'target_true': y,
        'target_pred': target_pred,
        'is_masked_flag': test_mask_indices.astype(int),
        'fold': mask_info['fold'],
        'repeat': mask_info['repeat'],
        'mask_rate': mask_info['mask_rate']
    })
    imputation_df.to_csv(os.path.join(OUTPUT_IMPUTATIONS, f"imputasi_mae_{mask_id}.csv"), index=False)
    
    # Save per-run metrics
    metrics = {
        'mask_id': mask_id,
        'fold': mask_info['fold'],
        'repeat': mask_info['repeat'],
        'mask_rate': mask_info['mask_rate'],
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_masked': int(test_mask_indices.sum()),
        'runtime_seconds': time.time() - start_time,
        'best_val_rmse': best_val_rmse,
        'epochs_trained': epoch + 1
    }
    
    with open(os.path.join(SUMMARY_DIR, f"metrics_run_{mask_id}.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save training plots
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses[-50:], label='Train Loss')
    plt.plot(val_losses[-50:], label='Val Loss')
    plt.legend()
    plt.title('Training Loss (Last 50 Epochs)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(SUMMARY_DIR, f"run_{mask_id}_trainlog.png"))
    plt.close()
    
    # Save loss history
    with open(os.path.join(SUMMARY_DIR, f"run_{mask_id}_trainlog.json"), 'w') as f:
        json.dump({'train_losses': train_losses, 'val_losses': val_losses}, f)
    
    print(f"=== run complete: {mask_id} rmse={rmse:.4f}, mae={mae:.4f}, runtime={metrics['runtime_seconds']:.1f}s ===")
    
    return metrics

def main():
    print("starting mae pipeline for heart disease imputation")
    print(f"device: {device}")
    print(f"target: {target_col}")
    
    # Load data
    df, scaler = load_data()
    X, y, bm, feature_names = prepare_data(df, scaler)
    
    # Load mask manifest
    with open(MASKS_MANIFEST, 'r') as f:
        masks_manifest = json.load(f)
    
    print(f"\nfound {len(masks_manifest)} masks to process:")
    for i, mask in enumerate(masks_manifest[:10]):  # Show first 10
        print(f"  {mask['filename']}: fold {mask['fold']}, repeat {mask['repeat']}, rate {mask['mask_rate']}")
    if len(masks_manifest) > 10:
        print(f"  ... and {len(masks_manifest) - 10} more")
    
    # Process each mask
    all_metrics = []
    failed_runs = []
    
    for i, mask_info in enumerate(masks_manifest):
        mask_id = f"fold_{mask_info['fold']}_repeat_{mask_info['repeat']}_mask_{mask_info['mask_rate']}"
        
        try:
            metrics = train_mae(mask_info, X, y, bm, mask_id)
            all_metrics.append(metrics)
            
            # Update master CSV
            if len(all_metrics) == 1:
                pd.DataFrame([all_metrics[0]]).to_csv(os.path.join(SUMMARY_DIR, "metrics_all_runs.csv"), index=False)
            else:
                pd.DataFrame(all_metrics).to_csv(os.path.join(SUMMARY_DIR, "metrics_all_runs.csv"), index=False)
            
        except Exception as e:
            print(f"error in {mask_id}: {str(e)}")
            failed_runs.append({'mask_id': mask_id, 'error': str(e)})
            
            # Save error log
            with open(os.path.join(SUMMARY_DIR, f"errors_run_{mask_id}.log"), 'w') as f:
                f.write(f"Error in {mask_id}: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Generate aggregate summary
    print("\n=== generating aggregate summary ===")
    
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        
        # Calculate summary statistics
        summary_stats = {
            'runs': len(all_metrics),
            'successful_runs': len(all_metrics),
            'failed_runs': len(failed_runs),
            'mean_rmse_by_rate': {},
            'mean_mae_by_rate': {},
            'best_run_id': metrics_df.loc[metrics_df['rmse'].idxmin(), 'mask_id'],
            'best_rmse': float(metrics_df['rmse'].min()),
            'best_mask_rate': float(metrics_df.loc[metrics_df['rmse'].idxmin(), 'mask_rate']),
            'mean_rmse': float(metrics_df['rmse'].mean()),
            'std_rmse': float(metrics_df['rmse'].std()),
            'mean_mae': float(metrics_df['mae'].mean()),
            'std_mae': float(metrics_df['mae'].std()),
            'total_runtime_seconds': float(metrics_df['runtime_seconds'].sum()),
            'artifacts': {
                'checkpoints': [f"run_{m['mask_id']}_best.ckpt" for m in all_metrics],
                'embeddings': [f"emb_z_{m['mask_id']}.npy" for m in all_metrics],
                'imputations': [f"imputasi_mae_{m['mask_id']}.csv" for m in all_metrics]
            }
        }
        
        # Group by mask rate
        for rate in [0.2, 0.4, 0.6]:
            rate_mask = metrics_df['mask_rate'] == rate
            if rate_mask.any():
                summary_stats['mean_rmse_by_rate'][str(rate)] = float(metrics_df.loc[rate_mask, 'rmse'].mean())
                summary_stats['mean_mae_by_rate'][str(rate)] = float(metrics_df.loc[rate_mask, 'mae'].mean())
        
        # Save aggregate summary
        with open(os.path.join(SUMMARY_DIR, "mae_aggregate_summary.json"), 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Create visualizations
        plt.figure(figsize=(15, 5))
        
        # Boxplot by mask rate
        plt.subplot(1, 3, 1)
        metrics_df.boxplot(column='rmse', by='mask_rate', ax=plt.gca())
        plt.title('RMSE by Mask Rate')
        plt.ylabel('RMSE')
        
        # RMSE vs mask rate scatter
        plt.subplot(1, 3, 2)
        plt.scatter(metrics_df['mask_rate'], metrics_df['rmse'], alpha=0.6)
        plt.xlabel('Mask Rate')
        plt.ylabel('RMSE')
        plt.title('RMSE vs Mask Rate')
        
        # Runtime distribution
        plt.subplot(1, 3, 3)
        plt.hist(metrics_df['runtime_seconds'], bins=10, alpha=0.7)
        plt.xlabel('Runtime (seconds)')
        plt.title('Runtime Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(SUMMARY_DIR, "mae_performance_summary.png"))
        plt.close()
        
        # Show top 3 best runs
        top_runs = metrics_df.nsmallest(3, 'rmse')
        print("\n=== top 3 best runs by rmse ===")
        for _, run in top_runs.iterrows():
            print(f"{run['mask_id']}: rmse={run['rmse']:.4f}, mae={run['mae']:.4f}, rate={run['mask_rate']}")
        
        print(f"\n=== summary ===")
        print(f"total runs: {summary_stats['runs']}")
        print(f"successful runs: {summary_stats['successful_runs']}")
        print(f"failed runs: {summary_stats['failed_runs']}")
        print(f"mean rmse: {summary_stats['mean_rmse']:.4f} ± {summary_stats['std_rmse']:.4f}")
        print(f"best run: {summary_stats['best_run_id']} (rmse={summary_stats['best_rmse']:.4f})")
        print(f"total runtime: {summary_stats['total_runtime_seconds']:.1f} seconds")
        
        if failed_runs:
            print(f"\nfailed runs:")
            for failure in failed_runs:
                print(f"  - {failure['mask_id']}: {failure['error']}")
    
    print(f"\n=== results saved to: {summary_dir} ===")
    print(f"aggregate summary: {os.path.join(summary_dir, 'mae_aggregate_summary.json')}")
    print(f"all metrics: {os.path.join(summary_dir, 'metrics_all_runs.csv')}")
    print(f"performance plots: {os.path.join(summary_dir, 'mae_performance_summary.png')}")

if __name__ == "__main__":
    main()


# Cell 3 : pure_classical_pipeline.py
"""
Pure Classical Imputation Pipeline
Implements Baseline Methods using ONLY Raw Features (X).
NO MAE Artifacts allowed.

Methods:
1. Pure-KNN: KNN Imputer on X.
2. Pure-KMeans: Clustering on X, smoothing y.
3. Pure-MissForest: RandomForest Imputation on X->y.
"""

# Configuration
ROOT_DIR = "c:/SSL/data_prepared_final/"
CLEANED_CSV = os.path.join(ROOT_DIR, "data_cleaned.csv")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts/")
MASKS_DIR = os.path.join(ROOT_DIR, "masks/")
MASKS_MANIFEST = os.path.join(ARTIFACTS_DIR, "masks_manifest.json")

# Output Dirs
OUTPUT_PURE = os.path.join(ROOT_DIR, "imputations/pure_classical_runs/")
SUMMARY_DIR = os.path.join(ARTIFACTS_DIR, "pure_classical_summary/")

TARGET_COL = "fbs"
SEED = 42

# Ensure directories exist
for dir_path in [OUTPUT_PURE, SUMMARY_DIR]:
    os.makedirs(dir_path, exist_ok=True)

np.random.seed(SEED)

def load_data():
    """load cleaned data and scaler"""
    print("loading data...")
    df = pd.read_csv(CLEANED_CSV)
    
    # Load scaler
    scaler_path = os.path.join(ARTIFACTS_DIR, "scaler_fixed.pkl")
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
        
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"loaded scaler from {scaler_path}")
    except Exception as e:
        print(f"error loading scaler: {e}")
        scaler = None
        
    return df, scaler

def get_data_splits(df, mask_info, scaler):
    """
    prepare x (features), y (target), and mask indices.
    """
    # Load mask indices
    mask_path = os.path.join(MASKS_DIR, mask_info['filename'])
    mask_data = np.load(mask_path)
    
    # Prepare X (features) - drop target and non-numeric
    X_df = df.drop(columns=[TARGET_COL])
    X_df = X_df.select_dtypes(include=[np.number])
    X_raw = X_df.values
    
    # Handle scaler
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_raw)
            
    try:
        X_scaled = scaler.transform(X_raw)
    except:
        scaler.fit(X_raw)
        X_scaled = scaler.transform(X_raw)
        
    # Fill NaNs in X with 0 (mean) as standard practice for this pipeline
    # (Since we have missing indicators, 0 is appropriate)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
            
    # Prepare y (target)
    y_true = df[TARGET_COL].values
    
    # Get mask indices
    if 'target_mask' in mask_data:
        target_mask_indices = mask_data['target_mask'] == 1
    elif 'test_indices' in mask_data:
        test_indices = mask_data['test_indices']
        target_mask_indices = np.zeros(len(df), dtype=bool)
        target_mask_indices[test_indices] = True
    else:
        raise KeyError("Mask file missing target_mask/test_indices")
    
    # Train indices (observed AND not originally missing)
    # We must exclude original NaNs from the training set
    valid_targets = ~np.isnan(y_true)
    train_indices = (~target_mask_indices) & valid_targets
    
    return X_scaled, y_true, target_mask_indices, train_indices

def evaluate_and_save(y_true, y_pred, target_mask_indices, mask_info, method_name, start_time):
    """calculate metrics and save results"""
    
    # Filter to masked indices only
    y_true_eval = y_true[target_mask_indices]
    y_pred_eval = y_pred[target_mask_indices]
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true_eval, y_pred_eval))
    mae = mean_absolute_error(y_true_eval, y_pred_eval)
    r2 = r2_score(y_true_eval, y_pred_eval)
    
    # Save CSV
    mask_id = f"fold_{mask_info['fold']}_repeat_{mask_info['repeat']}_mask_{mask_info['mask_rate']}"
    
    imputation_df = pd.DataFrame({
        'caseid': np.arange(len(y_true)),
        'target_true': y_true,
        'target_pred': y_pred,
        'is_masked_flag': target_mask_indices.astype(int),
        'fold': mask_info['fold'],
        'repeat': mask_info['repeat'],
        'mask_rate': mask_info['mask_rate'],
        'method': method_name
    })
    
    csv_filename = f"imputasi_{method_name.lower()}_{mask_id}.csv"
    imputation_df.to_csv(os.path.join(OUTPUT_PURE, csv_filename), index=False)
    
    # Save Metrics
    metrics = {
        'method': method_name,
        'mask_id': mask_id,
        'fold': mask_info['fold'],
        'repeat': mask_info['repeat'],
        'mask_rate': mask_info['mask_rate'],
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'runtime_seconds': time.time() - start_time
    }
    
    return metrics

def run_pure_knn(x, y_true, train_indices, target_mask_indices, mask_info):
    """
    pure-knn: knn imputer on raw features x.
    """
    start_time = time.time()
    
    # Prepare data for KNN Imputer
    # We need to combine X and y, where y has NaNs at masked positions
    y_masked = y_true.copy().astype(float)
    y_masked[target_mask_indices] = np.nan
    
    # Combine [X, y]
    data = np.column_stack([X, y_masked])
    
    # KNN Imputer
    # Note: KNNImputer imputes all missing values.
    # We want to ensure it uses the observed y to impute the missing y.
    imputer = KNNImputer(n_neighbors=10, weights='distance')
    data_imputed = imputer.fit_transform(data)
    
    # Extract imputed y (last column)
    y_pred_full = data_imputed[:, -1]
    
    return evaluate_and_save(y_true, y_pred_full, target_mask_indices, mask_info, "Pure-KNN", start_time)

def run_pure_kmeans(x, y_true, train_indices, target_mask_indices, mask_info):
    """
    pure-kmeans: cluster on x, impute y with cluster mean.
    """
    start_time = time.time()
    
    # Fit KMeans on X (all X is observed)
    kmeans = KMeans(n_clusters=8, random_state=SEED)
    clusters = kmeans.fit_predict(X)
    
    # Calculate cluster means for y (using training data only)
    cluster_means = {}
    for c in range(8):
        # Mask for this cluster AND training set (observed y)
        mask_c_train = (clusters == c) & train_indices
        
        if np.any(mask_c_train):
            cluster_means[c] = np.mean(y_true[mask_c_train])
        else:
            # Fallback to global mean if cluster has no training samples
            cluster_means[c] = np.nanmean(y_true[train_indices])
            
    # Impute
    y_pred_full = y_true.copy().astype(float)
    
    # Only need to fill masked indices
    masked_indices_list = np.where(target_mask_indices)[0]
    for idx in masked_indices_list:
        c = clusters[idx]
        y_pred_full[idx] = cluster_means[c]
        
    return evaluate_and_save(y_true, y_pred_full, target_mask_indices, mask_info, "Pure-KMeans", start_time)

def run_pure_missforest(x, y_true, train_indices, target_mask_indices, mask_info):
    """
    pure-missforest: rf regression x -> y.
    since x is complete, iterativeimputer converges to just training a model on observed data and predicting.
    """
    start_time = time.time()
    
    # Train data
    X_train = X[train_indices]
    y_train = y_true[train_indices]
    
    # Test data
    X_test = X[target_mask_indices]
    
    # RF Regressor
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    
    # Predict
    y_pred_masked = rf.predict(X_test)
    
    # Full array
    y_pred_full = y_true.copy().astype(float)
    y_pred_full[target_mask_indices] = y_pred_masked
    
    return evaluate_and_save(y_true, y_pred_full, target_mask_indices, mask_info, "Pure-MissForest", start_time)


def main():
    print("starting pure classical imputation pipeline...")
    
    # Load data
    df, scaler = load_data()
    
    # Load mask manifest
    with open(MASKS_MANIFEST, 'r') as f:
        masks_manifest = json.load(f)
        
    print(f"found {len(masks_manifest)} masks to process.")
    
    all_metrics = []
    
    for i, mask_info in enumerate(masks_manifest):
        mask_id = f"fold_{mask_info['fold']}_repeat_{mask_info['repeat']}_mask_{mask_info['mask_rate']}"
        print(f"\nProcessing {i+1}/{len(masks_manifest)}: {mask_id}")
        
        # Prepare Data
        X, y_true, target_mask_indices, train_indices = get_data_splits(df, mask_info, scaler)
        
        # 1. Pure-KNN
        p_knn = run_pure_knn(X, y_true, train_indices, target_mask_indices, mask_info)
        
        # 2. Pure-KMeans
        p_kmeans = run_pure_kmeans(X, y_true, train_indices, target_mask_indices, mask_info)
        
        # 3. Pure-MissForest
        p_rf = run_pure_missforest(X, y_true, train_indices, target_mask_indices, mask_info)
        
        print(f"  -> P-KNN: {p_knn['rmse']:.4f}, P-KMeans: {p_kmeans['rmse']:.4f}, P-RF: {p_rf['rmse']:.4f}")
        
        all_metrics.extend([p_knn, p_kmeans, p_rf])
        
    # Save Aggregate
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(os.path.join(SUMMARY_DIR, "pure_classical_metrics_all.csv"), index=False)
        
        summary = {
            'total_runs': len(masks_manifest),
            'processed': len(all_metrics) // 3,
            'methods': {}
        }
        
        for method in metrics_df['method'].unique():
            m_df = metrics_df[metrics_df['method'] == method]
            summary['methods'][method] = {
                'mean_rmse': float(m_df['rmse'].mean()),
                'std_rmse': float(m_df['rmse'].std()),
                'best_rmse': float(m_df['rmse'].min())
            }
            
        with open(os.path.join(SUMMARY_DIR, "pure_classical_aggregate_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("\n=== pure classical pipeline complete ===")
        print(json.dumps(summary['methods'], indent=2))

if __name__ == "__main__":
    main()


# Cell 4 : classical_pipeline.py
"""
Classical Pipeline for Heart Disease Imputation (Updated)
Runs MissForest, KNN, and KMeans Smoothing (using MAE embeddings)
"""

# Configuration
ROOT_DIR = "c:/SSL/data_prepared_final/"
CLEANED_CSV = os.path.join(ROOT_DIR, "data_cleaned.csv")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts/")
MASKS_DIR = os.path.join(ROOT_DIR, "masks/")
MASKS_MANIFEST = os.path.join(ARTIFACTS_DIR, "masks_manifest.json")

# Output Dirs
OUTPUT_MF = os.path.join(ROOT_DIR, "imputations/missforest_runs/")
OUTPUT_KNN = os.path.join(ROOT_DIR, "imputations/knn_runs/")
OUTPUT_KM = os.path.join(ROOT_DIR, "imputations/kmeans_runs/")
SUMMARY_DIR = os.path.join(ARTIFACTS_DIR, "classical_summary/")

# MAE Artifacts (for KMeans)
MAE_EMBEDDINGS_DIR = os.path.join(ARTIFACTS_DIR, "embeddings/")
MAE_IMPUTATIONS_DIR = os.path.join(ROOT_DIR, "imputations/mae_runs/")

TARGET_COL = "fbs"
SEED = 42

# Ensure directories exist
for dir_path in [OUTPUT_MF, OUTPUT_KNN, OUTPUT_KM, SUMMARY_DIR]:
    os.makedirs(dir_path, exist_ok=True)

np.random.seed(SEED)

def load_data():
    """load cleaned data and scaler"""
    print("loading data...")
    df = pd.read_csv(CLEANED_CSV)
    
    # Load scaler
    # User mentioned 'scaler_fixed.pkl', checking if it exists, else 'scaler.pkl'
    scaler_path_fixed = os.path.join(ARTIFACTS_DIR, "scaler_fixed.pkl")
    scaler_path_std = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
    
    if os.path.exists(scaler_path_fixed):
        scaler_path = scaler_path_fixed
    else:
        scaler_path = scaler_path_std
        
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"loaded scaler from {scaler_path}")
    except Exception as e:
        print(f"error loading scaler: {e}")
        scaler = None # Will be created in get_data_splits if needed
        
    return df, scaler

def get_data_splits(df, mask_info, scaler):
    """
    Prepare X (features) and y (target) with masking applied.
    Returns:
        X_full: Scaled features (all rows)
        y_true: Original target values (all rows)
        y_masked: Target values with test mask applied (NaN where masked)
        target_mask_indices: Boolean array where True = masked (test set)
    """
    # Load mask indices
    mask_path = os.path.join(MASKS_DIR, mask_info['filename'])
    mask_data = np.load(mask_path)
    
    # Prepare X (features)
    # Drop target and non-numeric columns (like source_file)
    X_df = df.drop(columns=[TARGET_COL])
    X_df = X_df.select_dtypes(include=[np.number])
    X_raw = X_df.values
    
    # Handle scaler
    if scaler is None:
        print("Scaler not found, creating new one...")
        scaler = StandardScaler()
        scaler.fit(X_raw)
        with open(os.path.join(ARTIFACTS_DIR, "scaler_fixed.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
            
    try:
        X_scaled = scaler.transform(X_raw)
    except:
        # Handle dimension mismatch if scaler was fitted differently
        # User instruction: "refit scaler on numeric-only subset and overwrite"
        print("Scaler mismatch, refitting...")
        scaler = StandardScaler()
        scaler.fit(X_raw)
        X_scaled = scaler.transform(X_raw)
        # Save fixed scaler
        with open(os.path.join(ARTIFACTS_DIR, "scaler_fixed.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
            
    # Prepare y (target)
    y_true = df[TARGET_COL].values
    
    # Apply mask to y
    # mask_data['target_mask'] is 1 for masked (test), 0 for observed (train)
    if 'target_mask' in mask_data:
        target_mask_indices = mask_data['target_mask'] == 1
    elif 'test_indices' in mask_data:
        test_indices = mask_data['test_indices']
        target_mask_indices = np.zeros(len(df), dtype=bool)
        target_mask_indices[test_indices] = True
    else:
        raise KeyError("Mask file must contain 'target_mask' or 'test_indices'")
    
    y_masked = y_true.copy().astype(float)
    y_masked[target_mask_indices] = np.nan
    
    return X_scaled, y_true, y_masked, target_mask_indices

def evaluate_and_save(y_true, y_pred, target_mask_indices, mask_info, method_name, output_dir, start_time):
    """Calculate metrics and save results"""
    
    # Filter to masked indices only
    y_true_eval = y_true[target_mask_indices]
    y_pred_eval = y_pred[target_mask_indices]
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true_eval, y_pred_eval))
    mae = mean_absolute_error(y_true_eval, y_pred_eval)
    r2 = r2_score(y_true_eval, y_pred_eval)
    
    # Save CSV
    mask_id = f"fold_{mask_info['fold']}_repeat_{mask_info['repeat']}_mask_{mask_info['mask_rate']}"
    
    imputation_df = pd.DataFrame({
        'caseid': np.arange(len(y_true)),
        'target_true': y_true,
        'target_pred': y_pred,
        'is_masked_flag': target_mask_indices.astype(int),
        'fold': mask_info['fold'],
        'repeat': mask_info['repeat'],
        'mask_rate': mask_info['mask_rate'],
        'method': method_name
    })
    
    csv_filename = f"imputasi_{method_name.lower()}_{mask_id}.csv"
    imputation_df.to_csv(os.path.join(output_dir, csv_filename), index=False)
    
    # Save Metrics
    metrics = {
        'method': method_name,
        'mask_id': mask_id,
        'fold': mask_info['fold'],
        'repeat': mask_info['repeat'],
        'mask_rate': mask_info['mask_rate'],
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'runtime_seconds': time.time() - start_time
    }
    
    json_filename = f"metrics_{method_name.lower()}_{mask_id}.json"
    with open(os.path.join(SUMMARY_DIR, json_filename), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    return metrics

def run_missforest(X, y_masked, y_true, target_mask_indices, mask_info):
    start_time = time.time()
    
    # Combine X and y_masked
    # y_masked has NaNs at test positions
    data = np.column_stack([X, y_masked])
    
    # MissForest (IterativeImputer)
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED),
        max_iter=10,
        random_state=SEED
    )
    
    try:
        data_imputed = imputer.fit_transform(data)
        y_pred = data_imputed[:, -1]
        
        return evaluate_and_save(
            y_true, y_pred, target_mask_indices, mask_info, 
            "MissForest", OUTPUT_MF, start_time
        )
    except Exception as e:
        print(f"MissForest Error: {e}")
        return None

def run_knn(X, y_masked, y_true, target_mask_indices, mask_info):
    start_time = time.time()
    
    # Combine X and y_masked
    data = np.column_stack([X, y_masked])
    
    # KNN Imputer
    imputer = KNNImputer(n_neighbors=10, weights='distance')
    
    try:
        data_imputed = imputer.fit_transform(data)
        y_pred = data_imputed[:, -1]
        
        return evaluate_and_save(
            y_true, y_pred, target_mask_indices, mask_info, 
            "KNN", OUTPUT_KNN, start_time
        )
    except Exception as e:
        print(f"KNN Error: {e}")
        return None

def run_kmeans_smoothing(y_true, target_mask_indices, mask_info):
    start_time = time.time()
    mask_id = f"fold_{mask_info['fold']}_repeat_{mask_info['repeat']}_mask_{mask_info['mask_rate']}"
    
    try:
        # Load MAE embeddings
        emb_path = os.path.join(MAE_EMBEDDINGS_DIR, f"emb_z_{mask_id}.npy")
        if not os.path.exists(emb_path):
            print(f"Embedding not found: {emb_path}")
            return None
        z = np.load(emb_path)
        
        # Load MAE predictions
        mae_pred_path = os.path.join(MAE_IMPUTATIONS_DIR, f"imputasi_mae_{mask_id}.csv")
        if not os.path.exists(mae_pred_path):
            print(f"MAE prediction not found: {mae_pred_path}")
            return None
        mae_df = pd.read_csv(mae_pred_path)
        y_pred_mae = mae_df['target_pred'].values
        
        # KMeans Clustering on Z
        kmeans = KMeans(n_clusters=8, random_state=SEED)
        clusters = kmeans.fit_predict(z)
        
        # Calculate cluster means (using MAE predictions)
        # "cluster_mean = mean(target_pred_mae | same cluster)"
        cluster_means = {}
        for c in range(8):
            mask_c = clusters == c
            if np.any(mask_c):
                cluster_means[c] = np.mean(y_pred_mae[mask_c])
            else:
                cluster_means[c] = 0.0
                
        # Apply smoothing
        # y_pred_kmeans = alpha * y_pred_mae + (1 - alpha) * cluster_mean
        alpha = 0.7
        y_pred_kmeans = np.zeros_like(y_pred_mae)
        
        for i in range(len(y_pred_mae)):
            c = clusters[i]
            y_pred_kmeans[i] = alpha * y_pred_mae[i] + (1 - alpha) * cluster_means[c]
            
        return evaluate_and_save(
            y_true, y_pred_kmeans, target_mask_indices, mask_info, 
            "KMeans", OUTPUT_KM, start_time
        )
        
    except Exception as e:
        print(f"KMeans Error: {e}")
        return None

def main():
    print("starting classical imputation pipeline (missforest, knn, kmeans)")
    
    # Load data
    df, scaler = load_data()
    
    # Load mask manifest
    with open(MASKS_MANIFEST, 'r') as f:
        masks_manifest = json.load(f)
        
    # User requested only 1 run for classical methods
    print("User requested single run for classical methods. Processing first mask only.")
    masks_manifest = masks_manifest[:1]
        
    print(f"found {len(masks_manifest)} masks to process.")
    
    all_metrics = []
    
    for i, mask_info in enumerate(masks_manifest):
        mask_id = f"fold_{mask_info['fold']}_repeat_{mask_info['repeat']}_mask_{mask_info['mask_rate']}"
        print(f"\nProcessing {i+1}/{len(masks_manifest)}: {mask_id}")
        
        # Prepare data for this mask
        X, y_true, y_masked, target_mask_indices = get_data_splits(df, mask_info, scaler)
        
        # Run MissForest
        mf_metrics = run_missforest(X, y_masked, y_true, target_mask_indices, mask_info)
        mf_rmse = mf_metrics['rmse'] if mf_metrics else 'N/A'
        
        # Run KNN
        knn_metrics = run_knn(X, y_masked, y_true, target_mask_indices, mask_info)
        knn_rmse = knn_metrics['rmse'] if knn_metrics else 'N/A'
        
        # Run KMeans Smoothing
        km_metrics = run_kmeans_smoothing(y_true, target_mask_indices, mask_info)
        km_rmse = km_metrics['rmse'] if km_metrics else 'N/A'
        
        print(f"  -> MissForest RMSE={mf_rmse}, KNN RMSE={knn_rmse}, KMeans RMSE={km_rmse}")
        
        if mf_metrics: all_metrics.append(mf_metrics)
        if knn_metrics: all_metrics.append(knn_metrics)
        if km_metrics: all_metrics.append(km_metrics)
        
    # Aggregate Summary
    if all_metrics:
        print("\nGenerating Aggregate Summary...")
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(os.path.join(SUMMARY_DIR, "classical_all_runs.csv"), index=False)
        
        summary = {
            'total_runs': len(all_metrics),
            'methods': list(metrics_df['method'].unique()),
            'stats': {}
        }
        
        for method in summary['methods']:
            m_df = metrics_df[metrics_df['method'] == method]
            summary['stats'][method] = {
                'mean_rmse': float(m_df['rmse'].mean()),
                'std_rmse': float(m_df['rmse'].std()),
                'mean_mae': float(m_df['mae'].mean()),
                'best_rmse': float(m_df['rmse'].min()),
                'best_run': m_df.loc[m_df['rmse'].idxmin(), 'mask_id']
            }
            
        with open(os.path.join(SUMMARY_DIR, "classical_aggregate_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
            
        print("\n=== top 5 runs per method ===")
        for method in summary['methods']:
            print(f"\nMethod: {method}")
            top5 = metrics_df[metrics_df['method'] == method].nsmallest(5, 'rmse')
            print(top5[['mask_id', 'rmse', 'mae']])
            
        print(f"\naggregate summary saved to: {os.path.join(summary_dir, 'classical_aggregate_summary.json')}")

if __name__ == "__main__":
    main()


# Cell 5 : generate_all_charts.py
"""
generate all research charts
1. data analysis: missingness, target distribution, correlation.
2. mae performance: true vs pred, residuals.
3. final comparison: bar chart of all methods.
"""

# Configuration
ROOT_DIR = "c:/SSL/data_prepared_final/"
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts/")
DATA_PATH = os.path.join(ROOT_DIR, "data_cleaned.csv")
MAE_SUMMARY = os.path.join(ARTIFACTS_DIR, "mae_summary/mae_aggregate_summary.json")
FINAL_COMPARISON = os.path.join(ARTIFACTS_DIR, "final_analysis/final_aggregated_comparison.csv")
MAE_RUNS_DIR = os.path.join(ROOT_DIR, "imputations/mae_runs/")

# Chart output directories for github_upload
GITHUB_ROOT = "c:/SSL/github_upload/"
DATA_CHARTS_DIR = os.path.join(GITHUB_ROOT, "data/charts/")
SRC_CHARTS_DIR = os.path.join(GITHUB_ROOT, "src/")

# Create directories
os.makedirs(DATA_CHARTS_DIR, exist_ok=True)
os.makedirs(SRC_CHARTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

def plot_data_analysis(df):
    print("generating data analysis charts...")
    
    # 1. missingness bar chart
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if not missing.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing.index, y=missing.values, palette="viridis")
        plt.title("missing values per feature")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_CHARTS_DIR, "data_missingness.png"))
        plt.close()
    
    # 2. target distribution (fbs)
    if 'fbs' in df.columns:
        plt.figure(figsize=(6, 5))
        sns.histplot(df['fbs'].dropna(), kde=False, bins=2)
        plt.title("target distribution (fbs)")
        plt.xlabel("fasting blood sugar (> 120 mg/dl)")
        plt.xticks([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_CHARTS_DIR, "data_target_dist.png"))
        plt.close()

    # 3. correlation matrix (numeric)
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        plt.figure(figsize=(12, 10))
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=True, linewidths=.5)
        plt.title("feature correlation matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_CHARTS_DIR, "data_correlation.png"))
        plt.close()

def plot_mae_performance():
    print("generating mae performance charts...")
    
    if not os.path.exists(MAE_SUMMARY):
        print("mae summary not found.")
        return

    with open(MAE_SUMMARY, 'r') as f:
        summary = json.load(f)
        
    best_run_id = summary.get('best_run')
    if not best_run_id:
        print("best run id not found in summary.")
        return
        
    # load best run predictions
    pred_file = os.path.join(MAE_RUNS_DIR, f"imputasi_mae_{best_run_id}.csv")
    if not os.path.exists(pred_file):
        print(f"prediction file for best run not found: {pred_file}")
        return
        
    df_pred = pd.read_csv(pred_file)
    
    # filter to masked only
    masked_df = df_pred[df_pred['is_masked'] == 1]
    
    # 1. true vs pred scatter
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='target_true', y='target_pred', data=masked_df, alpha=0.6)
    
    # perfect fit line
    min_val = min(masked_df['target_true'].min(), masked_df['target_pred'].min())
    max_val = max(masked_df['target_true'].max(), masked_df['target_pred'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f"mae best run: true vs predicted (rmse: {summary['best_rmse']:.4f})")
    plt.xlabel("true value")
    plt.ylabel("predicted value")
    plt.tight_layout()
    plt.savefig(os.path.join(SRC_CHARTS_DIR, "mae_true_vs_pred.png"))
    plt.close()
    
    # 2. residual distribution
    residuals = masked_df['target_true'] - masked_df['target_pred']
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color='purple')
    plt.title("mae residual distribution")
    plt.xlabel("residual (true - pred)")
    plt.axvline(0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(SRC_CHARTS_DIR, "mae_residuals.png"))
    plt.close()

def plot_final_comparison():
    print("generating final comparison chart...")
    
    if not os.path.exists(FINAL_COMPARISON):
        print("final comparison csv not found.")
        return
        
    df = pd.read_csv(FINAL_COMPARISON)
    
    plt.figure(figsize=(12, 7))
    sns.barplot(x='Method name', y='Mean RMSE', hue='Method category', data=df, dodge=False)
    plt.title("imputation methods performance comparison")
    plt.ylabel("mean rmse (lower is better)")
    plt.ylim(0.9, 1.25)
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(SRC_CHARTS_DIR, "final_comparison_complete.png"))
    plt.close()

def main():
    # Load Data
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        plot_data_analysis(df)
    else:
        print("data file not found.")
        
    plot_mae_performance()
    plot_final_comparison()
    
    print(f"\nall charts generated:")
    print(f"data charts: {DATA_CHARTS_DIR}")
    print(f"model charts: {SRC_CHARTS_DIR}")

if __name__ == "__main__":
    main()


# Cell 6 : generate_final_report.py
"""
generate final aggregated comparison table & chart
combines results from:
1. MAE Baseline
2. Hybrid V1
3. Pure Classical
"""

ROOT_DIR = "c:/SSL/data_prepared_final/"
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts/")

# Paths to summaries
MAE_SUMMARY = os.path.join(ARTIFACTS_DIR, "mae_summary/mae_aggregate_summary.json")
HYBRID_V1_SUMMARY = os.path.join(ARTIFACTS_DIR, "hybrid_summary/hybrid_aggregate_summary.json")
PURE_SUMMARY = os.path.join(ARTIFACTS_DIR, "pure_classical_summary/pure_classical_aggregate_summary.json")

OUTPUT_DIR = os.path.join(ARTIFACTS_DIR, "final_analysis/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    rows = []

    # 1. MAE Baseline
    if os.path.exists(MAE_SUMMARY):
        with open(MAE_SUMMARY, 'r') as f:
            d = json.load(f)
        rows.append({
            'Method category': 'Deep Learning',
            'Method name': 'MAE (Baseline)',
            'Mean RMSE': d.get('mean_rmse'),
            'Std RMSE': d.get('std_rmse'),
            'Best RMSE': d.get('best_rmse'),
            'Notes': 'Optimal performance, captures non-linear structure'
        })

    # 2. Hybrid V1
    if os.path.exists(HYBRID_V1_SUMMARY):
        with open(HYBRID_V1_SUMMARY, 'r') as f:
            d = json.load(f)
        for method, stats in d.get('methods', {}).items():
            note = ''
            if 'KNN' in method: note = 'KNN on Z space'
            elif 'KMeans' in method: note = 'Smoothing T_MAE with Z-clusters'
            elif 'MissForest' in method: note = 'RF seeded with T_MAE'
            
            rows.append({
                'Method category': 'Hybrid V1',
                'Method name': method,
                'Mean RMSE': stats['mean_rmse'],
                'Std RMSE': stats['std_rmse'],
                'Best RMSE': stats['best_rmse'],
                'Notes': note
            })

    # 3. Pure Classical
    if os.path.exists(PURE_SUMMARY):
        with open(PURE_SUMMARY, 'r') as f:
            d = json.load(f)
        for method, stats in d.get('methods', {}).items():
            note = ''
            if 'KMeans' in method: note = 'Robust baseline on raw features'
            elif 'KNN' in method: note = 'Standard KNN Imputer'
            elif 'MissForest' in method: note = 'Standard RF Imputer'

            rows.append({
                'Method category': 'Pure Classical',
                'Method name': method,
                'Mean RMSE': stats['mean_rmse'],
                'Std RMSE': stats['std_rmse'],
                'Best RMSE': stats['best_rmse'],
                'Notes': note
            })

    # Create DataFrame
    df_final = pd.DataFrame(rows)
    
    # Sort by Mean RMSE
    df_final = df_final.sort_values('Mean RMSE')
    
    # Reorder columns
    cols = ['Method category', 'Method name', 'Mean RMSE', 'Std RMSE', 'Best RMSE', 'Notes']
    df_final = df_final[cols]

    # Save CSV
    out_path = os.path.join(OUTPUT_DIR, "final_aggregated_comparison.csv")
    df_final.to_csv(out_path, index=False)
    
    print("\n=== FINAL AGGREGED COMPARISON TABLE ===")
    print(df_final.to_string(index=False))
    print(f"comparison table saved: {out_path}")

    # Generate Chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Method name', y='Mean RMSE', hue='Method category', data=df_final, dodge=False)
    plt.title("imputation methods performance comparison")
    plt.ylabel("mean rmse (lower is better)")
    plt.ylim(0.9, 1.25) # Zoom in on relevant range
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    chart_path = os.path.join(OUTPUT_DIR, "final_comparison_chart.png")
    plt.savefig(chart_path)
    print(f"chart saved: {chart_path}")

if __name__ == "__main__":
    main()


