import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              RandomForestClassifier, ExtraTreesRegressor,
                              HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import r2_score
import re
import warnings
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

warnings.filterwarnings('ignore')

app = FastAPI(title="Task 3 - Part 3 (INCLUDING 2x and 7x Series)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================================
# 1. DATA PREPARATION & FEATURE ENGINEERING (UPDATED FOR NEW PROPERTIES)
# ======================================================================================
def load_and_prep_data(file_path):
    print(f"[{'='*20} 1. DATA LOADING & PREPARATION (INCLUDING 2x & 7x) {'='*20}]")
    try:
        df = pd.read_csv(file_path)
    except:
        df = pd.read_excel(file_path)

    df.columns = df.columns.str.strip()

    elements = ['Al', 'Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Ni', 'Zn', 'Ga', 'V', 'Ti']

    def parse_val(val):
        if pd.isna(val) or val == '': return 0.0
        s = str(val).strip()
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", s)]
        if len(nums) == 2: return sum(nums)/2
        if len(nums) == 1: return nums[0]
        return 0.0
        
    col_mapping = {
        'Yield Strength (MPa)': 'YS (MPa)', 
        'Thermal Expansion (Âµm/m-K)': 'TE Coeff', 
        'Thermal Expansion (µm/m-K)': 'TE Coeff',
        'Elec. Conductivity Vol (% IACS)': 'EC Volume (% IACS)', 
        'Thermal Conductivity (W/m-K)': 'TC (W/m-K)',
        'Fatigue Strength (MPa)': 'Fatigue Strength (MPa)'
    }
    df.rename(columns=col_mapping, inplace=True)

    objectives = ['YS (MPa)', 'UTS (MPa)', 'EC Volume (% IACS)', 'TC (W/m-K)', 'TE Coeff', 'Fatigue Strength (MPa)']

    cols_to_clean = elements + objectives
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(parse_val)

    for col in objectives:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    avail_obj = [o for o in objectives if o in df.columns]
    df = df[(df[avail_obj] > 0).any(axis=1)].fillna(0.0)

    print(f" > Data Loaded. Valid Alloys for CVAE: {len(df)}")
    return df, elements, avail_obj

# ======================================================================================
# 2. FORWARD MODELS (ALL 7 ADVANCED MODELS) & TEMPER CLASSIFIER
# ======================================================================================
class ForwardEnsemble:
    def __init__(self):
        self.models = {}
        self.accuracies = {}

    def train(self, X, df, objectives):
        print(f"\n[{'='*20} 2. TRAINING FORWARD MODELS (ALL 7 ADVANCED MODELS) {'='*20}]")
        for obj in objectives:
            y = df[obj]
            self.models[obj] = []

            rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
            rf.fit(X, y)
            self.models[obj].append(rf)

            gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            gb.fit(X, y)
            self.models[obj].append(gb)

            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, objective='reg:squarederror')
            xgb_model.fit(X, y)
            self.models[obj].append(xgb_model)

            et = ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42)
            et.fit(X, y)
            self.models[obj].append(et)

            hgb = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42)
            hgb.fit(X, y)
            self.models[obj].append(hgb)

            svr = Pipeline([('scaler', StandardScaler()), ('svr', SVR(C=10, gamma='scale'))])
            svr.fit(X, y)
            self.models[obj].append(svr)

            mlp = Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))])
            mlp.fit(X, y)
            self.models[obj].append(mlp)

            preds = self.predict(X, obj)[0]
            r2 = r2_score(y, preds)
            self.accuracies[obj] = r2
            print(f" > Trained 7 Advanced Models for {obj[:15]:<15} | Ensemble R2: {r2:.3f}")

    def predict(self, X, obj):
        all_preds = np.array([m.predict(X) for m in self.models[obj]])
        return np.mean(all_preds, axis=0), np.std(all_preds, axis=0)

def train_temper_model_v2(df, elements):
    print("\n > Training Temper Classifier (Model-2 Logic: 12 Elements + YS)...")
    if 'Temper' not in df.columns or 'YS (MPa)' not in df.columns: return None, None
    le = LabelEncoder()
    valid_idx = (df['Temper'].astype(str).str.strip() != '') & (df['YS (MPa)'] > 0)
    df_valid = df[valid_idx]

    if len(df_valid) == 0: return None, None

    X_valid = df_valid[elements + ['YS (MPa)']].values
    y_valid = le.fit_transform(df_valid['Temper'].astype(str))

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_valid, y_valid)
    return clf, le

# ======================================================================================
# 3. CVAE ARCHITECTURE (THE GENERATOR)
# ======================================================================================
class CVAE(nn.Module):
    def __init__(self, input_dim, cond_dim, latent_dim=8):
        super(CVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar, beta=0.8):
    MSE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + beta * KLD

# ======================================================================================
# 4. INVERSE DESIGN PIPELINE (DYNAMIC FILTERING BASED ON BATCH)
# ======================================================================================
def inverse_design_pipeline(cvae, forward_ensemble, scaler_x, scaler_y, objectives, temper_model, le_temper, active_batch_targets, elements, df_train):
    print(f"\n[{'='*20} 5. INVERSE DESIGN (GENERATING & FILTERING) {'='*20}]")

    full_target_wishlist = {}
    for obj in objectives:
        if obj in active_batch_targets:
            full_target_wishlist[obj] = active_batch_targets[obj]
        else:
            full_target_wishlist[obj] = df_train[obj].mean()

    target_df = pd.DataFrame([full_target_wishlist])[objectives]
    target_scaled = scaler_y.transform(target_df)
    target_cond_tensor = torch.FloatTensor(target_scaled)

    element_bounds = {i: {'max': df_train[el].max(), 'min': df_train[el].min()} for i, el in enumerate(elements)}

    POP_SIZE = 500
    GENERATIONS = 30
    LATENT_DIM = 8
    LAMBDA_PENALTY = 10.0

    z_pop = torch.randn(POP_SIZE, LATENT_DIM)
    c_cond = target_cond_tensor.repeat(POP_SIZE, 1)

    print(" > Generating new alloys and scoring ONLY against your active Batch Targets...")
    for gen in range(GENERATIONS):
        with torch.no_grad():
            recon_x = cvae.decoder(torch.cat([z_pop, c_cond], dim=1)).numpy()
        real_x = scaler_x.inverse_transform(recon_x)

        penalty_scores = np.abs(real_x.sum(axis=1) - 100.0)
        for i in range(len(elements)):
            penalty_scores += np.maximum(0, element_bounds[i]['min'] - real_x[:, i])
            penalty_scores += np.maximum(0, real_x[:, i] - element_bounds[i]['max'])

        real_x_corr = np.maximum(real_x, 0)
        sums_corr = real_x_corr.sum(axis=1, keepdims=True)
        sums_corr[sums_corr==0] = 1
        real_x_corr = (real_x_corr / sums_corr) * 100.0

        preds_list, unc_list = [], []
        for obj in objectives:
            mu, std = forward_ensemble.predict(real_x_corr, obj)
            preds_list.append(mu)
            unc_list.append(std)

        preds_arr = np.column_stack(preds_list)
        unc_arr = np.column_stack(unc_list)

        fitness = np.zeros(POP_SIZE)
        for obj, target_val in active_batch_targets.items():
            if obj in objectives:
                idx = objectives.index(obj)
                error = np.abs(preds_arr[:, idx] - target_val) / target_val
                fitness -= error

        risk = np.mean(unc_arr, axis=1) * 0.5
        fitness = fitness - risk - (LAMBDA_PENALTY * penalty_scores)

        top_k = POP_SIZE // 2
        top_indices = np.argsort(fitness)[-top_k:]
        best_z = z_pop[top_indices]

        new_z = [(best_z[np.random.randint(top_k)] + best_z[np.random.randint(top_k)])/2.0 + torch.randn(LATENT_DIM)*0.1 for _ in range(POP_SIZE)]
        z_pop = torch.stack(new_z)

    with torch.no_grad():
        final_x_raw = cvae.decoder(torch.cat([z_pop, c_cond], dim=1)).numpy()

    final_x = scaler_x.inverse_transform(final_x_raw)
    final_penalties = np.abs(final_x.sum(axis=1) - 100.0)
    for i in range(len(elements)):
        final_penalties += np.maximum(0, element_bounds[i]['min'] - final_x[:, i])
        final_penalties += np.maximum(0, final_x[:, i] - element_bounds[i]['max'])

    final_x = np.maximum(final_x, 0)
    final_x = (final_x / final_x.sum(axis=1, keepdims=True)) * 100.0

    final_preds_list, final_unc_list = [], []
    for obj in objectives:
        mu, std = forward_ensemble.predict(final_x, obj)
        final_preds_list.append(mu)
        final_unc_list.append(std)

    final_preds_arr = np.column_stack(final_preds_list)
    final_unc_arr = np.mean(np.column_stack(final_unc_list), axis=1)

    results_df = pd.DataFrame(final_x, columns=elements)
    results_df['Uncertainty'] = final_unc_arr
    results_df['Penalty_Score'] = final_penalties

    for i, obj in enumerate(objectives):
        results_df[obj] = final_preds_arr[:, i]

    if temper_model and 'YS (MPa)' in objectives:
        ys_index = objectives.index('YS (MPa)')
        predicted_ys = final_preds_arr[:, ys_index].reshape(-1, 1)
        temper_input = np.hstack((final_x, predicted_ys))
        temper_preds = temper_model.predict(temper_input)
        results_df['Predicted Temper'] = le_temper.inverse_transform(temper_preds)

    valid_results = results_df[results_df['Penalty_Score'] < 5.0]
    if valid_results.empty: valid_results = results_df

    final_fitness = np.zeros(len(valid_results))
    for obj, target_val in active_batch_targets.items():
        if obj in objectives:
            error = np.abs(valid_results[obj] - target_val) / target_val
            final_fitness -= error

    valid_results['Dynamic_Score'] = final_fitness - valid_results['Uncertainty']
    top_3 = valid_results.sort_values('Dynamic_Score', ascending=False).head(3)

    return top_3

# ======================================================================================
# GLOBAL CACHE
# ======================================================================================
pipeline_cache = {}

def get_pipeline():
    if "cvae" in pipeline_cache:
        return pipeline_cache
    
    file_path = 'wrought_alloys_final.xlsx'
    if not os.path.exists(file_path):
        file_path = 'final_dataset_filled.csv.xlsx'

    df, inputs, objectives = load_and_prep_data(file_path)

    X_raw = df[inputs].values
    forward_model = ForwardEnsemble()
    forward_model.train(X_raw, df, objectives)

    temper_clf, le_temper = train_temper_model_v2(df, inputs)

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(df[inputs].values)
    y_scaled = scaler_y.fit_transform(df[objectives].values)

    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled)
    dataloader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

    cvae = CVAE(input_dim=len(inputs), cond_dim=len(objectives))
    optimizer = optim.Adam(cvae.parameters(), lr=1e-3)
    cvae.train()
    for epoch in range(50):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            recon_x, mu, logvar = cvae(batch_x, batch_y)
            loss = loss_function(recon_x, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()
    
    pipeline_cache.update({
        "cvae": cvae, "forward_model": forward_model, "scaler_x": scaler_x,
        "scaler_y": scaler_y, "objectives": objectives, "temper_clf": temper_clf,
        "le_temper": le_temper, "df": df, "inputs": inputs
    })
    return pipeline_cache

# ======================================================================================
# API ENDPOINTS
# ======================================================================================
class GenerateRequest(BaseModel):
    batch_name: str
    custom_targets: dict = None

@app.post("/generate")
def generate(req: GenerateRequest):
    all_batches = {
        'B1': {'YS (MPa)': 280.0, 'UTS (MPa)': 310.0, 'EC Volume (% IACS)': 55.0},
        'B2': {'TC (W/m-K)': 200.0, 'TE Coeff': 23.0},
        'B3': {'YS (MPa)': 250.0, 'EC Volume (% IACS)': 58.0},
        'B4': {'YS (MPa)': 300.0, 'Fatigue Strength (MPa)': 120.0}
    }
    if req.batch_name not in all_batches:
        raise HTTPException(400, "Invalid batch name. Choose B1, B2, B3, or B4.")
    
    p = get_pipeline()
    p["cvae"].eval()
    
    if req.custom_targets:
        active_batch_targets = req.custom_targets
    else:
        active_batch_targets = all_batches[req.batch_name]
        
    top_alloys = inverse_design_pipeline(
        p["cvae"], p["forward_model"], p["scaler_x"], p["scaler_y"], p["objectives"],
        p["temper_clf"], p["le_temper"], active_batch_targets,
        df_train=p["df"], elements=p["inputs"]
    )
    
    # Show ALL properties in output (exactly like original code), not just targeted ones
    display_cols = p["inputs"] + p["objectives"] + ['Uncertainty']
    if p["temper_clf"]:
        display_cols.append('Predicted Temper')
    # Only include cols that exist in the dataframe
    display_cols = [c for c in display_cols if c in top_alloys.columns]
        
    final_df = top_alloys[display_cols]
    
    results = final_df.to_dict(orient="records")
    return {
        "status": "success", 
        "batch": req.batch_name, 
        "targeted_keys": list(active_batch_targets.keys()),
        "targets": active_batch_targets, 
        "objectives": p["objectives"],
        "accuracies": p["forward_model"].accuracies,
        "data": results
    }

@app.get("/")
def read_root():
    return {"message": "API Inclusive is running. Go to /docs for Swagger UI."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
