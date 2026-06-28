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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any

warnings.filterwarnings('ignore')

# ======================================================================================
# CONFIGURATION
# ======================================================================================
# IMPORTANT: Update this path to where your Excel/CSV file is located on the server
DATA_FILE_PATH = 'final_dataset_filled.csv (2) (1).xlsx' 

app = FastAPI(
    title="Alloy Inverse Design API",
    description="REST API for generating Aluminum Alloys using CVAE and Ensemble Machine Learning.",
    version="2.0 (Boss Model with 30-Gen Evolution)"
)

# Enable CORS so your Vercel frontend can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================================
# ML ARCHITECTURE
# ======================================================================================

class ForwardEnsemble:
    def __init__(self):
        self.models = {}

    def train(self, X, df, objectives):
        for obj in objectives:
            y = df[obj]
            self.models[obj] = []
            self.models[obj].append(RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42).fit(X, y))
            self.models[obj].append(GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42).fit(X, y))
            self.models[obj].append(xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42, objective='reg:squarederror').fit(X, y))
            self.models[obj].append(ExtraTreesRegressor(n_estimators=100, max_depth=15, random_state=42).fit(X, y))
            self.models[obj].append(HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42).fit(X, y))
            self.models[obj].append(Pipeline([('scaler', StandardScaler()), ('svr', SVR(C=10, gamma='scale'))]).fit(X, y))
            self.models[obj].append(Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))]).fit(X, y))

    def predict(self, X, obj):
        all_preds = np.array([m.predict(X) for m in self.models[obj]])
        return np.mean(all_preds, axis=0), np.std(all_preds, axis=0)

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

def train_temper_model(df, elements):
    if 'Temper' not in df.columns or 'YS (MPa)' not in df.columns: return None, None
    le = LabelEncoder()
    df_valid = df[(df['Temper'].astype(str).str.strip() != '') & (df['YS (MPa)'] > 0)]
    if len(df_valid) == 0: return None, None
    X_valid = df_valid[elements + ['YS (MPa)']].values
    y_valid = le.fit_transform(df_valid['Temper'].astype(str))
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_valid, y_valid)
    return clf, le

# ======================================================================================
# DATA & PIPELINE MANAGEMENT
# ======================================================================================

def load_and_prep_data(file_path, exclude_2x_7x=False):
    try:
        df = pd.read_csv(file_path, encoding='latin1')
    except:
        df = pd.read_excel(file_path)

    df.columns = df.columns.str.strip()
    
    # DYNAMIC FILTER LOGIC
    if exclude_2x_7x and 'Series' in df.columns:
        print(" > Filter ACTIVE: Excluding 2xxx and 7xxx series.")
        df = df[~df['Series'].str.contains('2xxx|7xxx|2000|7000', case=False, na=False)]
    else:
        print(" > Filter INACTIVE: Training on entire dataset.")

    elements = ['Al', 'Si', 'Fe', 'Cu', 'Mn', 'Mg', 'Cr', 'Ni', 'Zn', 'Ga', 'V', 'Ti']

    def parse_val(val):
        if pd.isna(val) or val == '': return 0.0
        s = str(val).strip()
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", s)]
        if len(nums) == 2: return sum(nums)/2
        if len(nums) == 1: return nums[0]
        return 0.0

    col_mapping = {
        'Elastic Modulus (GPa)': 'Y (GPa)', 'Density (g/cmÂ³)': 'Density (g/cc)', 'Density (g/cm³)': 'Density (g/cc)',
        'Yield Strength (MPa)': 'YS (MPa)', 'Thermal Expansion (Âµm/m-K)': 'TE Coeff', 'Thermal Expansion (µm/m-K)': 'TE Coeff',
        'Elec. Conductivity Vol (% IACS)': 'EC Volume (% IACS)', 'Thermal Conductivity (W/m-K)': 'TC (W/m-K)'
    }
    df.rename(columns=col_mapping, inplace=True)

    cols_to_clean = elements + ['Y (GPa)', 'Density (g/cc)', 'YS (MPa)', 'TE Coeff']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].apply(parse_val)

    if 'Y (GPa)' in df.columns and 'Density (g/cc)' in df.columns:
        df['Sag Resistance'] = df['Y (GPa)'] / df['Density (g/cc)']
    else:
        df['Sag Resistance'] = 0.0
    df['Sag Resistance'] = df['Sag Resistance'].replace([np.inf, -np.inf], 0).fillna(0)

    objectives = ['EC Volume (% IACS)', 'YS (MPa)', 'Sag Resistance', 'TC (W/m-K)', 'TE Coeff']
    for col in objectives:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    available_obj = [o for o in objectives if o in df.columns]
    df = df[(df[available_obj] > 0).any(axis=1)].fillna(0.0)

    return df, elements, available_obj

# Cache to store trained pipelines so we don't retrain on every API call
ml_cache = {
    "included": None,
    "excluded": None
}

def get_or_train_pipeline(exclude_2x_7x: bool):
    cache_key = "excluded" if exclude_2x_7x else "included"
    
    if ml_cache[cache_key] is not None:
        return ml_cache[cache_key]
    
    print(f"[*] Training pipeline from scratch. Mode: {cache_key}")
    df, inputs, objectives = load_and_prep_data(DATA_FILE_PATH, exclude_2x_7x)
    
    X_raw = df[inputs].values
    fw_model = ForwardEnsemble()
    fw_model.train(X_raw, df, objectives)
    
    t_clf, le_t = train_temper_model(df, inputs)

    sx, sy = MinMaxScaler(), MinMaxScaler()
    X_scaled = torch.FloatTensor(sx.fit_transform(X_raw))
    y_scaled = torch.FloatTensor(sy.fit_transform(df[objectives].values))
    
    cvae = CVAE(len(inputs), len(objectives))
    opt = optim.Adam(cvae.parameters(), lr=1e-3)
    
    dataloader = DataLoader(TensorDataset(X_scaled, y_scaled), batch_size=32, shuffle=True)
    cvae.train()
    for _ in range(50):
        for batch_x, batch_y in dataloader:
            opt.zero_grad()
            recon_x, mu, logvar = cvae(batch_x, batch_y)
            loss = loss_function(recon_x, batch_x, mu, logvar)
            loss.backward()
            opt.step()
            
    pipeline = {
        "cvae": cvae, "fw_model": fw_model, "sx": sx, "sy": sy, 
        "objs": objectives, "t_clf": t_clf, "le_t": le_t, "inputs": inputs, "df": df
    }
    ml_cache[cache_key] = pipeline
    return pipeline

# ======================================================================================
# API SCHEMAS & ENDPOINTS
# ======================================================================================

class TargetWishlist(BaseModel):
    ec_volume: float = Field(58.0, alias="EC Volume (% IACS)")
    ys_mpa: float = Field(280.0, alias="YS (MPa)")
    sag_resistance: float = Field(105.0, alias="Sag Resistance")
    tc: float = Field(180.0, alias="TC (W/m-K)")
    te_coeff: float = Field(22.0, alias="TE Coeff")
    exclude_2x_7x: bool = Field(False, description="Set to True to exclude Aerospace alloys")

@app.post("/generate-alloy")
def generate_alloy(wishlist: TargetWishlist):
    try:
        # 1. Load or Train the appropriate pipeline based on user request
        p = get_or_train_pipeline(wishlist.exclude_2x_7x)
        
        # 2. Format targets
        targets = {
            'EC Volume (% IACS)': wishlist.ec_volume,
            'YS (MPa)': wishlist.ys_mpa,
            'Sag Resistance': wishlist.sag_resistance,
            'TC (W/m-K)': wishlist.tc,
            'TE Coeff': wishlist.te_coeff
        }
        valid_wishlist = {k: v for k, v in targets.items() if k in p["objs"]}
        
        p["cvae"].eval()
        t_tensor = torch.FloatTensor(p["sy"].transform(pd.DataFrame([valid_wishlist])[p["objs"]]))
        
        # ==========================================================
        # FULL 30-GENERATION EVOLUTIONARY LOOP
        # ==========================================================
        POP_SIZE = 500
        GENERATIONS = 30
        LATENT_DIM = 8
        LAMBDA_PENALTY = 10.0

        z_pop = torch.randn(POP_SIZE, LATENT_DIM)
        c_cond = t_tensor.repeat(POP_SIZE, 1)
        
        element_bounds = {i: {'max': p["df"][el].max(), 'min': p["df"][el].min()} for i, el in enumerate(p["inputs"])}

        for gen in range(GENERATIONS):
            with torch.no_grad():
                recon_x = p["cvae"].decoder(torch.cat([z_pop, c_cond], dim=1)).numpy()
            real_x = p["sx"].inverse_transform(recon_x)

            penalty_scores = np.abs(real_x.sum(axis=1) - 100.0)
            for i in range(len(p["inputs"])):
                penalty_scores += np.maximum(0, element_bounds[i]['min'] - real_x[:, i])
                penalty_scores += np.maximum(0, real_x[:, i] - element_bounds[i]['max'])

            real_x_corr = np.maximum(real_x, 0)
            sums_corr = real_x_corr.sum(axis=1, keepdims=True)
            sums_corr[sums_corr==0] = 1
            real_x_corr = (real_x_corr / sums_corr) * 100.0

            preds_list, unc_list = [], []
            for obj in p["objs"]:
                mu, std = p["fw_model"].predict(real_x_corr, obj)
                preds_list.append(mu)
                unc_list.append(std)

            preds_arr = np.column_stack(preds_list)
            unc_arr = np.column_stack(unc_list)

            # Dynamic Grading
            fitness = np.zeros(POP_SIZE)
            for obj, target_val in valid_wishlist.items():
                idx = p["objs"].index(obj)
                # Compute percentage error (negative because we maximize fitness)
                error = np.abs(preds_arr[:, idx] - target_val) / (target_val + 1e-9)
                fitness -= error

            risk = np.mean(unc_arr, axis=1) * 0.5
            fitness = fitness - risk - (LAMBDA_PENALTY * penalty_scores)

            top_k = POP_SIZE // 2
            top_indices = np.argsort(fitness)[-top_k:]
            best_z = z_pop[top_indices]

            new_z = [(best_z[np.random.randint(top_k)] + best_z[np.random.randint(top_k)])/2.0 + torch.randn(LATENT_DIM)*0.1 for _ in range(POP_SIZE)]
            z_pop = torch.stack(new_z)

        # ==========================================================
        # FINAL GENERATION & RESULTS FORMATTING
        # ==========================================================
        with torch.no_grad():
            final_x_raw = p["cvae"].decoder(torch.cat([z_pop, c_cond], dim=1)).numpy()

        final_x = p["sx"].inverse_transform(final_x_raw)
        final_penalties = np.abs(final_x.sum(axis=1) - 100.0)
        for i in range(len(p["inputs"])):
            final_penalties += np.maximum(0, element_bounds[i]['min'] - final_x[:, i])
            final_penalties += np.maximum(0, final_x[:, i] - element_bounds[i]['max'])

        final_x = np.maximum(final_x, 0)
        final_x = (final_x / final_x.sum(axis=1, keepdims=True)) * 100.0

        final_preds_list, final_unc_list = [], []
        for obj in p["objs"]:
            mu, std = p["fw_model"].predict(final_x, obj)
            final_preds_list.append(mu)
            final_unc_list.append(std)

        final_preds_arr = np.column_stack(final_preds_list)
        final_unc_arr = np.mean(np.column_stack(final_unc_list), axis=1)

        res_df = pd.DataFrame(final_x, columns=p["inputs"])
        res_df['Uncertainty'] = final_unc_arr
        res_df['Penalty_Score'] = final_penalties

        for i, obj in enumerate(p["objs"]): 
            res_df[obj] = final_preds_arr[:, i]
        
        if p["t_clf"] and 'YS (MPa)' in p["objs"]:
            ys_idx = p["objs"].index('YS (MPa)')
            temper_in = np.hstack((final_x, final_preds_arr[:, ys_idx].reshape(-1, 1)))
            res_df['Predicted Temper'] = p["le_t"].inverse_transform(p["t_clf"].predict(temper_in))

        # Filter and Pareto Score
        valid_results = res_df[res_df['Penalty_Score'] < 5.0]
        if valid_results.empty: valid_results = res_df

        final_fitness = np.zeros(len(valid_results))
        for obj, target_val in valid_wishlist.items():
            error = np.abs(valid_results[obj] - target_val) / (target_val + 1e-9)
            final_fitness -= error

        valid_results['Dynamic_Score'] = final_fitness - valid_results['Uncertainty']
        top3 = valid_results.sort_values('Dynamic_Score', ascending=False).head(3)

        # 4. Format Beautiful Output for the Frontend
        output = []
        for rank, (_, row) in enumerate(top3.iterrows(), 1):
            composition = {el: round(row[el], 3) for el in p["inputs"] if row[el] > 0.01}
            properties = {obj: round(row[obj], 2) for obj in p["objs"]}
            
            candidate = {
                "rank": rank,
                "composition": composition,
                "predicted_properties": properties,
                "uncertainty_score": round(row['Uncertainty'], 4),
                "dynamic_score": round(row['Dynamic_Score'], 4),
                "predicted_temper": row.get('Predicted Temper', 'Unknown')
            }
            output.append(candidate)

        return {
            "status": "success",
            "model_mode": "Excluding 2x/7x" if wishlist.exclude_2x_7x else "Including All Series",
            "target_wishlist": valid_wishlist,
            "top_candidates": output
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))