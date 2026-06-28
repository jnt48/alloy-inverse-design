# 🧪 Alloy Inverse Design Platform

AI-powered generation of high-performance aluminum alloy compositions using **Conditional Variational Autoencoders (CVAE)** and a **7-model Ensemble ML pipeline**.

---

## 🚀 Live Demo
> **API:** `https://alloy-inverse-design-api.onrender.com`  
> **Frontend:** Served directly from the API at `/`

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────┐
│              Frontend (index.html)            │
│     Beautiful dashboard served by FastAPI    │
└───────────────────┬──────────────────────────┘
                    │ POST /generate
                    ▼
┌──────────────────────────────────────────────┐
│           FastAPI Backend (app.py)           │
│                                              │
│  Mode: inclusive → Trains on all alloys      │
│  Mode: exclusive → Filters 2xxx/7xxx         │
│                                              │
│  ┌────────────┐   ┌──────────────────────┐  │
│  │ CVAE       │   │ 7-Model Ensemble     │  │
│  │ Generator  │   │ RF, GB, XGB, ET,     │  │
│  │ (50 epochs)│   │ HGB, SVR, MLP        │  │
│  └────────────┘   └──────────────────────┘  │
│                                              │
│  30-Generation Evolutionary Loop             │
│  → Top 3 alloy candidates returned          │
└──────────────────────────────────────────────┘
```

---

## ⚡ Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the unified API (serves both backend + frontend)
python app.py

# Open browser at http://localhost:8000
```

---

## ☁️ Deploy to Render (Free Hosting)

### Step 1 — Push to GitHub
```bash
cd r_internship_api
git init
git add .
git commit -m "Initial commit: Alloy Inverse Design API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/alloy-inverse-design.git
git push -u origin main
```

### Step 2 — Deploy on Render
1. Go to [render.com](https://render.com) → **Sign up** (free)
2. Click **"New +"** → **"Web Service"**
3. Connect your **GitHub repo**
4. Fill in:
   - **Name:** `alloy-inverse-design-api`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Click **"Create Web Service"**
6. Wait ~5 minutes for build → Your API is live! 🎉

### Step 3 — Access your app
- **Frontend Dashboard:** `https://your-app-name.onrender.com/`
- **API Docs (Swagger):** `https://your-app-name.onrender.com/docs`
- **Health Check:** `https://your-app-name.onrender.com/health`

---

## 📡 API Reference

### `POST /generate`
Generate top 3 alloy candidates for a given batch target.

**Request Body:**
```json
{
  "batch_name": "B1",
  "mode": "inclusive",
  "custom_targets": null
}
```

| Field | Values | Description |
|---|---|---|
| `batch_name` | `B1`, `B2`, `B3`, `B4` | Pre-defined target batch |
| `mode` | `inclusive`, `exclusive` | Include or exclude 2xxx/7xxx series |
| `custom_targets` | `{"YS (MPa)": 300, ...}` | Override default batch targets |

**Batch Definitions:**
| Batch | Properties |
|---|---|
| B1 | YS=280, UTS=310, EC=55 |
| B2 | TC=200, TE Coeff=23 |
| B3 | YS=250, EC=58 |
| B4 | YS=300, Fatigue=120 |

---

## 🤝 Internship Project
Developed as part of materials science internship — AI-driven inverse design of aluminum alloys using generative deep learning and multi-objective optimization.
