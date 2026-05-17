"""
deep_fm.py — DeepFM mit paarweiser LambdaRank-Loss + Optuna Hyperparameter-Suche

Architektur:
  - Shared Embedding Layer (identisch zu FM und pairwise_fm_v2.py)
  - FM-Komponente:  linearer Term + Σ Σ <v_i, v_j>
  - Deep-Komponente: MLP auf den verketteten Embeddings
  - Output: FM-Score + MLP-Score (kein Sigmoid — pairwise Loss)

Warum DeepFM über pairwise FM hinaus?
  - FM lernt nur paarweise Interaktionen (Ordnung 2)
  - MLP lernt implizit Interaktionen höherer Ordnung (per_fee, score2ma etc.)
  - Beide nutzen dieselbe Embedding-Schicht → end-to-end Training

Notebook-Nutzung:
    from pointwise_fm import fit_encoders, transform
    from pairwise_fm_v2 import build_pairs
    from deep_fm import DeepFMScorer, train_deep_lambda_fm, run_optuna

    encoders, field_dims = fit_encoders(df_train_pd, n_bins=32)
    X_tr, _ = transform(df_train_pd, encoders)
    X_va, _ = transform(df_val_pd,   encoders)
    pairs    = build_pairs(df_train_pd, max_pairs_per_group=50)
    rel_tr   = df_train_pd["relevance"].to_numpy()
    grp_tr   = df_train_pd["srch_id"].to_numpy()

    # Einzelner Run:
    model, best_ndcg = train_deep_lambda_fm(
        X_tr, pairs, rel_tr, grp_tr, X_va, df_val_pd, field_dims)

    # Optuna Hyperparameter-Suche:
    best_params = run_optuna(
        X_tr, pairs, rel_tr, grp_tr, X_va, df_val_pd, field_dims,
        n_trials=20)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pointwise_fm import ndcg_at_k, GROUP_COL, RELEVANCE_COL
from pairwise_fm_v2 import compute_delta_ndcg_weights


# -----------------------------------------------------------------------
# DeepFM Architektur
#
#                    Sparse Input (Integer-Codes)
#                           |
#              ┌────────────┴────────────┐
#              │    Shared Embeddings    │   ← eine Tabelle, beide Teile
#              └──────┬──────────┬───────┘
#                     │          │
#              ┌──────┴──┐  ┌───┴──────────┐
#              │ FM-Teil │  │  Deep-Teil   │
#              │ lin + Σ │  │  MLP(concat) │
#              │<vi,vj>  │  │  ReLU layers │
#              └──────┬──┘  └───┬──────────┘
#                     │         │
#                     └────┬────┘
#                        Score (kein Sigmoid)
# -----------------------------------------------------------------------
class DeepFMScorer(nn.Module):
    def __init__(self, field_dims, embed_dim=16, mlp_dims=(256, 128, 64), dropout=0.2):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear    = FeaturesLinear(field_dims)
        self.fm        = FactorizationMachine(reduce_sum=True)

        # MLP Input: alle Embeddings flachgeklopft
        mlp_input_dim = len(field_dims) * embed_dim
        layers = []
        in_dim = mlp_input_dim
        for out_dim in mlp_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        # Kleinere Initialisierung verhindert explodierende Scores
        nn.init.normal_(self.embedding.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        emb = self.embedding(x)                          # (B, F, k)
        fm_out  = self.linear(x) + self.fm(emb)         # (B, 1)
        mlp_out = self.mlp(emb.view(emb.size(0), -1))   # (B, 1)
        return (fm_out + mlp_out).squeeze(-1)            # (B,)


# -----------------------------------------------------------------------
# Val-Scoring in Batches (verhindert OOM bei ~1M Val-Zeilen)
# -----------------------------------------------------------------------
def _score_batched(model, X_cpu, device, batch_size=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_cpu), batch_size):
            out.append(model(X_cpu[i:i + batch_size].to(device)).cpu())
    return torch.cat(out).numpy()


# -----------------------------------------------------------------------
# Training: LambdaRank-Loss + gestufte Paare + ΔNDCG-Gewichtung
# -----------------------------------------------------------------------
def train_deep_lambda_fm(X_tr:      np.ndarray,
                         pairs:     np.ndarray,   # (n_pairs, 3) aus build_pairs
                         rel_tr:    np.ndarray,
                         grp_tr:    np.ndarray,
                         X_va:      np.ndarray,
                         df_val:    pd.DataFrame,
                         field_dims,
                         embed_dim:  int          = 16,
                         mlp_dims:   tuple         = (256, 128, 64),
                         dropout:    float         = 0.2,
                         batch_size: int           = 4096,
                         epochs:     int           = 30,
                         patience:   int           = 5,
                         lr:         float         = 1e-3,
                         l2:         float         = 1e-3,
                         ndcg_k:     int           = 5,
                         device:     str           = None,
                         trial=None):              # Optuna trial (optional)
    """
    trial: wenn von Optuna aufgerufen, wird pruning aktiviert.
    """
    device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rel_va  = df_val[RELEVANCE_COL].to_numpy(dtype="float64")
    grp_va  = df_val[GROUP_COL].to_numpy()

    model = DeepFMScorer(field_dims, embed_dim=embed_dim,
                         mlp_dims=mlp_dims, dropout=dropout).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    X_tr_t  = torch.from_numpy(np.array(X_tr, copy=True)).to(device)
    Xva_cpu = torch.from_numpy(np.array(X_va, copy=True))

    pair_w_base = pairs[:, 2].astype(np.float32) / 100.0

    best_ndcg, best_state, waited = -1.0, None, 0

    for epoch in range(1, epochs + 1):

        # 1. ΔNDCG-Gewichte aus aktuellen Train-Scores
        scores_tr  = _score_batched(model, torch.from_numpy(np.array(X_tr, copy=True)), device)
        delta_ndcg = compute_delta_ndcg_weights(scores_tr, pairs, rel_tr, grp_tr, k=ndcg_k)
        combined_w = torch.from_numpy(pair_w_base * delta_ndcg)

        # 2. Training
        model.train()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(np.array(pairs[:, :2], copy=True)), combined_w),
            batch_size=batch_size, shuffle=True)

        running_loss, n_seen = 0.0, 0
        for pair_batch, w_batch in loader:
            pair_batch = pair_batch.to(device)
            w_batch    = w_batch.to(device)
            s_pos = model(X_tr_t[pair_batch[:, 0]])
            s_neg = model(X_tr_t[pair_batch[:, 1]])
            loss  = (F.softplus(-(s_pos - s_neg)) * w_batch).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item() * len(pair_batch); n_seen += len(pair_batch)

        # 3. Validierung
        scores = _score_batched(model, Xva_cpu, device)
        ndcg   = ndcg_at_k(rel_va, scores, grp_va, k=ndcg_k)
        print(f"  Epoche {epoch:2d} | Loss {running_loss/n_seen:.4f} | "
              f"Val NDCG@{ndcg_k} {ndcg:.4f} | mean Δw {delta_ndcg.mean():.4f}")

        # Optuna Pruning
        if trial is not None:
            trial.report(ndcg, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if ndcg > best_ndcg:
            best_ndcg, waited = ndcg, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                print(f"  Early Stopping in Epoche {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    print(f"  Bestes Val-NDCG@{ndcg_k}: {best_ndcg:.4f}")
    return model, best_ndcg


# -----------------------------------------------------------------------
# Optuna Hyperparameter-Suche
# -----------------------------------------------------------------------
def run_optuna(X_tr, pairs, rel_tr, grp_tr, X_va, df_val, field_dims,
               n_trials: int = 20,
               epochs_per_trial: int = 10,
               patience_per_trial: int = 3,
               ndcg_k: int = 5,
               device: str = None):
    """
    Sucht über:
      embed_dim:   [8, 16, 32, 64]
      mlp_dims:    verschiedene Architekturen
      dropout:     0.1 – 0.5
      lr:          1e-4 – 1e-2  (log-scale)
      l2:          1e-4 – 1e-1  (log-scale)

    Gibt best_params dict zurück.
    Nutze diese dann für einen langen finalen Run.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        embed_dim = trial.suggest_categorical("embed_dim", [8, 16, 32, 64])
        mlp_arch  = trial.suggest_categorical("mlp_arch", [
            "small",   # (128, 64)
            "medium",  # (256, 128, 64)
            "large",   # (512, 256, 128, 64)
            "deep",    # (256, 256, 128, 64)
        ])
        mlp_dims_map = {
            "small":  (128, 64),
            "medium": (256, 128, 64),
            "large":  (512, 256, 128, 64),
            "deep":   (256, 256, 128, 64),
        }
        dropout = trial.suggest_float("dropout",  0.1, 0.5)
        lr      = trial.suggest_float("lr",       1e-4, 1e-2, log=True)
        l2      = trial.suggest_float("l2",       1e-4, 1e-1, log=True)

        print(f"\nTrial {trial.number}: embed={embed_dim}, mlp={mlp_arch}, "
              f"dropout={dropout:.2f}, lr={lr:.1e}, l2={l2:.1e}")

        _, ndcg = train_deep_lambda_fm(
            X_tr, pairs, rel_tr, grp_tr, X_va, df_val, field_dims,
            embed_dim  = embed_dim,
            mlp_dims   = mlp_dims_map[mlp_arch],
            dropout    = dropout,
            lr         = lr,
            l2         = l2,
            epochs     = epochs_per_trial,
            patience   = patience_per_trial,
            ndcg_k     = ndcg_k,
            device     = device,
            trial      = trial,
        )
        return ndcg

    # MedianPruner: stoppt schwache Trials früh
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    sampler = optuna.samplers.TPESampler(seed=42)
    study   = optuna.create_study(direction="maximize",
                                  pruner=pruner, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    print("\n" + "="*50)
    print(f"Beste Trial: NDCG@{ndcg_k} = {study.best_value:.4f}")
    print("Beste Parameter:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("="*50)
    return study.best_params, study
