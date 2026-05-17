"""
Pairwise Factorization Machine fuer Expedia Hotel Ranking.

Nutzt die volle Datenmenge (kein Downsampling) und optimiert direkt
die paarweise Reihenfolge mittels BPR-Loss.

Abhaengigkeiten:
  Es wird zwingend `pointwise_fm.py` im selben Verzeichnis benoetigt,
  da wir von dort die Encoder, Features und NDCG-Berechnung importieren.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear

# Wir recyclen die hart erarbeiteten Encoder und Metriken aus der Pointwise-Pipeline
from pointwise_fm import (
    fit_encoders, 
    transform, 
    ndcg_at_k, 
    RELEVANCE_COL, 
    GROUP_COL
)

# ----------------------------------------------------------------------
# 1. Das Modell (Ohne Sigmoid-Kopf!)
# ----------------------------------------------------------------------
class FMScorer(torch.nn.Module):
    """
    Identisch zur Standard-FM, aber ohne Sigmoid-Aktivierung am Ende.
    Wir brauchen die rohen Logits fuer die BPR-Loss (Score-Differenz).
    """
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        torch.nn.init.normal_(self.embedding.embedding.weight, mean=0, std=0.01)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x):
        # x shape: (batch_size, num_fields)
        linear_term = self.linear(x)
        fm_term = self.fm(self.embedding(x))
        # squeeze(-1) macht aus (batch, 1) einen 1D Tensor (batch,)
        return (linear_term + fm_term).squeeze(-1)


# ----------------------------------------------------------------------
# 2. Pair-Sampling (Die Paar-Bildung)
# ----------------------------------------------------------------------
def build_pair_index(df_train: pd.DataFrame,
                     max_pairs_per_group: int = 100,
                     seed: int = 42) -> np.ndarray:
    """
    Sucht pro srch_id nach Paaren (i, j), wobei Hotel i relevanter ist als Hotel j.
    Gibt ein Array von Indizes zurueck, das auf die transformierte Matrix (X_tr) zeigt.
    """
    print("Erstelle Paar-Indizes (Pair Sampling)...")
    rng = np.random.default_rng(seed)
    rel = df_train[RELEVANCE_COL].to_numpy()
    grp = df_train[GROUP_COL].to_numpy()

    # Hilfs-DataFrame fuer superschnelles Gruppieren
    df = pd.DataFrame({"g": grp, "rel": rel, "pos": np.arange(len(grp))})
    pos_list, neg_list = [], []
    
    for _, sub in df.groupby("g", sort=False):
        # Alle Hotels in dieser Suche mit Relevance > 0 (Klick oder Buchung)
        pos_idx = sub.loc[sub["rel"] > 0, "pos"].to_numpy()
        if len(pos_idx) == 0:
            continue
            
        # Alle Hotels in dieser Suche mit Relevance == 0 (Ignoriert)
        neg_idx_all = sub.loc[sub["rel"] == 0, "pos"].to_numpy()
        if len(neg_idx_all) == 0:
            continue
            
        # Optional koennte man hier auch (Buchung vs. Klick) einbauen,
        # wir belassen es fuer Stablitaet bei (Klick/Buchung vs. Ignoriert).
        
        # Deckelung gegen quadratische Explosion bei riesigen Suchanfragen
        n_pairs = min(len(pos_idx) * len(neg_idx_all), max_pairs_per_group)
        
        # Ziehe zufaellige Paare
        chosen_pos = rng.choice(pos_idx, size=n_pairs, replace=True)
        chosen_neg = rng.choice(neg_idx_all, size=n_pairs, replace=True)
        
        pos_list.append(chosen_pos)
        neg_list.append(chosen_neg)

    pairs = np.stack([np.concatenate(pos_list), np.concatenate(neg_list)], axis=1)
    print(f"-> {len(pairs):,} Trainings-Paare generiert.")
    return pairs.astype("int64")


# ----------------------------------------------------------------------
# 3. Trainings-Loop mit BPR-Loss
# ----------------------------------------------------------------------
def train_pairwise_fm(X_tr: np.ndarray,
                      pair_idx: np.ndarray,
                      X_va: np.ndarray,
                      df_val: pd.DataFrame,
                      field_dims,
                      embed_dim: int = 32,
                      batch_size: int = 4096,
                      epochs: int = 20,
                      patience: int = 7,
                      lr: float = 1e-3,
                      l2: float = 1e-3,
                      ndcg_k: int = 5,
                      device: str | None = None):
                      
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starte Training auf {device}...")
    
    rel_va = df_val[RELEVANCE_COL].to_numpy(dtype="float64")
    grp_va = df_val[GROUP_COL].to_numpy()

    model = FMScorer(field_dims, embed_dim=embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    # Features als kontinuierliche Tensoren im Speicher ablegen
    X_tr_t = torch.from_numpy(np.array(X_tr, copy=True)).to(device)
    Xva_t = torch.from_numpy(np.array(X_va, copy=True)).to(device)

    # Der DataLoader zieht nun (idx_pos, idx_neg) Paare!
    pair_ds = TensorDataset(torch.from_numpy(np.array(pair_idx, copy=True)))
    loader = DataLoader(pair_ds, batch_size=batch_size, shuffle=True)

    best_ndcg, best_state, waited = -1.0, None, 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        
        for (pair_batch,) in loader:
            pair_batch = pair_batch.to(device)
            i_pos, i_neg = pair_batch[:, 0], pair_batch[:, 1]
            
            # Scoren beider Hotels
            s_pos = model(X_tr_t[i_pos])
            s_neg = model(X_tr_t[i_neg])
            
            # BPR-Loss: softplus(-(s_pos - s_neg)) ist numerisch stabil
            loss = F.softplus(-(s_pos - s_neg)).mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            
            
            running_loss += loss.item() * len(i_pos)
            n_seen += len(i_pos)

        # track grad_norm to get more information on training behaviour
        #grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
        #print(f"  Gradient Norm: {grad_norm:.4f}")

        # Evaluierung auf dem Val-Set
        model.eval()
        scores_list = []
        with torch.no_grad():
            for i in range(0, len(X_va), 8192):
                batch = Xva_t[i:i+8192]
                scores_list.append(model(batch).cpu())
        scores = torch.cat(scores_list).numpy()
            
        ndcg = ndcg_at_k(rel_va, scores, grp_va, k=ndcg_k)
        print(f"  Epoche {epoch:2d} | BPR Loss: {running_loss / n_seen:.4f} | Val NDCG@{ndcg_k}: {ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg, waited = ndcg, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                print(f"  Early Stopping ausgeloest in Epoche {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"--- Training beendet! Bestes Val-NDCG@{ndcg_k}: {best_ndcg:.4f} ---")
    return model, best_ndcg


# ----------------------------------------------------------------------
# 4. Standalone-Aufruf (CLI)
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Pairwise FM fuer Expedia.")
    p.add_argument("--data", required=True, help="Pfad zur training_set_VU_DM.csv")
    p.add_argument("--n-bins", type=int, default=32)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--quick", action="store_true", help="Mini-Lauf zum Testen")
    p.add_argument("--nrows", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    
    if args.quick:
        args.nrows = args.nrows or 80_000
        args.epochs = min(args.epochs, 3)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.data, nrows=args.nrows)
    print(f"Daten geladen: {len(df):,} Zeilen.")

    # 1. Relevance-Label berechnen (0, 1, 5)
    if RELEVANCE_COL not in df.columns:
        df[RELEVANCE_COL] = np.where(df["booking_bool"] == 1, 5,
                            np.where(df["click_bool"] == 1, 1, 0))

    # 2. Split (ACHTUNG: KEIN Downsampling mehr!)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, va_idx = next(gss.split(df, groups=df[GROUP_COL]))
    df_train, df_val = df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()
    
    print(f"Vollen Train-Set behalten: {len(df_train):,} Zeilen (Kein Downsampling)")
    print(f"Val-Set: {len(df_val):,} Zeilen, {df_val[GROUP_COL].nunique():,} Suchen\n")

    # 3. Encoders auf VOLLEN Train-Set fitten und transformieren
    print("Fitte Encoder...")
    encoders, field_dims = fit_encoders(df_train, n_bins=args.n_bins)
    
    print("Transformiere Train und Val...")
    X_tr, _ = transform(df_train, encoders)
    X_va, _ = transform(df_val, encoders)
    print(f"-> {len(field_dims)} Felder | Dimension der Embedding-Matrix: {sum(field_dims):,} Zeilen\n")

    # 4. Pair Sampling generieren
    pair_idx = build_pair_index(df_train, max_pairs_per_group=100, seed=args.seed)

    # 5. Training starten
    train_pairwise_fm(
        X_tr=X_tr, 
        pair_idx=pair_idx, 
        X_va=X_va, 
        df_val=df_val, 
        field_dims=field_dims,
        embed_dim=args.embed_dim, 
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()