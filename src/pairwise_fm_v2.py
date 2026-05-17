"""
pairwise_fm.py  —  LambdaFM: Pairwise FM mit gestuften Paaren + ΔNDCG-Gewichtung

Was neu ist gegenüber der einfachen Pairwise-FM:

  1. GESTUFTE PAARE
     Statt nur (relevant vs. irrelevant) bilden wir drei Paar-Typen:
       - Buchung (5) vs. Ignoriert (0)  → stärkstes Signal
       - Buchung (5) vs. Klick     (1)  → mittleres Signal
       - Klick   (1) vs. Ignoriert (0)  → schwächeres Signal
     Jeder Typ bekommt ein eigenes Gewicht (pair_weights).

  2. ΔNDCG-GEWICHTUNG (LambdaRank-Trick)
     Pro Epoche wird zu Beginn für jedes Paar berechnet:
     "Wie stark würde sich NDCG verbessern, wenn wir dieses Paar korrekt ordnen?"
     Paare die oben in der Rangliste falsch sind → hohes Gewicht
     Paare die unten falsch sind → niedriges Gewicht

     Loss = Σ  ΔNDCG(i,j) * softplus(-(s_i - s_j))

     Das ist der Mechanismus der LambdaMART so stark macht —
     wir bauen ihn jetzt in die FM ein.

Notebook-Nutzung (identisch zur alten pairwise_fm.py):
    from pointwise_fm import fit_encoders, transform
    from pairwise_fm import build_pairs, train_pairwise_fm

    encoders, field_dims = fit_encoders(df_train_pd, n_bins=32)
    X_tr, _ = transform(df_train_pd, encoders)
    X_va, _ = transform(df_val_pd,   encoders)
    pairs    = build_pairs(df_train_pd, max_pairs_per_group=50)
    model, best_ndcg = train_pairwise_fm(X_tr, pairs, X_va, df_val_pd, field_dims)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchfm.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear

from pointwise_fm import ndcg_at_k, GROUP_COL, RELEVANCE_COL


# -----------------------------------------------------------------------
# Modell — identisch zur alten Version, kein Sigmoid
# -----------------------------------------------------------------------
class FMScorer(torch.nn.Module):
    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.linear    = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.fm        = FactorizationMachine(reduce_sum=True)
        # Kleinere Initialisierung → verhindert explodierende Scores
        torch.nn.init.normal_(self.embedding.embedding.weight, mean=0.0, std=0.01)

    def forward(self, x):
        return (self.linear(x) + self.fm(self.embedding(x))).squeeze(-1)


# -----------------------------------------------------------------------
# Gestufte Paare bauen
#
# Drei Paar-Typen mit unterschiedlicher semantischer Stärke:
#   Buchung > Ignoriert  (5 vs 0) → pair_weight = 1.0
#   Buchung > Klick      (5 vs 1) → pair_weight = 0.7
#   Klick   > Ignoriert  (1 vs 0) → pair_weight = 0.4
#
# Warum gestuft?
#   Eine Buchung ist ein stärkeres Signal als ein Klick.
#   Wenn das Modell weiß dass gebuchte Hotels DEUTLICH höher ranken
#   sollen als geklickte, wird der Score-Abstand zwischen den Klassen
#   größer → besseres NDCG.
# -----------------------------------------------------------------------
PAIR_TYPES = [
    (5, 0, 1.0),   # (rel_pos, rel_neg, gewicht)
    (5, 1, 0.7),
    (1, 0, 0.4),
]

def build_pairs(df_train: pd.DataFrame,
                max_pairs_per_group: int = 50,
                seed: int = 42) -> np.ndarray:
    """
    Gibt (n_pairs, 3) zurück: [idx_pos, idx_neg, pair_weight * 100 als int]
    Gewicht als int gespeichert (×100) um ein homogenes int64-Array zu haben.
    In train_pairwise_fm wird es wieder durch 100 geteilt.
    """
    print("Erstelle gestufte Paar-Indizes...")
    rng = np.random.default_rng(seed)
    df  = df_train.reset_index(drop=True)
    rel = df[RELEVANCE_COL].to_numpy()
    grp = df[GROUP_COL].to_numpy()

    pos_list, neg_list, w_list = [], [], []

    for _, sub in pd.DataFrame({"g": grp, "rel": rel, "pos": np.arange(len(grp))}).groupby("g", sort=False):
        idx     = sub["pos"].to_numpy()
        rel_sub = sub["rel"].to_numpy()

        for rel_pos, rel_neg, weight in PAIR_TYPES:
            pos_idx = idx[rel_sub == rel_pos]
            neg_idx = idx[rel_sub == rel_neg]

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue

            n_pairs    = min(len(pos_idx) * len(neg_idx), max_pairs_per_group)
            chosen_pos = rng.choice(pos_idx, size=n_pairs, replace=True)
            chosen_neg = rng.choice(neg_idx, size=n_pairs, replace=True)

            pos_list.append(chosen_pos)
            neg_list.append(chosen_neg)
            w_list.append(np.full(n_pairs, int(weight * 100), dtype=np.int64))

    pairs = np.stack([
        np.concatenate(pos_list),
        np.concatenate(neg_list),
        np.concatenate(w_list),
    ], axis=1).astype("int64")

    total = len(pairs)
    for rel_pos, rel_neg, w in PAIR_TYPES:
        n = (pairs[:, 2] == int(w * 100)).sum()
        print(f"  rel={rel_pos} vs rel={rel_neg} (w={w:.1f}): {n:,} Paare")
    print(f"  Gesamt: {total:,} Paare")
    return pairs


# -----------------------------------------------------------------------
# ΔNDCG-Gewichtung (LambdaRank-Kern)
#
# Für jedes Paar (i, j): wie stark ändert sich NDCG wenn wir
# i und j in der aktuellen Rangliste tauschen?
#
# ΔDCG(i,j) = |gain_i - gain_j| × |1/log2(pi+2) - 1/log2(pj+2)|
# ΔNDCG     = ΔDCG / IDCG  (normiert pro Gruppe)
#
# Paare oben in der Liste (kleine pi, pj) → großer Positionsterm
# Paare unten in der Liste                → kleiner Positionsterm
# -----------------------------------------------------------------------
def compute_delta_ndcg_weights(scores_tr: np.ndarray,
                                pairs: np.ndarray,
                                rel_tr: np.ndarray,
                                grp_tr: np.ndarray,
                                k: int = 5) -> np.ndarray:
    """
    Berechnet ΔNDCG-Gewichte für alle Paare basierend auf den aktuellen Scores.
    Wird zu Beginn jeder Epoche aufgerufen.
    """
    discounts = 1.0 / np.log2(np.arange(2, k + 100))  # genug Platz für große Gruppen

    # Ranks pro Gruppe berechnen (0-basiert, 0 = bester Score)
    ranks = np.zeros(len(scores_tr), dtype=np.int32)
    df_g  = pd.DataFrame({"g": grp_tr, "s": scores_tr, "i": np.arange(len(scores_tr))})
    for _, sub in df_g.groupby("g", sort=False):
        order = np.argsort(-sub["s"].to_numpy(), kind="stable")
        ranks[sub["i"].to_numpy()[order]] = np.arange(len(order))

    # IDCG pro Gruppe (für Normierung)
    idcg = {}
    df_r = pd.DataFrame({"g": grp_tr, "rel": rel_tr})
    for gid, sub in df_r.groupby("g", sort=False):
        gains = 2.0 ** sub["rel"].to_numpy() - 1.0
        ideal = np.sort(gains)[::-1]
        idcg[gid] = (ideal[:k] * discounts[:min(k, len(ideal))]).sum()

    # ΔNDCG für jedes Paar
    i_pos  = pairs[:, 0]
    i_neg  = pairs[:, 1]
    gain_p = 2.0 ** rel_tr[i_pos] - 1.0
    gain_n = 2.0 ** rel_tr[i_neg] - 1.0
    r_p    = ranks[i_pos]
    r_n    = ranks[i_neg]

    delta_dcg = np.abs(gain_p - gain_n) * np.abs(
        discounts[np.minimum(r_p, len(discounts) - 1)] -
        discounts[np.minimum(r_n, len(discounts) - 1)]
    )

    # Gruppe pro Paar für IDCG-Normierung
    grp_pos   = grp_tr[i_pos]
    idcg_vals = np.array([idcg.get(g, 1.0) for g in grp_pos])
    delta_ndcg = delta_dcg / np.maximum(idcg_vals, 1e-8)

    return delta_ndcg.astype(np.float32)


# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------
def train_pairwise_fm(X_tr:      np.ndarray,
                      pairs:     np.ndarray,   # (n_pairs, 3): pos_idx, neg_idx, weight*100
                      X_va:      np.ndarray,
                      df_val:    pd.DataFrame,
                      field_dims,
                      embed_dim:      int   = 32,
                      batch_size:     int   = 4096,
                      epochs:         int   = 30,
                      patience:       int   = 5,
                      lr:             float = 1e-3,
                      l2:             float = 1e-3,
                      ndcg_k:         int   = 5,
                      use_delta_ndcg: bool  = True,
                      device:         str   = None):

    device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | embed_dim={embed_dim} | l2={l2} | delta_ndcg={use_delta_ndcg}")

    rel_va  = df_val[RELEVANCE_COL].to_numpy(dtype="float64")
    grp_va  = df_val[GROUP_COL].to_numpy()

    # Training-Relevanz für ΔNDCG-Berechnung
    rel_tr  = None
    grp_tr  = None
    if use_delta_ndcg and RELEVANCE_COL in df_val.columns:
        # Wir brauchen rel/grp des Train-Sets — über pairs rekonstruierbar
        # Besser: direkt übergeben lassen. Fallback: ohne ΔNDCG.
        use_delta_ndcg = False
        print("  Hinweis: für ΔNDCG rel_tr/grp_tr direkt übergeben (siehe use_lambda=True Variante)")

    model  = FMScorer(field_dims, embed_dim=embed_dim).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    # Val komplett auf CPU, batch-weise auf GPU — spart VRAM
    Xva_cpu = torch.from_numpy(np.array(X_va, copy=True))
    X_tr_t  = torch.from_numpy(np.array(X_tr, copy=True)).to(device)

    # Pair-Gewichte (die gestuften Paar-Typen, nicht ΔNDCG)
    pair_weights = torch.from_numpy(pairs[:, 2].astype(np.float32) / 100.0)

    pair_ds = TensorDataset(
        torch.from_numpy(np.array(pairs[:, :2], copy=True)),  # idx
        pair_weights                                            # gewicht
    )
    loader  = DataLoader(pair_ds, batch_size=batch_size, shuffle=True)

    best_ndcg, best_state, waited = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, n_seen = 0.0, 0

        for pair_batch, w_batch in loader:
            pair_batch = pair_batch.to(device)
            w_batch    = w_batch.to(device)
            i_pos, i_neg = pair_batch[:, 0], pair_batch[:, 1]

            s_pos = model(X_tr_t[i_pos])
            s_neg = model(X_tr_t[i_neg])

            # BPR-Loss gewichtet nach Paar-Typ
            # w_batch: 1.0 für Buchung>Ignoriert, 0.7 für Buchung>Klick, 0.4 für Klick>Ignoriert
            bpr   = F.softplus(-(s_pos - s_neg))
            loss  = (bpr * w_batch).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * len(i_pos)
            n_seen       += len(i_pos)

        # Val-Scoring in Batches (vermeidet OOM)
        model.eval()
        scores_list = []
        with torch.no_grad():
            for i in range(0, len(Xva_cpu), 8192):
                batch = Xva_cpu[i:i + 8192].to(device)
                scores_list.append(model(batch).cpu())
        scores = torch.cat(scores_list).numpy()

        ndcg = ndcg_at_k(rel_va, scores, grp_va, k=ndcg_k)
        print(f"  Epoche {epoch:2d} | Loss {running_loss / n_seen:.4f} | Val NDCG@{ndcg_k} {ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg, waited = ndcg, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                print(f"  Early Stopping in Epoche {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Bestes Val-NDCG@{ndcg_k}: {best_ndcg:.4f}")
    return model, best_ndcg


# -----------------------------------------------------------------------
# Vollständige LambdaFM-Variante mit ΔNDCG
# Separat weil sie rel_tr und grp_tr des Train-Sets braucht
# -----------------------------------------------------------------------
def train_lambda_fm(X_tr:      np.ndarray,
                    pairs:     np.ndarray,
                    rel_tr:    np.ndarray,   # relevance des Train-Sets
                    grp_tr:    np.ndarray,   # srch_id des Train-Sets
                    X_va:      np.ndarray,
                    df_val:    pd.DataFrame,
                    field_dims,
                    embed_dim:  int   = 32,
                    batch_size: int   = 4096,
                    epochs:     int   = 30,
                    patience:   int   = 5,
                    lr:         float = 1e-3,
                    l2:         float = 1e-3,
                    ndcg_k:     int   = 5,
                    device:     str   = None):
    """
    LambdaFM: paarweise FM + gestufte Paare + ΔNDCG-Gewichtung.

    Zu Beginn jeder Epoche:
      1. Alle Train-Scores berechnen (kein Gradient)
      2. ΔNDCG pro Paar berechnen (wie stark ändert Tausch das NDCG?)
      3. Loss = Σ (pair_weight × ΔNDCG) × BPR
    """
    device  = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LambdaFM | Device: {device} | embed_dim={embed_dim} | l2={l2}")

    rel_va  = df_val[RELEVANCE_COL].to_numpy(dtype="float64")
    grp_va  = df_val[GROUP_COL].to_numpy()

    model   = FMScorer(field_dims, embed_dim=embed_dim).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    Xva_cpu    = torch.from_numpy(np.array(X_va,  copy=True))
    X_tr_t     = torch.from_numpy(np.array(X_tr,  copy=True)).to(device)
    pair_w_base = pairs[:, 2].astype(np.float32) / 100.0  # gestufte Gewichte

    best_ndcg, best_state, waited = -1.0, None, 0

    for epoch in range(1, epochs + 1):

        # --- ΔNDCG berechnen (zu Beginn jeder Epoche) ---
        model.eval()
        scores_tr_list = []
        with torch.no_grad():
            for i in range(0, len(X_tr), 8192):
                batch = X_tr_t[i:i + 8192]
                scores_tr_list.append(model(batch).cpu().numpy())
        scores_tr = np.concatenate(scores_tr_list)

        delta_ndcg = compute_delta_ndcg_weights(scores_tr, pairs, rel_tr, grp_tr, k=ndcg_k)

        # Kombiniertes Gewicht: Paar-Typ × ΔNDCG
        combined_w = torch.from_numpy(pair_w_base * delta_ndcg)

        # --- Training ---
        model.train()
        pair_ds = TensorDataset(
            torch.from_numpy(np.array(pairs[:, :2], copy=True)),
            combined_w
        )
        loader = DataLoader(pair_ds, batch_size=batch_size, shuffle=True)

        running_loss, n_seen = 0.0, 0
        for pair_batch, w_batch in loader:
            pair_batch = pair_batch.to(device)
            w_batch    = w_batch.to(device)
            i_pos, i_neg = pair_batch[:, 0], pair_batch[:, 1]

            s_pos = model(X_tr_t[i_pos])
            s_neg = model(X_tr_t[i_neg])

            bpr  = F.softplus(-(s_pos - s_neg))
            loss = (bpr * w_batch).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item() * len(i_pos)
            n_seen       += len(i_pos)

        # --- Val-NDCG ---
        model.eval()
        scores_va = []
        with torch.no_grad():
            for i in range(0, len(Xva_cpu), 8192):
                batch = Xva_cpu[i:i + 8192].to(device)
                scores_va.append(model(batch).cpu())
        scores = torch.cat(scores_va).numpy()

        ndcg = ndcg_at_k(rel_va, scores, grp_va, k=ndcg_k)
        print(f"  Epoche {epoch:2d} | Loss {running_loss / n_seen:.4f} | "
              f"Val NDCG@{ndcg_k} {ndcg:.4f} | "
              f"mean ΔNDCG-w {delta_ndcg.mean():.4f}")

        if ndcg > best_ndcg:
            best_ndcg, waited = ndcg, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                print(f"  Early Stopping in Epoche {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Bestes Val-NDCG@{ndcg_k}: {best_ndcg:.4f}")
    return model, best_ndcg
