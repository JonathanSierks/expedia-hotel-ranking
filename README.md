![Status](https://img.shields.io/badge/status-In--Progress-orange)
![Field](https://img.shields.io/badge/field-Learning--to--Rank-blue)


# Predicting Expedia Hotel Rankings – A Learning-to-Rank Approach

This project tackles the **Personalize Expedia Hotel Searches** problem as a **learning-to-rank (LTR)** task: given a user's search query and the list of hotels returned, predict the order in which the user is most likely to **click** and **book**.

The dataset originates from the 2013 [Personalize Expedia Hotel Searches](https://www.kaggle.com/c/expedia-personalized-sort) ICDM competition. It contains roughly **4.96 million rows** across the train (54 columns) and test (50 columns) sets, where each row is a single `(search query, hotel property)` pair. Rows belonging to the same search share a `srch_id`, and the training labels are `click_bool` and `booking_bool`.

We evaluate **three models of increasing complexity**: a Logistic Regression baseline, a Deep Factorization Machine (DeepFM), and a LambdaMART learning-to-rank model. Moreover, we finished with an ethical AI assessment of performance bias between family and non-family users.

## Background / Context

The goal is to rank the hotels within each query so that booked hotels appear above clicked hotels, which in turn appear above ignored hotels. Performance is measured with **Normalized Discounted Cumulative Gain at 5 (NDCG@5)**, which rewards placing the most relevant hotels at the top of the list:

$$
\text{NDCG@}N = \frac{\text{DCG@}N}{\text{IDCG@}N}, \qquad
\text{DCG@}N = \sum_{i=1}^{N} \frac{rel_i}{\log_2(i+1)}
$$

Relevance grades follow the competition definition: **5** = booked, **1** = clicked, **0** = neither. The logarithmic discount means that getting the top ranks right matters far more than the tail, which directly motivates the choice of a ranking-aware objective.

The most successful model, **LambdaMART**, optimizes this ranking objective by scaling pairwise RankNet gradients with the change in the metric caused by swapping two items:

$$
\lambda_i = \sum_{j:\, i \succ j} |\Delta\text{NDCG}_{ij}|\,\sigma(s_j - s_i)
            \; - \sum_{j:\, j \succ i} |\Delta\text{NDCG}_{ij}|\,\sigma(s_i - s_j)
$$

where $s_i$ is the predicted score of item $i$. These pseudo-gradients let the model focus its learning on the pairs that most affect the ranking metric, and are fit by gradient-boosted regression trees (MART).

## Approaches

Three techniques were implemented, spanning the spectrum from interpretable baseline to dedicated ranking model:

- **Logistic Regression (baseline)** – predicts the probability of engagement (click or book) per row, with NDCG-aligned sample weights (booking rows weighted higher than click rows) and explicit price × quality interaction terms. Tuned over regularization strength C, penalty type (L1/L2), and booking weight.
- **Factorization Machines / DeepFM** – learn latent vectors per feature to model pairwise interactions efficiently in sparse settings. Implemented pointwise, pairwise, and DeepFM variants (FM branch + parallel MLP) using the `torchfm` library, optimizing a LambdaRank-weighted pairwise loss.
- **LambdaMART** – the dedicated LTR model, implemented with **LightGBM** GBDTs. Boosting for up to 3000 rounds with early stopping on validation NDCG@5, tuned over ~300 Optuna trials, with DART regularization explored to reduce the generalization gap.

## Feature Engineering

Around **97 features** were engineered, combining content-based and collaborative-filtering signals. The most impactful groups were:

- **Within-query relative features** – per-query aggregates (mean, std, min, max) of price, log price, star rating, review score, and location scores, used to derive standardized scores, percentile ranks, and deviations from the query mean (e.g. `price_zscore`, `price_pct_rank`, `cheapest_hotel_flag`).
- **Composite features** adapted from prior competitors – e.g. `ump` (price vs. historical price), `per_person_fee`, `score2ma` (location score × query affinity).
- **Collaborative-filtering signals** – historical click/book rates smoothed with **Bayesian smoothing**, plus per-property aggregates and estimated position per destination. To prevent leakage, these were computed with **Out-of-Fold (OOF) target encoding** on the train split, and over combined train+val data for the test set.
- **Missing-value flags** – explicit boolean indicators for the many >90% NULL columns (competitor data, user history, affinity score), since missingness itself is informative.

Data cleaning included converting string `"NULL"` to true nulls, downcasting to the smallest safe dtypes (~60% memory reduction), conversion to **Parquet**, and quantile-based outlier clipping for `price_usd`, `srch_booking_window`, and `srch_length_of_stay`. Missing values were **not imputed** — the tree-based models handle them natively.

## Results

NDCG@5 across the modeling progression (validation set unless noted):

| Model                          | NDCG@5    | Notes                                          |
|--------------------------------|-----------|------------------------------------------------|
| Logistic Regression (baseline) | 0.329     | C=0.01, L1, booking weight 5                   |
| Pointwise FM                   | 0.3504    | 25 epochs, embed dim 32                         |
| Pairwise FM                    | 0.3800    | plain pairwise loss, increased L2               |
| DeepFM                         | 0.3846    | marginal gain at much higher complexity         |
| **LambdaMART (LightGBM)**      | **0.4255**| 1219 boosting rounds**|

The collaborative-filtering signals dominated the baseline's top coefficients (`prop_dest_book_rate`, `prop_click_rate`, `prop_book_rate`), confirming that historical user behavior and within-search relative competitiveness are the strongest predictors. **LambdaMART** was the clear winner.

## Bias Mitigation

As part of ethical AI assessment, performance was split by **family** (travel parties with children, ≈24%) vs. **non-family** (≈76%) queries. Contrary to the initial hypothesis, the *underrepresented* family group performed **better** (NDCG@5 0.4328 vs. 0.4179), likely because family searches are more homogeneous and predictable.

To close the gap, **non-family rows were re-weighted** (by $n_{\text{non-family}}/n_{\text{family}}$, normalized to mean 1) so the model paid more attention to their errors. This reduced the performance gap from **0.0149 to 0.0103**, at the cost of a small drop in overall NDCG@5 — a direct illustration of the fairness–accuracy trade-off.

## Scalable Deployment

Once trained, LambdaMART is a fixed ensemble of shallow trees, so inference is just a series of if-else comparisons: a single CPU can score millions of hotel–query pairs with no GPU, and the model loads from a single file behind an API. The main deployment challenge is the **feature pipeline**: OOF rates, per-property aggregates, and estimated-position statistics depend on historical data and must be periodically recomputed as prices, availability, and behavior shift. The model also produces static rankings and cannot adapt to within-session signals in real time.

## Repository Structure
```
project-root/
├── LogisticRegression                      # Logisitic Regression folder
├── DeepFM                                  # DeepFM folder
├── LambdaMART                              # LambdaMART folder
        ├── figures/                        # EDA plots, correlation heatmap, distributions
        ├── data/raw/                       # Raw data CSV files
        ├── src/   
            ├── data_loading.py             # Loading csv data files    
            ├── data_preprocessing.py       # NULL handling, downcasting, clipping, CSV-to-Parquet  
            ├── EDA.py                      # Exploratory data analysis & plots  
            ├── feature_engineering.py      # Feature engineering 
            ├── train.py                    # Train LambdaMART model and finetune with Optuna
            ├── final_model.py              # Extract test predictions for final model
            ├── fairness_analysis.py        # Compare family vs. non-family groups and mitigate bias through re-weighing   
├── requirements.txt                  # Dependencies
├── .gitignore                
└── README.md

```
> Adjust the paths above to match your actual repo layout.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Expedia-Hotel-Ranking.git
cd Expedia-Hotel-Ranking
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the data

Download `train.csv` and `test.csv` from the [in-class Kaggle competition](https://www.kaggle.com/competitions/dmt-2026-2nd-assignment) and place them in `data/raw/`.

## Usage

Run the full pipeline:

```bash
python3 main.py
```

This will:
- Clean the data and convert it to Parquet
- Run feature engineering (≈97 features)
- Train the Logistic Regression, DeepFM, and LambdaMART models
- Evaluate with NDCG@5 and run the bias analysis
- Generate a Kaggle submission file

## Configuration

The best-performing LambdaMART configuration found via Optuna:

```yaml
lambdamart:
  boosting_type: gbdt
  num_leaves: 59
  min_data_in_leaf: 1428
  feature_fraction: 0.31
  feature_fraction_bynode: 0.80
  bagging_fraction: 0.72
  bagging_freq: 3
  lambda_l1: 50.16
  lambda_l2: 83.94
  min_gain_to_split: 0.43
  path_smooth: 3.90
  learning_rate: 0.09
  num_boost_round: 1219
  early_stopping_rounds: 150
```

## Authors

Group 8 — Vrije Universiteit Amsterdam, *Data Mining Techniques* (2026):
Kieran Carroll, Jonathan Sierks, Zane Bjornerud.

## References

- Liu, X., et al. (2013). *Combination of diverse ranking models for personalized Expedia hotel searches*. arXiv:1311.7679.
- Burges, C. J. C. (2010). *From RankNet to LambdaRank to LambdaMART: An overview*. Microsoft Research Technical Report MSR-TR-2010-82.
- Rendle, S. (2010). *Factorization Machines*. IEEE ICDM, 995–1000.
- Guo, H., et al. (2017). *DeepFM: A factorization-machine based neural network for CTR prediction*. IJCAI, 1725–1731.
- Ke, G., et al. (2017). *LightGBM: A highly efficient gradient boosting decision tree*. NeurIPS 30, 3146–3154.
- Akiba, T., et al. (2019). *Optuna: A next-generation hyperparameter optimization framework*. KDD '19, 2623–2631.
- Rashmi, K. V., & Gilad-Bachrach, R. (2015). *DART: Dropouts meet multiple additive regression trees*. AISTATS, PMLR 38, 489–497.
- Pejic, I. *Personalize Expedia hotel searches 2013*. [GitHub](https://github.com/igorpejic/personalize_expedia_hotel_searches_2013).
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/stable/)