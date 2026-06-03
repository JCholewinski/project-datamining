import pandas as pd
import numpy as np
from lightgbm import LGBMRanker
from tqdm import tqdm

# =========================================================
# LOAD DATA
# =========================================================

train = pd.read_csv("/datasets/search_stream_training.csv")

search = pd.read_csv("/datasets/searchinfo.csv")

ad = pd.read_csv("/datasets/adinfo.csv")

val_query = pd.read_csv("/datasets/ad_validation_query.csv")
val_answer = pd.read_csv("/datasets/ad_validation_answer.csv")

test_query = pd.read_csv("/datasets/ad_test_query.csv")

# =========================================================
# LOAD EMBEDDINGS
# =========================================================

search_emb = np.load("searchinfo_text_embs.npy")
ad_emb = np.load("adinfo_title_embs.npy")

# =========================================================
# BUILD INDEX MAPS
# =========================================================

search_id_to_idx = {
    sid: idx
    for idx, sid in enumerate(search["SearchID"])
}

ad_id_to_idx = {
    aid: idx
    for idx, aid in enumerate(ad["AdID"])
}

# =========================================================
# POSITIVE CLICK DATA ONLY
# =========================================================

clicked = train[train["IsClick"] == 1].copy()

# =========================================================
# JOIN SEARCH + AD INFO
# =========================================================

clicked = clicked.merge(
    search,
    on="SearchID",
    how="left"
)

clicked = clicked.merge(
    ad[["AdID", "CategoryID", "Price"]],
    on="AdID",
    how="left",
    suffixes=("_search", "_ad")
)

# =========================================================
# POPULARITY FEATURES
# =========================================================

ad_popularity = (
    clicked["AdID"]
    .value_counts()
    .to_dict()
)

global_popular_ads = (
    clicked["AdID"]
    .value_counts()
    .head(300)
    .index
    .tolist()
)

# =========================================================
# CATEGORY-WISE POPULAR ADS
# =========================================================

category_popular_ads = (
    clicked.groupby("CategoryID_search")["AdID"]
    .apply(
        lambda x: x.value_counts().head(300).index.tolist()
    )
    .to_dict()
)

# =========================================================
# CANDIDATE GENERATION
# =========================================================

def get_candidates(search_id, topk=100):

    row = search.loc[
        search["SearchID"] == search_id
    ].iloc[0]

    category = row["CategoryID"]

    candidates = category_popular_ads.get(category, []).copy()

    candidates = candidates[:topk]

    if len(candidates) < topk:

        for aid in global_popular_ads:

            if aid not in candidates:
                candidates.append(aid)

            if len(candidates) >= topk:
                break

    return candidates

# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(search_id, ad_id):

    if search_id not in search_id_to_idx:
        return 0.0

    if ad_id not in ad_id_to_idx:
        return 0.0

    sidx = search_id_to_idx[search_id]
    aidx = ad_id_to_idx[ad_id]

    s = search_emb[sidx]
    a = ad_emb[aidx]

    denom = (
        np.linalg.norm(s)
        * np.linalg.norm(a)
        + 1e-9
    )

    return np.dot(s, a) / denom

# =========================================================
# BUILD TRAINING DATA
# =========================================================

train_rows = []

unique_clicked = (
    clicked[["SearchID", "AdID"]]
    .drop_duplicates()
)

for _, row in tqdm(
    unique_clicked.iterrows(),
    total=len(unique_clicked)
):

    search_id = row["SearchID"]
    true_ad = row["AdID"]

    candidates = get_candidates(search_id, topk=100)

    if true_ad not in candidates:
        candidates = [true_ad] + candidates[:-1]

    for aid in candidates:

        train_rows.append({
            "SearchID": search_id,
            "AdID": aid,
            "label": int(aid == true_ad)
        })

# =========================================================
# TRAIN DF
# =========================================================

rank_df = pd.DataFrame(train_rows)

# =========================================================
# MERGE FEATURES
# =========================================================

rank_df = rank_df.merge(
    search,
    on="SearchID",
    how="left"
)

rank_df = rank_df.merge(
    ad[["AdID", "CategoryID", "Price"]],
    on="AdID",
    how="left",
    suffixes=("_search", "_ad")
)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

print("computing cosine similarities...")

rank_df["cosine"] = rank_df.apply(
    lambda x: cosine_similarity(
        x["SearchID"],
        x["AdID"]
    ),
    axis=1
)

rank_df["category_match"] = (
    rank_df["CategoryID_search"]
    == rank_df["CategoryID_ad"]
).astype(int)

rank_df["ad_popularity"] = (
    rank_df["AdID"]
    .map(ad_popularity)
    .fillna(0)
)

rank_df["log_price"] = np.log1p(
    rank_df["Price"].fillna(0)
)

# =========================================================
# FEATURE LIST
# =========================================================

features = [
    "UserID",
    "IPID",
    "IsUserLoggedOn",
    "CategoryID_search",
    "CategoryID_ad",
    "cosine",
    "category_match",
    "ad_popularity",
    "log_price"
]

# =========================================================
# SORT BY QUERY
# =========================================================

rank_df = rank_df.sort_values("SearchID")

X_train = rank_df[features]
y_train = rank_df["label"]

group = (
    rank_df.groupby("SearchID")
    .size()
    .values
)

# =========================================================
# LIGHTGBM RANKER
# =========================================================

model = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="gbdt",
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# =========================================================
# TRAIN
# =========================================================

print("training model...")

model.fit(
    X_train,
    y_train,
    group=group
)

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_top3(search_id):

    candidates = get_candidates(
        search_id,
        topk=100
    )

    tmp = pd.DataFrame({
        "SearchID": [search_id] * len(candidates),
        "AdID": candidates
    })

    tmp = tmp.merge(
        search,
        on="SearchID",
        how="left"
    )

    tmp = tmp.merge(
        ad[["AdID", "CategoryID", "Price"]],
        on="AdID",
        how="left",
        suffixes=("_search", "_ad")
    )

    tmp["cosine"] = tmp.apply(
        lambda x: cosine_similarity(
            x["SearchID"],
            x["AdID"]
        ),
        axis=1
    )

    tmp["category_match"] = (
        tmp["CategoryID_search"]
        == tmp["CategoryID_ad"]
    ).astype(int)

    tmp["ad_popularity"] = (
        tmp["AdID"]
        .map(ad_popularity)
        .fillna(0)
    )

    tmp["log_price"] = np.log1p(
        tmp["Price"].fillna(0)
    )

    tmp["score"] = model.predict(
        tmp[features]
    )

    top3 = (
        tmp.sort_values(
            "score",
            ascending=False
        )["AdID"]
        .drop_duplicates()
        .head(3)
        .tolist()
    )

    while len(top3) < 3:

        for aid in global_popular_ads:

            if aid not in top3:
                top3.append(aid)

            if len(top3) >= 3:
                break

    return top3

# =========================================================
# VALIDATION PREDICTIONS
# =========================================================

print("predicting validation set...")

val_preds = []

for sid in tqdm(val_query["SearchID"]):

    top3 = predict_top3(sid)

    val_preds.append([
        sid,
        top3[0],
        top3[1],
        top3[2]
    ])

val_pred_df = pd.DataFrame(
    val_preds,
    columns=[
        "SearchID",
        "AdID 1",
        "AdID 2",
        "AdID 3"
    ]
)

val_pred_df.to_csv(
    "ad_validation_pred.csv",
    index=False
)

# =========================================================
# NDCG@3
# =========================================================

merged = val_pred_df.merge(
    val_answer,
    on="SearchID",
    how="left"
)
# =========================================================
# EXTRA EVALUATION METRICS FOR TASK 2
# =========================================================

def evaluate_ranking_metrics(merged):
    ndcgs = []
    reciprocal_ranks = []

    hit1 = 0
    hit3 = 0

    rank1 = 0
    rank2 = 0
    rank3 = 0
    miss = 0

    for _, row in merged.iterrows():
        true_ad = row["AdID"]

        preds = [
            row["AdID 1"],
            row["AdID 2"],
            row["AdID 3"]
        ]

        found = False

        for r, pred in enumerate(preds, start=1):
            if pred == true_ad:
                ndcgs.append(1.0 / np.log2(r + 1))
                reciprocal_ranks.append(1.0 / r)

                if r == 1:
                    hit1 += 1
                    rank1 += 1
                elif r == 2:
                    rank2 += 1
                elif r == 3:
                    rank3 += 1

                hit3 += 1
                found = True
                break

        if not found:
            ndcgs.append(0.0)
            reciprocal_ranks.append(0.0)
            miss += 1

    n = len(merged)

    print("\n========== TASK 2 VALIDATION METRICS ==========")
    print(f"NDCG@3 : {np.mean(ndcgs):.6f}")
    print(f"MRR    : {np.mean(reciprocal_ranks):.6f}")
    print(f"Hits@1 : {hit1 / n:.6f}")
    print(f"Hits@3 : {hit3 / n:.6f}")
    print("----------------------------------------------")
    print(f"Rank 1 correct : {rank1}")
    print(f"Rank 2 correct : {rank2}")
    print(f"Rank 3 correct : {rank3}")
    print(f"Missed         : {miss}")
    print(f"Total queries  : {n}")
    print("==============================================\n")


evaluate_ranking_metrics(merged)
def ndcg3(row):

    true_ad = row["AdID"]

    preds = [
        row["AdID 1"],
        row["AdID 2"],
        row["AdID 3"]
    ]

    for rank, pred in enumerate(preds, start=1):

        if pred == true_ad:
            return 1 / np.log2(rank + 1)

    return 0

score = merged.apply(
    ndcg3,
    axis=1
).mean()

print("\n==========================")
print("Validation NDCG@3:", score)
print("==========================\n")

# =========================================================
# TEST PREDICTION
# =========================================================

print("predicting test set...")

test_preds = []

for sid in tqdm(test_query["SearchID"]):

    top3 = predict_top3(sid)

    test_preds.append([
        sid,
        top3[0],
        top3[1],
        top3[2]
    ])

test_pred_df = pd.DataFrame(
    test_preds,
    columns=[
        "SearchID",
        "AdID 1",
        "AdID 2",
        "AdID 3"
    ]
)

# =========================================================
# SAVE SUBMISSION
# =========================================================

test_pred_df.to_csv(
    "ad_test_answer.csv",
    index=False
)

print("saved: ad_test_answer.csv")
