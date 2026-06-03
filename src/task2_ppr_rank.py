import math
import os
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


DATA_DIR = "datasets/"
OUT_DIR = "outputs_PageRank/"

SEMANTIC_TOP_K = 200
SEMANTIC_EDGE_SCALE = 6.0

SEARCH_WEIGHT = 0.60
USER_WEIGHT = 0.25
CATEGORY_WEIGHT = 0.1
USER_CATEGORY_WEIGHT = 0.05

ALPHA = 0.85
MAX_ITER = 100
TOL = 1e-8

CLICK_EDGE_SCALE = 1.0
CATEGORY_EDGE_SCALE = 1.0
USER_CATEGORY_EDGE_SCALE = 1.5
STRUCTURAL_EDGE_WEIGHT = 0.2

RERANK_PPR_WEIGHT = 0.5
RERANK_COSINE_WEIGHT = 0.5
RERANK_CATEGORY_POP_WEIGHT = 0.0


PPR_CANDIDATE_TOP_K = 1000
CATEGORY_POP_TOP_K = 300
GLOBAL_POP_TOP_K = 100


os.makedirs(OUT_DIR, exist_ok=True)


def read_csv(filename):
    return pd.read_csv(DATA_DIR + filename)


def load_embeddings(possible_names):
    for name in possible_names:
        path = DATA_DIR + name
        if os.path.exists(path):
            print("Loaded embeddings:", path)
            try:
                emb = np.load(path)
            except ValueError:
                emb = np.load(path, allow_pickle=True)
            if emb.dtype == object:
                emb = np.array(list(emb), dtype=np.float32)
            return emb.astype(np.float32)
    raise FileNotFoundError("Embedding file not found: " + str(possible_names))


def normalize_rows(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return x / norm


def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    low = values.min()
    high = values.max()
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def add_edge(G, source, target, weight):
    if G.has_edge(source, target):
        G[source][target]["weight"] += float(weight)
    else:
        G.add_edge(source, target, weight=float(weight))


def add_bi_edge(G, node_a, node_b, weight):
    add_edge(G, node_a, node_b, weight)
    add_edge(G, node_b, node_a, weight)


def print_validation_metrics(pred_df, answer_df):
    answer = dict(zip(answer_df["SearchID"].astype(int), answer_df["AdID"].astype(int)))

    ndcgs = []
    reciprocal_ranks = []
    rank1 = 0
    rank2 = 0
    rank3 = 0
    miss = 0

    for _, row in pred_df.iterrows():
        true_ad = answer[int(row["SearchID"])]
        pred_ads = [int(row["AdID 1"]), int(row["AdID 2"]), int(row["AdID 3"])]

        found_rank = None
        for rank, ad_id in enumerate(pred_ads, start=1):
            if ad_id == true_ad:
                found_rank = rank
                break

        if found_rank == 1:
            rank1 += 1
        elif found_rank == 2:
            rank2 += 1
        elif found_rank == 3:
            rank3 += 1
        else:
            miss += 1

        if found_rank is None:
            ndcgs.append(0.0)
            reciprocal_ranks.append(0.0)
        else:
            ndcgs.append(1.0 / math.log2(found_rank + 1))
            reciprocal_ranks.append(1.0 / found_rank)

    n = len(pred_df)
    hit1 = rank1
    hit3 = rank1 + rank2 + rank3

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


print("Loading data...")
train = read_csv("search_stream_training.csv")
searchinfo = read_csv("searchinfo.csv")
adinfo = read_csv("adinfo.csv")
val_query = read_csv("ad_validation_query.csv")
val_answer = read_csv("ad_validation_answer.csv")

ad_emb = load_embeddings(["adinfo_title_embs.npy", "adinfo title embs.npy", "adinfo_title_emb.npy"])
search_emb = load_embeddings(["searchinfo_text_embs.npy", "searchinfo text embs.npy", "searchinfo_text_emb.npy"])

searchinfo = searchinfo.rename(columns={"CategoryID": "SearchCategoryID"})
adinfo = adinfo.rename(columns={"CategoryID": "AdCategoryID"})

if len(adinfo) != len(ad_emb) or len(searchinfo) != len(search_emb):
    raise ValueError("Embedding row count does not match metadata row count")

ad_emb = normalize_rows(ad_emb)
search_emb = normalize_rows(search_emb)

ad_ids = adinfo["AdID"].astype(int).to_numpy()
search_ids = searchinfo["SearchID"].astype(int).to_numpy()

ad_row = {int(ad_id): i for i, ad_id in enumerate(ad_ids)}
search_row = {int(search_id): i for i, search_id in enumerate(search_ids)}
search_lookup = searchinfo.set_index("SearchID")

train = train.merge(searchinfo[["SearchID", "UserID", "SearchCategoryID"]], on="SearchID", how="left")
train = train.merge(adinfo[["AdID", "AdCategoryID"]], on="AdID", how="left")
clicked = train[train["IsClick"] == 1].copy()

print("Clicked train rows:", len(clicked))

user_ad_counts = Counter()
category_ad_counts = Counter()
user_category_counts = Counter()
user_category_ad_counts = Counter()

for _, row in clicked.iterrows():
    if pd.isna(row["UserID"]) or pd.isna(row["SearchCategoryID"]) or pd.isna(row["AdID"]):
        continue

    user_id = int(row["UserID"])
    category_id = int(row["SearchCategoryID"])
    ad_id = int(row["AdID"])

    user_ad_counts[(user_id, ad_id)] += 1
    category_ad_counts[(category_id, ad_id)] += 1
    user_category_counts[(user_id, category_id)] += 1
    user_category_ad_counts[(user_id, category_id, ad_id)] += 1

category_popular = defaultdict(list)
for (category_id, ad_id), count in category_ad_counts.items():
    category_popular[category_id].append((ad_id, count))

for category_id in category_popular:
    category_popular[category_id].sort(key=lambda x: x[1], reverse=True)
    category_popular[category_id] = [ad_id for ad_id, _ in category_popular[category_id]]

global_popular = clicked["AdID"].astype(int).value_counts().index.tolist()
if len(global_popular) < 3:
    global_popular = adinfo["AdID"].astype(int).head(3).tolist()

print("Building graph...")
G = nx.DiGraph()

for _, row in adinfo.iterrows():
    ad_id = int(row["AdID"])
    category_id = int(row["AdCategoryID"]) if not pd.isna(row["AdCategoryID"]) else -1
    add_bi_edge(G, f"ad_{ad_id}", f"category_{category_id}", STRUCTURAL_EDGE_WEIGHT)

for (user_id, ad_id), count in user_ad_counts.items():
    add_bi_edge(G, f"user_{user_id}", f"ad_{ad_id}", CLICK_EDGE_SCALE * math.log1p(count))

for (category_id, ad_id), count in category_ad_counts.items():
    add_bi_edge(
        G,
        f"category_{category_id}",
        f"ad_{ad_id}",
        CATEGORY_EDGE_SCALE * math.log1p(count),
    )

for (user_id, category_id), _ in user_category_counts.items():
    user_node = f"user_{user_id}"
    category_node = f"category_{category_id}"
    user_category_node = f"usercat_{user_id}_{category_id}"
    add_bi_edge(G, user_node, user_category_node, STRUCTURAL_EDGE_WEIGHT)
    add_bi_edge(G, category_node, user_category_node, STRUCTURAL_EDGE_WEIGHT)

for (user_id, category_id, ad_id), count in user_category_ad_counts.items():
    add_bi_edge(
        G,
        f"usercat_{user_id}_{category_id}",
        f"ad_{ad_id}",
        USER_CATEGORY_EDGE_SCALE * math.log1p(count),
    )

print("Graph nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())


def semantic_candidates(search_id):
    if search_id not in search_row:
        return [], np.array([], dtype=np.float32)

    sims = ad_emb @ search_emb[search_row[search_id]]
    k = min(SEMANTIC_TOP_K, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [int(ad_ids[i]) for i in idx], sims[idx]


def run_pagerank(Gq, personalization):
    try:
        return nx.pagerank(
            Gq,
            alpha=ALPHA,
            personalization=personalization,
            weight="weight",
            max_iter=MAX_ITER,
            tol=TOL,
        )
    except nx.PowerIterationFailedConvergence:
        return nx.pagerank(
            Gq,
            alpha=ALPHA,
            personalization=personalization,
            weight="weight",
            max_iter=MAX_ITER * 3,
            tol=TOL * 10,
        )


def predict_one(search_id):
    search_id = int(search_id)
    if search_id not in search_lookup.index:
        return global_popular[:3]

    row = search_lookup.loc[search_id]
    user_id = int(row["UserID"]) if not pd.isna(row["UserID"]) else -1
    category_id = int(row["SearchCategoryID"]) if not pd.isna(row["SearchCategoryID"]) else -1

    search_node = f"search_{search_id}"
    user_node = f"user_{user_id}"
    category_node = f"category_{category_id}"
    user_category_node = f"usercat_{user_id}_{category_id}"

    Gq = G.copy()
    add_bi_edge(Gq, search_node, user_node, STRUCTURAL_EDGE_WEIGHT)
    add_bi_edge(Gq, search_node, category_node, STRUCTURAL_EDGE_WEIGHT)
    add_bi_edge(Gq, search_node, user_category_node, STRUCTURAL_EDGE_WEIGHT)
    add_bi_edge(Gq, user_node, user_category_node, STRUCTURAL_EDGE_WEIGHT)
    add_bi_edge(Gq, category_node, user_category_node, STRUCTURAL_EDGE_WEIGHT)

    sem_ads, sem_sims = semantic_candidates(search_id)
    for ad_id, sim in zip(sem_ads, sem_sims):
        weight = SEMANTIC_EDGE_SCALE * max((float(sim) + 1.0) / 2.0, 1e-6)
        add_bi_edge(Gq, search_node, f"ad_{ad_id}", weight)

    personalization = {
        search_node: SEARCH_WEIGHT,
        user_node: USER_WEIGHT,
        category_node: CATEGORY_WEIGHT,
        user_category_node: USER_CATEGORY_WEIGHT,
    }
    total = sum(personalization.values())
    personalization = {node: value / total for node, value in personalization.items() if value > 0}

    pr = run_pagerank(Gq, personalization)

    ppr_ads = [(int(ad_id), pr.get(f"ad_{int(ad_id)}", 0.0)) for ad_id in ad_ids]
    ppr_ads.sort(key=lambda x: x[1], reverse=True)

    candidates = []
    seen = set()

    def add_candidate(ad_id):
        ad_id = int(ad_id)
        if ad_id not in seen:
            seen.add(ad_id)
            candidates.append(ad_id)

    for ad_id in sem_ads:
        add_candidate(ad_id)
    for ad_id, _ in ppr_ads[:PPR_CANDIDATE_TOP_K]:
        add_candidate(ad_id)
    for ad_id in category_popular.get(category_id, [])[:CATEGORY_POP_TOP_K]:
        add_candidate(ad_id)
    for ad_id in global_popular[:GLOBAL_POP_TOP_K]:
        add_candidate(ad_id)

    if len(candidates) == 0:
        candidates = global_popular[:3]

    sims = ad_emb @ search_emb[search_row[search_id]] if search_id in search_row else np.zeros(len(ad_ids))

    ppr_values = []
    cosine_values = []
    category_pop_values = []

    for ad_id in candidates:
        ppr_values.append(pr.get(f"ad_{ad_id}", 0.0))
        cosine_values.append(float(sims[ad_row[ad_id]]) if ad_id in ad_row else 0.0)
        category_pop_values.append(math.log1p(category_ad_counts.get((category_id, ad_id), 0)))

    final_score = (
        RERANK_PPR_WEIGHT * minmax(ppr_values)
        + RERANK_COSINE_WEIGHT * minmax(cosine_values)
        + RERANK_CATEGORY_POP_WEIGHT * minmax(category_pop_values)
    )

    ranked = np.argsort(-final_score)
    top_ads = []
    for idx in ranked:
        ad_id = int(candidates[int(idx)])
        if ad_id not in top_ads:
            top_ads.append(ad_id)
        if len(top_ads) == 3:
            break

    for ad_id in global_popular:
        if ad_id not in top_ads:
            top_ads.append(int(ad_id))
        if len(top_ads) == 3:
            break

    return top_ads[:3]


def predict_dataframe(query_df, output_path):
    rows = []
    for search_id in tqdm(query_df["SearchID"].astype(int), desc="predicting"):
        ad1, ad2, ad3 = predict_one(search_id)
        rows.append({"SearchID": int(search_id), "AdID 1": ad1, "AdID 2": ad2, "AdID 3": ad3})

    pred = pd.DataFrame(rows)
    pred.to_csv(output_path, index=False)
    print("Wrote:", output_path)
    return pred


print("Evaluating validation...")
val_pred = predict_dataframe(val_query, OUT_DIR + "ad_validation_answer.csv")
print_validation_metrics(val_pred, val_answer)

test_path = DATA_DIR + "ad_test_query.csv"
if os.path.exists(test_path):
    print("Predicting test...")
    test_query = pd.read_csv(test_path)
    predict_dataframe(test_query, OUT_DIR + "ad_test_answer.csv")
