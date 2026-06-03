import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import os

DATA_DIR = "datasets/"
OUT_DIR = "outputs_XGBOOST/"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load data
train = pd.read_csv(DATA_DIR + "search_stream_training.csv")
val_query = pd.read_csv(DATA_DIR + "click_validation_query.csv")
val_answer = pd.read_csv(DATA_DIR + "click_validation_answer.csv")

searchinfo = pd.read_csv(DATA_DIR + "searchinfo.csv")
userinfo = pd.read_csv(DATA_DIR + "userinfo.csv")
adinfo = pd.read_csv(DATA_DIR + "adinfo.csv")

searchinfo = searchinfo.rename(columns={"CategoryID": "SearchCategoryID"})
adinfo = adinfo.rename(columns={"CategoryID": "AdCategoryID"})

# embeddings
search_embs = np.load("datasets/searchinfo_text_embs.npy")
ad_embs = np.load("datasets/adinfo_title_embs.npy")

# mappings from IDs to embedding row indices
search_id_to_idx = dict(zip(searchinfo["SearchID"], range(len(searchinfo))))
ad_id_to_idx = dict(zip(adinfo["AdID"], range(len(adinfo))))


def add_embedding_similarity(df):
    search_idx = df["SearchID"].map(search_id_to_idx).values
    ad_idx = df["AdID"].map(ad_id_to_idx).values

    search_vecs = search_embs[search_idx]
    ad_vecs = ad_embs[ad_idx]

    df["SearchAdCosineSim"] = np.sum(search_vecs * ad_vecs, axis=1) / (
        np.linalg.norm(search_vecs, axis=1) * np.linalg.norm(ad_vecs, axis=1) + 1e-8
    )

    return df

print("Train shape:", train.shape)
print("Validation query shape:", val_query.shape)
print("Validation answer shape:", val_answer.shape)

# 3. Add metadata and create simple features
searchinfo = searchinfo.rename(columns={"CategoryID": "SearchCategoryID"})
adinfo = adinfo.rename(columns={"CategoryID": "AdCategoryID"})

train = train.merge(searchinfo, on="SearchID", how="left")
train = train.merge(userinfo, on="UserID", how="left")
train = train.merge(adinfo, on="AdID", how="left")
train["CategoryMatch"] = (train["SearchCategoryID"] == train["AdCategoryID"]).astype(int)
train = add_embedding_similarity(train)

val = val_answer.merge(searchinfo, on="SearchID", how="left")
val = val.merge(userinfo, on="UserID", how="left")
val = val.merge(adinfo, on="AdID", how="left")
val["CategoryMatch"] = (val["SearchCategoryID"] == val["AdCategoryID"]).astype(int)
val = add_embedding_similarity(val)

# UserAgentID - A unique identifier of the user’s browser. - to detailed - no sense in this infromation alone

feature_cols = [
    "Position",
    "HistCTR",
    "Price",
    "IsUserLoggedOn",
    "SearchCategoryID",
    "AdCategoryID",
    "CategoryMatch",
    "SearchAdCosineSim",
    # "UserAgentID",
    "UserAgentOSID",
    "UserDeviceID",
    "UserAgentFamilyID",
]

# defining categorical variables because the differences in values may not have an impact
categorical_cols = ['IsUserLoggedOn',
                    'SearchCategoryID',
                    'AdCategoryID',
                    'CategoryMatch',
                    'UserAgentOSID',
                    'UserDeviceID',
                    'UserAgentFamilyID']


for col in categorical_cols:
    train[col] = train[col].astype("category")
    val[col] = val[col].astype("category")


X_train = train[feature_cols].fillna(-1)
y_train = train["IsClick"].astype(int)

X_val = val[feature_cols].fillna(-1)
y_val = val["IsClick"].astype(int)

print("\nFeatures:")
for col in feature_cols:
    print("-", col)

print("\nPositive rate in train:", y_train.mean())
print("Positive rate in validation:", y_val.mean())


# 4. HistCTR baseline evaluated with AUC and Gini
n_pos = (y_val == 1).sum()
n_neg = (y_val == 0).sum()

baseline_scores = val_query["HistCTR"]
baseline_ranks = baseline_scores.rank(method="average")
baseline_pos_ranks_sum = baseline_ranks[y_val == 1].sum()

baseline_auc = (baseline_pos_ranks_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
baseline_gini = 2 * baseline_auc - 1

print("\n=== HistCTR baseline ===")
print("AUC:", baseline_auc)
print("Gini:", baseline_gini)

thresholds = np.linspace(0.001, 0.999, 999)

best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:
    y_pred = (val["HistCTR"] >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

y_true_ctr = val["IsClick"].astype(int)
y_pred_ctr = (val["HistCTR"] > 0.01).astype(int)
print("F1 from base threshold:", f1_score(y_true_ctr, y_pred_ctr))
print("F1 from changed threshold:", best_f1)


# 5. Train XGBoost model with GridSearchCV
num_pos = y_train.sum()
num_neg = len(y_train) - num_pos
scale_pos_weight_value = num_neg / num_pos

base_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    scale_pos_weight=scale_pos_weight_value,
    random_state=42,
    n_jobs=-1,
    enable_categorical=True
)

param_grid = {
    "n_estimators": [300],
    "max_depth": [5],
    "learning_rate": [0.03],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "min_child_weight": [10, 30, 50],
    "gamma": [1],
    "reg_alpha": [0.1]
}


cv = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=42,
)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    verbose=1,
    n_jobs=-1,
)

print("\nTraining")
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_

importances = pd.DataFrame({
    "feature": X_train.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n=== Feature importances ===")
print(importances.to_string(index=False))

print("\n=== Best XGBoost parameters ===")
print(grid_search.best_params_)
print("Best cross-validation AUC:", grid_search.best_score_)
print("Best cross-validation Gini:", 2 * grid_search.best_score_ - 1)

# 6. Predict probability of click, not binary class 0/1
val["click_probability"] = model.predict_proba(X_val)[:, 1]

# 7. XGBoost evaluated with AUC and Gini
ranks = val["click_probability"].rank(method="average")
pos_ranks_sum = ranks[y_val == 1].sum()

auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
gini = 2 * auc - 1

print("\n=== XGBoost probability model ===")
print("AUC:", auc)
print("Gini:", gini)

# 8. Predict on test set and write output
test_path = DATA_DIR + "click_test_query.csv"

if os.path.exists(test_path):
    print("Predicting test...")
    test_query = pd.read_csv(test_path)

    test = test_query.merge(searchinfo, on="SearchID", how="left")
    test = test.merge(userinfo, on="UserID", how="left")
    test = test.merge(adinfo, on="AdID", how="left")

    test["CategoryMatch"] = (
        test["SearchCategoryID"] == test["AdCategoryID"]
    ).astype(int)

    test = add_embedding_similarity(test)

    for col in categorical_cols:
        if col in test.columns:
            test[col] = test[col].astype("category")

    X_test = test[feature_cols].fillna(-1)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)

    test_answer = test_query.copy()
    test_answer["IsClick"] = test_pred

    output_path = OUT_DIR + "click_test_answer.csv"
    test_answer.to_csv(output_path, index=False)

    print("Wrote:", output_path)
else:
    print("No click_test_query.csv found, skipping test prediction.")