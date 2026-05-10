import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report

DATA_DIR = "datasets/"

# 1. Load data
train = pd.read_csv(DATA_DIR + "search_stream_training.csv")
val_query = pd.read_csv(DATA_DIR + "click_validation_query.csv")
val_answer = pd.read_csv(DATA_DIR + "click_validation_answer.csv")

searchinfo = pd.read_csv(DATA_DIR + "searchinfo.csv")
userinfo = pd.read_csv(DATA_DIR + "userinfo.csv")
adinfo = pd.read_csv(DATA_DIR + "adinfo.csv")

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

val = val_answer.merge(searchinfo, on="SearchID", how="left")
val = val.merge(userinfo, on="UserID", how="left")
val = val.merge(adinfo, on="AdID", how="left")
val["CategoryMatch"] = (val["SearchCategoryID"] == val["AdCategoryID"]).astype(int)

feature_cols = [
    "Position",
    "HistCTR",
    "Price",
    "IsUserLoggedOn",
    "SearchCategoryID",
    "AdCategoryID",
    "CategoryMatch",
    "UserAgentID",
    "UserAgentOSID",
    "UserDeviceID",
    "UserAgentFamilyID",
]

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
)

# param_grid = {
#     "n_estimators": [100, 300],
#     "max_depth": [3, 4],
#     "learning_rate": [0.03, 0.05, 0.1],
#     "subsample": [0.8, 0.9],
#     "colsample_bytree": [0.8, 0.9],
#     "min_child_weight": [5, 10],
#     "gamma": [0.1, 1]
# }

# param_grid = {
#     "n_estimators": [300],
#     "max_depth": [3],
#     "learning_rate": [0.03],
#     "subsample": [0.8],
#     "colsample_bytree": [0.8],
#     "min_child_weight": [10],
#     "gamma": [1],
#     "reg_alpha": [0.1],
#     "tree_method":["hist"]
# }
param_grid = {
    "n_estimators": [300],
    "max_depth": [4],
    "learning_rate": [0.05],
    "subsample": [0.9],
    "colsample_bytree": [0.9],
    "tree_method":["hist"]
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

print("\n=== Best XGBoost parameters ===")
print(grid_search.best_params_)
print("Best cross-validation AUC:", grid_search.best_score_)
print("Best cross-validation Gini:", 2 * grid_search.best_score_ - 1)

# 6. Predict probability of click, not binary class 0/1
val["click_probability"] = model.predict_proba(X_val)[:, 1]

print("\nExample predicted probabilities:")
print(val[["SearchID", "AdID", "click_probability", "IsClick"]].head(10).to_string(index=False))


# 7. XGBoost evaluated with AUC and Gini
ranks = val["click_probability"].rank(method="average")
pos_ranks_sum = ranks[y_val == 1].sum()

auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
gini = 2 * auc - 1

print("\n=== XGBoost probability model ===")
print("AUC:", auc)
print("Gini:", gini)

# 8. Choose threshold for F1-score

thresholds = np.linspace(0.001, 0.999, 999)

best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:
    y_pred = (val["click_probability"] >= threshold).astype(int)
    f1 = f1_score(y_val, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print("\n=== Best threshold for F1 ===")
print("Best threshold:", best_threshold)
print("Best F1:", best_f1)