import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from tqdm import tqdm


def id_map(values):
    values = pd.Series(values).drop_duplicates().tolist()
    return {v: i for i, v in enumerate(values)}


def map_series(s, mapping):
    out = s.map(mapping)
    if out.isna().any():
        missing = s[out.isna()].unique()[:10]
        raise ValueError(f"Missing mapping values: {missing}")
    return out.astype("int64").to_numpy()


def safe_read_csv(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def edge_index_from(src, dst):
    return torch.from_numpy(np.vstack([src, dst])).long()


def load_tables(data_dir: Path):
    tables = {
        "users": safe_read_csv(data_dir / "userinfo.csv"),
        "ads": safe_read_csv(data_dir / "adinfo.csv"),
        "searches": safe_read_csv(data_dir / "searchinfo.csv"),
        "train": safe_read_csv(data_dir / "search_stream_training.csv"),
        "val_q": safe_read_csv(data_dir / "click_validation_query.csv"),
        "val_a": safe_read_csv(data_dir / "click_validation_answer.csv"),
        "test_q": safe_read_csv(data_dir / "click_test_query.csv"),
    }

    tables["ad_emb"] = np.load(data_dir / "adinfo_title_embs.npy").astype(np.float32)
    tables["search_emb"] = np.load(data_dir / "searchinfo_text_embs.npy").astype(np.float32)

    return tables


def build_heterodata(tables):
    users = tables["users"].copy()
    ads = tables["ads"].copy()
    searches = tables["searches"].copy()
    train = tables["train"].copy()
    val_q = tables["val_q"].copy()
    test_q = tables["test_q"].copy()

    user2idx = id_map(users["UserID"])

    ad2idx = id_map(pd.concat([
        ads["AdID"],
        train["AdID"],
        val_q["AdID"],
        test_q["AdID"],
    ]))

    search2idx = id_map(pd.concat([
        searches["SearchID"],
        train["SearchID"],
        val_q["SearchID"],
        test_q["SearchID"],
    ]))

    user_x = users[
        ["UserAgentID", "UserAgentOSID", "UserDeviceID", "UserAgentFamilyID"]
    ].fillna(-1).astype(np.float32).to_numpy()
    user_x = StandardScaler().fit_transform(user_x).astype(np.float32)

    ad_meta = ads[["CategoryID", "Price"]].fillna(0).astype(np.float32).to_numpy()
    ad_meta = StandardScaler().fit_transform(ad_meta).astype(np.float32)
    ad_x = np.concatenate([ad_meta, tables["ad_emb"]], axis=1)

    search_meta = searches[
        ["IPID", "IsUserLoggedOn", "CategoryID"]
    ].fillna(0).astype(np.float32).to_numpy()
    search_meta = StandardScaler().fit_transform(search_meta).astype(np.float32)
    search_x = np.concatenate([search_meta, tables["search_emb"]], axis=1)

    data = HeteroData()
    data["user"].x = torch.tensor(user_x, dtype=torch.float32)
    data["ad"].x = torch.tensor(ad_x, dtype=torch.float32)
    data["search"].x = torch.tensor(search_x, dtype=torch.float32)

    search_user = searches.dropna(subset=["UserID", "SearchID"])

    us_src = map_series(search_user["UserID"], user2idx)
    us_dst = map_series(search_user["SearchID"], search2idx)

    data[("user", "requests", "search")].edge_index = edge_index_from(us_src, us_dst)
    data[("search", "rev_requests", "user")].edge_index = edge_index_from(us_dst, us_src)

    sa_src = map_series(train["SearchID"], search2idx)
    sa_dst = map_series(train["AdID"], ad2idx)

    data[("search", "shows", "ad")].edge_index = edge_index_from(sa_src, sa_dst)
    data[("ad", "rev_shows", "search")].edge_index = edge_index_from(sa_dst, sa_src)

    mappings = {
        "user2idx": user2idx,
        "ad2idx": ad2idx,
        "search2idx": search2idx,
    }

    return data, mappings


def make_edge_frame(df, mappings, tables):
    out = df.copy()

    out["search_idx"] = out["SearchID"].map(mappings["search2idx"])
    out["ad_idx"] = out["AdID"].map(mappings["ad2idx"])

    search_row_map = {
        sid: i for i, sid in enumerate(tables["searches"]["SearchID"])
    }

    ad_row_map = {
        aid: i for i, aid in enumerate(tables["ads"]["AdID"])
    }

    out["search_row"] = out["SearchID"].map(search_row_map)
    out["ad_row"] = out["AdID"].map(ad_row_map)

    if out[["search_idx", "ad_idx", "search_row", "ad_row"]].isna().any().any():
        raise ValueError("Found unmapped SearchID or AdID.")

    edge_index = torch.from_numpy(
        out[["search_idx", "ad_idx"]].to_numpy().T
    ).long()

    return out, edge_index


def edge_features(df, tables, scaler=None, fit=False):
    out = pd.DataFrame()

    out["position"] = df["Position"].astype(np.float32)
    out["hist_ctr"] = df["HistCTR"].astype(np.float32)
    out["log_hist_ctr"] = np.log1p(df["HistCTR"]).astype(np.float32)

    out["ctr_per_pos"] = (
        df["HistCTR"] / (df["Position"] + 1)
    ).astype(np.float32)

    out["is_top1"] = (df["Position"] == 1).astype(np.float32)
    out["is_top3"] = (df["Position"] <= 3).astype(np.float32)

    search_emb = tables["search_emb"][
        df["search_row"].to_numpy(dtype=np.int64)
    ]

    ad_emb = tables["ad_emb"][
        df["ad_row"].to_numpy(dtype=np.int64)
    ]

    cosine = (
        np.sum(search_emb * ad_emb, axis=1)
        /
        (
            np.linalg.norm(search_emb, axis=1)
            * np.linalg.norm(ad_emb, axis=1)
            + 1e-8
        )
    )

    out["semantic_sim"] = cosine.astype(np.float32)

    abs_diff = np.abs(search_emb - ad_emb).astype(np.float32)
    hadamard = (search_emb * ad_emb).astype(np.float32)

    base_feat = out.to_numpy(dtype=np.float32)

    X = np.concatenate([
        base_feat,
        abs_diff,
        hadamard,
    ], axis=1).astype(np.float32)

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit=False")
        X = scaler.transform(X).astype(np.float32)

    return torch.tensor(X, dtype=torch.float32), scaler


class Task1HeteroGNN(nn.Module):
    def __init__(self, metadata, edge_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()

        self.proj = nn.ModuleDict({
            "user": Linear(-1, hidden_dim),
            "search": Linear(-1, hidden_dim),
            "ad": Linear(-1, hidden_dim),
        })

        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                HeteroConv({
                    edge_type: SAGEConv((-1, -1), hidden_dim)
                    for edge_type in metadata[1]
                }, aggr="sum")
            )

        decoder_dim = 512

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, decoder_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(decoder_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

        self.dropout = dropout

    def encode(self, x_dict, edge_index_dict):
        x_dict = {
            k: F.relu(self.proj[k](x))
            for k, x in x_dict.items()
        }

        for conv in self.convs:
            new_x_dict = conv(x_dict, edge_index_dict)

            x_dict = {
                k: F.dropout(
                    F.relu(new_x_dict[k]),
                    p=self.dropout,
                    training=self.training,
                )
                for k in new_x_dict
            }

        return x_dict

    def decode(self, z_dict, edge_label_index, edge_feat):
        search_idx, ad_idx = edge_label_index

        z = torch.cat([
            z_dict["search"][search_idx],
            z_dict["ad"][ad_idx],
            edge_feat,
        ], dim=1)

        return self.edge_mlp(z).view(-1)

    def forward(self, data, edge_label_index, edge_feat):
        z_dict = self.encode(data.x_dict, data.edge_index_dict)
        return self.decode(z_dict, edge_label_index, edge_feat)


class HybridF1RankingLoss(nn.Module):
    def __init__(self, pos_weight, alpha=0.7, margin=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.alpha = alpha
        self.margin = margin

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        pos_logits = logits[targets == 1]
        neg_logits = logits[targets == 0]

        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return bce_loss

        n = min(len(pos_logits), len(neg_logits))

        pos_sample = pos_logits[
            torch.randint(0, len(pos_logits), (n,), device=logits.device)
        ]

        neg_sample = neg_logits[
            torch.randint(0, len(neg_logits), (n,), device=logits.device)
        ]

        rank_loss = F.relu(
            self.margin - pos_sample + neg_sample
        ).mean()

        return self.alpha * bce_loss + (1.0 - self.alpha) * rank_loss


@torch.no_grad()
def predict_logits(model, data, edge_index, edge_feat, device):
    model.eval()
    return model(
        data,
        edge_index.to(device),
        edge_feat.to(device),
    ).cpu()


def best_threshold(y_true, prob):
    candidates = np.linspace(0.01, 0.99, 99)

    scores = [
        f1_score(y_true, prob >= t, zero_division=0)
        for t in candidates
    ]

    idx = int(np.argmax(scores))
    return float(candidates[idx]), float(scores[idx])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_tables(args.data_dir)
    data, mappings = build_heterodata(tables)

    train_df, train_edge_index = make_edge_frame(tables["train"], mappings, tables)
    val_df, val_edge_index = make_edge_frame(tables["val_a"], mappings, tables)
    test_df, test_edge_index = make_edge_frame(tables["test_q"], mappings, tables)

    train_feat, scaler = edge_features(train_df, tables, fit=True)
    val_feat, _ = edge_features(val_df, tables, scaler=scaler, fit=False)
    test_feat, _ = edge_features(test_df, tables, scaler=scaler, fit=False)

    y_train = torch.tensor(
        train_df["IsClick"].to_numpy(),
        dtype=torch.float32,
    )

    y_val = val_df["IsClick"].to_numpy().astype(int)

    pos = max(float(y_train.sum().item()), 1.0)
    neg = max(float(len(y_train) - pos), 1.0)

    pos_weight = torch.tensor(
        [neg / pos],
        dtype=torch.float32,
    )

    device = torch.device(args.device)

    data = data.to(device)
    train_edge_index = train_edge_index.to(device)
    train_feat = train_feat.to(device)
    y_train = y_train.to(device)
    pos_weight = pos_weight.to(device)

    model = Task1HeteroGNN(
        data.metadata(),
        edge_dim=train_feat.shape[1],
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
    )

    criterion = HybridF1RankingLoss(
        pos_weight=pos_weight,
        alpha=0.7,
    )

    best_val_f1 = -1.0
    best_state = None
    best_t = 0.5
    patience_counter = 0

    for epoch in tqdm(range(1, args.epochs + 1), desc="training"):
        model.train()
        optimizer.zero_grad()

        logits = model(data, train_edge_index, train_feat)
        loss = criterion(logits, y_train)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        val_logits = predict_logits(
            model,
            data,
            val_edge_index,
            val_feat,
            device,
        )

        val_prob = torch.sigmoid(val_logits).numpy()

        threshold, val_f1 = best_threshold(y_val, val_prob)

        try:
            val_auc = roc_auc_score(y_val, val_prob)
        except ValueError:
            val_auc = float("nan")

        scheduler.step(val_f1)

        improved = val_f1 > best_val_f1

        if improved:
            best_val_f1 = val_f1
            best_t = threshold
            patience_counter = 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            torch.save(best_state, args.out_dir / "best_model.pt")

        else:
            patience_counter += 1

        print(
            f"epoch={epoch:03d} "
            f"loss={loss.item():.4f} "
            f"val_f1={val_f1:.4f} "
            f"best_f1={best_val_f1:.4f} "
            f"val_auc={val_auc:.4f} "
            f"threshold={threshold:.2f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"patience={patience_counter}/{args.patience} "
            f"prob_mean={val_prob.mean():.4f} "
            f"prob_std={val_prob.std():.4f} "
            f"prob_max={val_prob.max():.4f}"
        )

        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    threshold = args.threshold if args.threshold is not None else best_t

    test_logits = predict_logits(
        model,
        data,
        test_edge_index,
        test_feat,
        device,
    )

    test_prob = torch.sigmoid(test_logits).numpy()
    test_pred = (test_prob >= threshold).astype(int)

    submission = tables["test_q"].copy()
    submission["IsClick"] = test_pred

    out_path = args.out_dir / "click_test_answer.csv"

    submission[
        [
            "SearchID",
            "AdID",
            "Position",
            "HistCTR",
            "IsClick",
        ]
    ].to_csv(out_path, index=False)

    print(
        f"Best validation F1: {best_val_f1:.4f} "
        f"at threshold={best_t:.2f}"
    )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
