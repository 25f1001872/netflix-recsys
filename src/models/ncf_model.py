"""
ncf_model.py
------------
Neural Collaborative Filtering (NCF) — He et al., 2017.
"""

import logging
import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.base_model import BaseRecommender

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# PyTorch Model Definition (Global Scope for Pickle Compatibility)
# ──────────────────────────────────────────────

class NeuMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int, layers: list, dropout: float):
        super().__init__()
        # GMF embeddings
        self.gmf_user_emb = nn.Embedding(n_users, embedding_dim)
        self.gmf_item_emb = nn.Embedding(n_items, embedding_dim)
        # MLP embeddings
        self.mlp_user_emb = nn.Embedding(n_users, embedding_dim)
        self.mlp_item_emb = nn.Embedding(n_items, embedding_dim)

        # MLP tower
        mlp_in = embedding_dim * 2
        mlp_layers = []
        for out_size in layers:
            mlp_layers += [
                nn.Linear(mlp_in, out_size),
                nn.BatchNorm1d(out_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            mlp_in = out_size
        self.mlp_tower = nn.Sequential(*mlp_layers)

        # Final prediction
        self.output = nn.Linear(embedding_dim + layers[-1], 1)

        # Init weights
        nn.init.normal_(self.gmf_user_emb.weight, std=0.01)
        nn.init.normal_(self.gmf_item_emb.weight, std=0.01)
        nn.init.normal_(self.mlp_user_emb.weight, std=0.01)
        nn.init.normal_(self.mlp_item_emb.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        # GMF path
        gmf_u = self.gmf_user_emb(user_ids)
        gmf_i = self.gmf_item_emb(item_ids)
        gmf_out = gmf_u * gmf_i  # element-wise

        # MLP path
        mlp_u = self.mlp_user_emb(user_ids)
        mlp_i = self.mlp_item_emb(item_ids)
        mlp_in = torch.cat([mlp_u, mlp_i], dim=1)
        mlp_out = self.mlp_tower(mlp_in)

        # NeuMF combination
        concat = torch.cat([gmf_out, mlp_out], dim=1)
        out = self.output(concat).squeeze(1)
        return out


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class RatingDataset:
    """PyTorch Dataset for (user_idx, movie_idx, rating) triples."""

    def __init__(self, user_idx, movie_idx, ratings):
        self.users = torch.tensor(user_idx, dtype=torch.long)
        self.items = torch.tensor(movie_idx, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


# ──────────────────────────────────────────────
# NCF Recommender Wrapper
# ──────────────────────────────────────────────

class NCFRecommender(BaseRecommender):
    """
    Neural Collaborative Filtering (NeuMF) recommender wrapper.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        layers: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 1024,
        epochs: int = 20,
        device: Optional[str] = None,
    ):
        super().__init__(name="NCF")
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.layers = layers or [256, 128, 64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._encoder = None

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        encoder=None,
        **kwargs,
    ) -> "NCFRecommender":
        from torch.utils.data import DataLoader

        logger.info(f"Training NCF on {self.device}: embedding={self.embedding_dim}, "
                    f"layers={self.layers}, epochs={self.epochs}")

        self._encoder = encoder
        self._store_train_data(train_df)

        # Build model straight from global scope class
        self._model = NeuMF(
            self.n_users, self.n_items, self.embedding_dim, self.layers, self.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5
        )
        criterion = nn.MSELoss()

        # Datasets
        train_dataset = RatingDataset(
            train_df["user_idx"].values,
            train_df["movie_idx"].values,
            train_df["rating"].values,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        val_loader = None
        if val_df is not None and len(val_df) > 0:
            val_dataset = RatingDataset(
                val_df["user_idx"].values,
                val_df["movie_idx"].values,
                val_df["rating"].values,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size * 4, shuffle=False, num_workers=0
            )

        # Training loop
        self.train_losses = []
        self.val_rmses = []
        best_val_rmse = float("inf")
        best_state = None

        for epoch in range(1, self.epochs + 1):
            self._model.train()
            total_loss = 0.0
            for users, items, ratings in train_loader:
                users = users.to(self.device)
                items = items.to(self.device)
                ratings = ratings.to(self.device)

                optimizer.zero_grad()
                preds = self._model(users, items)
                loss = criterion(preds, ratings)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item() * len(ratings)

            avg_loss = total_loss / len(train_dataset)
            self.train_losses.append(avg_loss)

            if val_loader is not None:
                val_rmse = self._evaluate_rmse(val_loader, criterion, torch)
                self.val_rmses.append(val_rmse)
                scheduler.step(val_rmse)

                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}

                if epoch % 5 == 0 or epoch == 1:
                    logger.info(f"   Epoch {epoch:3d}/{self.epochs} — "
                                f"train_loss={avg_loss:.4f}, val_rmse={val_rmse:.4f}")
            else:
                if epoch % 5 == 0 or epoch == 1:
                    logger.info(f"   Epoch {epoch:3d}/{self.epochs} — train_loss={avg_loss:.4f}")

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)
            logger.info(f"NCF training complete. Best val RMSE: {best_val_rmse:.4f}")
        else:
            logger.info("NCF training complete.")

        self.is_fitted = True
        return self

    def _evaluate_rmse(self, val_loader, criterion, torch) -> float:
        self._model.eval()
        total_sq_err = 0.0
        total_n = 0
        with torch.no_grad():
            for users, items, ratings in val_loader:
                users = users.to(self.device)
                items = items.to(self.device)
                ratings = ratings.to(self.device)
                preds = self._model(users, items)
                total_sq_err += ((preds - ratings) ** 2).sum().item()
                total_n += len(ratings)
        return float(np.sqrt(total_sq_err / total_n))

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")

        if self._encoder is None:
            raise RuntimeError("Encoder not attached to NCF model.")

        user_idx = np.array([self._encoder.user2idx.get(u, -1) for u in user_ids])
        item_idx = np.array([self._encoder.movie2idx.get(m, -1) for m in movie_ids])
        
        unknown_mask = (user_idx == -1) | (item_idx == -1)
        user_idx = np.where(user_idx == -1, 0, user_idx)
        item_idx = np.where(item_idx == -1, 0, item_idx)

        self._model.eval()
        with torch.no_grad():
            u_t = torch.tensor(user_idx, dtype=torch.long).to(self.device)
            i_t = torch.tensor(item_idx, dtype=torch.long).to(self.device)
            preds = self._model(u_t, i_t).cpu().numpy().flatten()
        
        preds[unknown_mask] = 3.0
        return np.clip(preds, 1.0, 5.0).astype(np.float32)

    def recommend(self, user_id: int, n: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        if self._encoder is None:
            return []

        if user_id not in self._encoder.user2idx:
            return []

        u_idx = self._encoder.user2idx[user_id]
        seen = self.get_seen_movies(user_id) if exclude_seen else set()

        all_movie_raw = list(self._encoder.movie2idx.keys())
        candidate_raw = [m for m in all_movie_raw if m not in seen]
        candidate_idx = np.array([self._encoder.movie2idx[m] for m in candidate_raw])

        user_idx_arr = np.full(len(candidate_idx), u_idx, dtype=np.int64)

        self._model.eval()
        with torch.no_grad():
            u_t = torch.tensor(user_idx_arr, dtype=torch.long).to(self.device)
            i_t = torch.tensor(candidate_idx, dtype=torch.long).to(self.device)

            batch = 4096
            all_scores = []
            for start in range(0, len(u_t), batch):
                scores = self._model(u_t[start:start+batch], i_t[start:start+batch])
                all_scores.append(scores.cpu().numpy())

        scores = np.concatenate(all_scores)
        scores = np.clip(scores, 1.0, 5.0)

        top_idx = np.argsort(scores)[::-1][:n]
        return [(int(candidate_raw[i]), float(scores[i])) for i in top_idx]

    def save_weights(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._model.state_dict(), path)
        logger.info(f"NCF weights saved to {path}")

    def load_weights(self, path: str):
        if self._model is None:
            self._model = NeuMF(
                self.n_users, self.n_items, self.embedding_dim, self.layers, self.dropout
            ).to(self.device)
        self._model.load_state_dict(torch.load(path, map_location=self.device))
        self.is_fitted = True
        logger.info(f"NCF weights loaded from {path}")