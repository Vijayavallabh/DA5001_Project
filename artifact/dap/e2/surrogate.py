import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import E2Config

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import train_test_split
except Exception:
    TfidfVectorizer = None
    LogisticRegression = None
    train_test_split = None


class SemanticMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TokenCNN(nn.Module):
    def __init__(self, vocab_size: int, pad_idx: int, emb_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.conv1 = nn.Conv1d(emb_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.drop = nn.Dropout(0.15)
        self.fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 1))

    def forward(self, x):
        h = self.emb(x).transpose(1, 2)
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.drop(h)
        h = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        return self.fc(h).squeeze(-1)


class FusionMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ConstantBinaryModel:
    def __init__(self, p: float):
        self.p = float(np.clip(p, 1e-6, 1.0 - 1e-6))

    def predict_proba(self, X):
        n = len(X)
        out = np.zeros((n, 2), dtype=np.float32)
        out[:, 1] = self.p
        out[:, 0] = 1.0 - self.p
        return out


class ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, X):
        return np.full((len(X),), self.value, dtype=np.float32)


class Standardizer:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        self.ready = False

    def fit(self, y: np.ndarray):
        y = np.asarray(y, dtype=np.float32)
        self.mean = float(np.mean(y))
        self.std = float(np.std(y))
        if self.std < 1e-6:
            self.std = 1.0
        self.ready = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        if not self.ready:
            return y
        return (y - self.mean) / self.std

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        if not self.ready:
            return y
        return y * self.std + self.mean


class SurrogateEnsemble:
    def __init__(self, cfg: E2Config, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.val_frac = 0.15
        self.seed = 0
        use_cuda = torch.cuda.is_available() and str(cfg.surrogate_device).startswith("cuda")
        self.device = torch.device(cfg.surrogate_device if use_cuda else "cpu")

        self.sent_model = SentenceTransformer(cfg.sentence_model_name) if SentenceTransformer is not None else None
        self.tfidf = None

        self.semantic = None
        self.token = None
        self.keyword_safe = None
        self.keyword_rho = None
        self.fusion_rho = None
        self.safe_models = []

        self.rho_scaler = Standardizer()
        self.margin_scaler = Standardizer()

        self.ready = False
        self.feature_dim = None
        self.last_fit_info = {}

    def _seed_everything(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _sigmoid_np(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        x = np.clip(x, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _standardize_feature(self, arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        mu = arr.mean() if arr.size else 0.0
        sd = arr.std() if arr.size else 1.0
        if sd < 1e-6:
            sd = 1.0
        return ((arr - mu) / sd).astype(np.float32)

    def sentence_embed(self, texts: List[str]) -> np.ndarray:
        if self.sent_model is not None:
            arr = self.sent_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return arr.astype(np.float32)
        out = np.zeros((len(texts), 768), dtype=np.float32)
        for i, txt in enumerate(texts):
            toks = txt.lower().split()[:768]
            vals = np.asarray([((hash(t) % 1000) / 1000.0) for t in toks], dtype=np.float32)
            out[i, :len(vals)] = vals
        return out

    def token_prefix_ids(self, texts: List[str], max_len: int = 64) -> np.ndarray:
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
        rows = []
        for txt in texts:
            ids = self.tokenizer(txt, return_tensors="pt").input_ids[0].tolist()[:max_len]
            ids += [pad_id] * (max_len - len(ids))
            rows.append(ids)
        return np.asarray(rows, dtype=np.int64)

    def fit_tfidf(self, texts: List[str]):
        if TfidfVectorizer is None:
            self.tfidf = None
            return
        self.tfidf = TfidfVectorizer(
            max_features=self.cfg.tfidf_features, lowercase=True, binary=False,
            ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        )
        self.tfidf.fit(texts)

    def tfidf_features(self, texts: List[str]) -> np.ndarray:
        if self.tfidf is None:
            return np.zeros((len(texts), self.cfg.tfidf_features), dtype=np.float32)
        arr = self.tfidf.transform(texts).toarray().astype(np.float32)
        if arr.shape[1] < self.cfg.tfidf_features:
            pad = np.zeros((arr.shape[0], self.cfg.tfidf_features - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        return arr

    def _weighted_bce_with_logits(self, logits, y, sample_weight=None, pos_weight=None):
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()

    def _weighted_mse(self, pred, y, sample_weight=None):
        loss = (pred - y) ** 2
        if sample_weight is not None:
            loss = loss * sample_weight
        return loss.mean()

    def _train_torch(self, model, x_train, y_train, x_val, y_val, sample_weight_train=None, sample_weight_val=None, token_mode=False, task="regression", pos_weight=None, seed=0):
        self._seed_everything(int(seed))
        model = model.to(self.device)

        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.surrogate_lr)
        best_state = None
        best_val = float("inf")
        bad = 0
        bs = self.cfg.surrogate_batch_size

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train, dtype=np.float32)
        x_val = np.asarray(x_val)
        y_val = np.asarray(y_val, dtype=np.float32)

        if sample_weight_train is None:
            sample_weight_train = np.ones(len(y_train), dtype=np.float32)
        if sample_weight_val is None:
            sample_weight_val = np.ones(len(y_val), dtype=np.float32)
        sample_weight_train = np.asarray(sample_weight_train, dtype=np.float32)
        sample_weight_val = np.asarray(sample_weight_val, dtype=np.float32)

        pos_weight_t = None
        if pos_weight is not None:
            pos_weight_t = torch.tensor([float(pos_weight)], device=self.device, dtype=torch.float32)

        for _ in range(self.cfg.surrogate_epochs):
            order = np.random.permutation(len(y_train))
            model.train()
            for start in range(0, len(order), bs):
                idx = order[start:start + bs]
                xb = torch.tensor(x_train[idx], device=self.device)
                xb = xb.long() if token_mode else xb.float()
                yb = torch.tensor(y_train[idx], device=self.device).float()
                wb = torch.tensor(sample_weight_train[idx], device=self.device).float()

                pred = model(xb)
                if task == "classification":
                    loss = self._weighted_bce_with_logits(pred, yb, sample_weight=wb, pos_weight=pos_weight_t)
                else:
                    loss = self._weighted_mse(pred, yb, sample_weight=wb)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                xv = torch.tensor(x_val, device=self.device)
                xv = xv.long() if token_mode else xv.float()
                yv = torch.tensor(y_val, device=self.device).float()
                wv = torch.tensor(sample_weight_val, device=self.device).float()
                pv = model(xv)
                if task == "classification":
                    val_loss = self._weighted_bce_with_logits(pv, yv, sample_weight=wv, pos_weight=pos_weight_t).item()
                else:
                    val_loss = self._weighted_mse(pv, yv, sample_weight=wv).item()

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                bad = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.cfg.surrogate_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    def _build_targets(self, archive_rows, K: float):
        rho = np.asarray([float(r.rho) for r in archive_rows], dtype=np.float32)
        U = np.asarray([float(r.U_EBB) for r in archive_rows], dtype=np.float32)
        B = np.asarray([float(r.effective_budget_min) for r in archive_rows], dtype=np.float32)
        delta_init = np.asarray([float(r.delta_init_mean) for r in archive_rows], dtype=np.float32)
        final_budget_mean = np.asarray([float(r.final_budget_mean) for r in archive_rows], dtype=np.float32)

        rho = np.nan_to_num(rho, nan=10.0, posinf=10.0, neginf=0.0)
        U = np.nan_to_num(U, nan=K * 10.0, posinf=K * 10.0, neginf=0.0)
        B = np.nan_to_num(B, nan=K, posinf=K, neginf=0.0)

        y_safe = np.asarray([1.0 if r.U_EBB <= K else 0.0 for r in archive_rows], dtype=np.float32)
        viol = 1.0 - y_safe
        margin = B - U
        margin_norm = margin / max(1e-6, float(K))

        return {"rho": rho, "U": U, "B": B, "delta_init": delta_init, "final_budget_mean": final_budget_mean, "y_safe": y_safe, "viol": viol, "margin_norm": margin_norm.astype(np.float32)}

    def _build_base_features(self, texts, delta_init_guess):
        sem = self.sentence_embed(texts)
        tok = self.token_prefix_ids(texts, max_len=64)
        kw = self.tfidf_features(texts)
        prompt_lens = np.asarray([len(self.tokenizer(t).input_ids) for t in texts], dtype=np.float32)
        length_norm = np.clip(prompt_lens / max(1, self.cfg.max_prompt_tokens), 0.0, 1.0)
        meta = np.stack([length_norm.astype(np.float32), np.asarray(delta_init_guess, dtype=np.float32)], axis=1).astype(np.float32)
        return sem, tok, kw, meta

    def _sample_replay_indices(self, y_safe, rho, replay_n):
        n = len(y_safe)
        if replay_n <= 0 or n == 0:
            return np.asarray([], dtype=np.int64)
        unsafe_idx = np.where(y_safe < 0.5)[0]
        safe_idx = np.where(y_safe >= 0.5)[0]
        boundary_idx = np.argsort(np.abs(rho - 1.0))[: max(1, replay_n // 3)]
        recent_idx = np.arange(max(0, n - replay_n), n)

        chosen = []
        if len(unsafe_idx) > 0:
            k = min(len(unsafe_idx), max(1, replay_n // 3))
            chosen.extend(np.random.choice(unsafe_idx, size=k, replace=False).tolist())
        if len(safe_idx) > 0:
            k = min(len(safe_idx), max(1, replay_n // 3))
            chosen.extend(np.random.choice(safe_idx, size=k, replace=False).tolist())
        chosen.extend(boundary_idx.tolist())
        chosen.extend(recent_idx.tolist())

        chosen = np.unique(np.asarray(chosen, dtype=np.int64))
        if len(chosen) > replay_n:
            chosen = np.random.choice(chosen, size=replay_n, replace=False)
        return np.asarray(chosen, dtype=np.int64)

    def _make_split(self, y, n):
        idx = np.arange(n)
        y = np.asarray(y)
        if n < 2:
            return idx, np.array([], dtype=int)
        val_size = max(1, int(round(self.val_frac * n)))
        if val_size >= n:
            val_size = n - 1
        counts = Counter(y.tolist())
        min_count = min(counts.values()) if counts else 0
        n_classes = len(counts)
        can_stratify = n_classes >= 2 and min_count >= 2 and val_size >= n_classes and (n - val_size) >= n_classes
        if can_stratify:
            tr_idx, va_idx = train_test_split(idx, test_size=val_size, random_state=self.seed, stratify=y)
        else:
            tr_idx, va_idx = train_test_split(idx, test_size=val_size, random_state=self.seed, stratify=None)
        return np.asarray(tr_idx), np.asarray(va_idx)

    def _class_pos_weight(self, y_bin: np.ndarray) -> float:
        pos = float((y_bin > 0.5).sum())
        neg = float((y_bin <= 0.5).sum())
        if pos < 1.0:
            return 1.0
        return max(1.0, neg / pos)

    def _safe_sample_weights(self, y_safe, viol):
        pos = max(1.0, float((y_safe > 0.5).sum()))
        neg = max(1.0, float((y_safe <= 0.5).sum()))
        w_pos = len(y_safe) / (2.0 * pos)
        w_neg = len(y_safe) / (2.0 * neg)
        base = np.where(y_safe > 0.5, w_pos, w_neg).astype(np.float32)
        viol_boost = np.where(viol > 0.5, self.cfg.violator_weight, 1.0).astype(np.float32)
        return base * viol_boost

    def _rho_sample_weights(self, rho, viol):
        boundary = 1.0 / (0.25 + np.abs(rho - 1.0))
        boundary = boundary / max(1e-6, float(boundary.mean()))
        viol_boost = np.where(viol > 0.5, self.cfg.violator_weight, 1.0)
        return (boundary * viol_boost).astype(np.float32)

    def fit(self, archive_rows, K: float):
        if len(archive_rows) < 24:
            self.ready = False
            return {"ready": False, "reason": "too_few_rows"}

        texts = [r.prompt_text for r in archive_rows]
        t = self._build_targets(archive_rows, K)

        self.fit_tfidf(texts)
        sem, tok, kw, meta = self._build_base_features(texts, t["delta_init"])

        tr_idx, va_idx = self._make_split(t["y_safe"], len(texts))

        replay_n = max(1, int(len(texts) * self.cfg.replay_fraction))
        replay_idx = self._sample_replay_indices(t["y_safe"], t["rho"], replay_n)
        tr_idx = np.unique(np.concatenate([tr_idx, replay_idx]))

        rho_train = np.log1p(np.clip(t["rho"], 0.0, None)).astype(np.float32)
        self.rho_scaler.fit(rho_train)
        y_rho_std = self.rho_scaler.transform(rho_train)

        self.semantic = self._train_torch(
            SemanticMLP(sem.shape[1]),
            sem[tr_idx], y_rho_std[tr_idx], sem[va_idx], y_rho_std[va_idx],
            sample_weight_train=self._rho_sample_weights(t["rho"], t["viol"])[tr_idx],
            sample_weight_val=self._rho_sample_weights(t["rho"], t["viol"])[va_idx],
            token_mode=False, task="regression", seed=11,
        )

        self.token = self._train_torch(
            TokenCNN(vocab_size=len(self.tokenizer), pad_idx=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0), emb_dim=128),
            tok[tr_idx], y_rho_std[tr_idx], tok[va_idx], y_rho_std[va_idx],
            sample_weight_train=self._rho_sample_weights(t["rho"], t["viol"])[tr_idx],
            sample_weight_val=self._rho_sample_weights(t["rho"], t["viol"])[va_idx],
            token_mode=True, task="regression", seed=17,
        )

        with torch.no_grad():
            sem_pred_std = self.semantic(torch.tensor(sem, device=self.device).float()).cpu().numpy().astype(np.float32)
            tok_pred_std = self.token(torch.tensor(tok, device=self.device).long()).cpu().numpy().astype(np.float32)

        sem_pred_rho = np.expm1(self.rho_scaler.inverse_transform(sem_pred_std)).astype(np.float32)
        tok_pred_rho = np.expm1(self.rho_scaler.inverse_transform(tok_pred_std)).astype(np.float32)

        if LogisticRegression is None or len(np.unique(t["y_safe"].astype(int))) < 2:
            self.keyword_safe = ConstantBinaryModel(float(t["y_safe"].mean()))
        else:
            self.keyword_safe = LogisticRegression(max_iter=2000)
            self.keyword_safe.fit(kw[tr_idx], t["y_safe"][tr_idx].astype(int), sample_weight=self._safe_sample_weights(t["y_safe"], t["viol"])[tr_idx])

        if Ridge is None:
            self.keyword_rho = ConstantRegressor(float(rho_train[tr_idx].mean()))
        else:
            self.keyword_rho = Ridge(alpha=1.0, random_state=13)
            self.keyword_rho.fit(kw[tr_idx], y_rho_std[tr_idx], sample_weight=self._rho_sample_weights(t["rho"], t["viol"])[tr_idx])

        kw_safe_pred = self.keyword_safe.predict_proba(kw)[:, 1].astype(np.float32)
        kw_rho_pred_std = self.keyword_rho.predict(kw).astype(np.float32)
        kw_rho_pred = np.expm1(self.rho_scaler.inverse_transform(kw_rho_pred_std)).astype(np.float32)

        fuse_in = np.concatenate([
            sem_pred_rho[:, None], tok_pred_rho[:, None], kw_rho_pred[:, None], kw_safe_pred[:, None],
            meta, sem[:, :256], kw[:, : min(256, kw.shape[1])],
        ], axis=1).astype(np.float32)

        self.feature_dim = int(fuse_in.shape[1])
        margin_target = t["margin_norm"].astype(np.float32)
        self.margin_scaler.fit(margin_target)
        y_margin_std = self.margin_scaler.transform(margin_target)

        self.fusion_rho = self._train_torch(
            FusionMLP(self.feature_dim),
            fuse_in[tr_idx], y_margin_std[tr_idx], fuse_in[va_idx], y_margin_std[va_idx],
            sample_weight_train=self._rho_sample_weights(t["rho"], t["viol"])[tr_idx],
            sample_weight_val=self._rho_sample_weights(t["rho"], t["viol"])[va_idx],
            token_mode=False, task="regression", seed=23,
        )

        safe_weights = self._safe_sample_weights(t["y_safe"], t["viol"])
        pos_weight = self._class_pos_weight(t["y_safe"])

        self.safe_models = []
        ensemble_size = 5
        for seed in [101, 103, 107, 109, 113][:ensemble_size]:
            boot = np.random.RandomState(seed).choice(tr_idx, size=len(tr_idx), replace=True)
            model = self._train_torch(
                FusionMLP(self.feature_dim),
                fuse_in[boot], t["y_safe"][boot], fuse_in[va_idx], t["y_safe"][va_idx],
                sample_weight_train=safe_weights[boot], sample_weight_val=safe_weights[va_idx],
                token_mode=False, task="classification", pos_weight=pos_weight, seed=seed,
            )
            self.safe_models.append(model)

        self.ready = True
        self.last_fit_info = {
            "rows": len(archive_rows), "train_rows": len(tr_idx), "val_rows": len(va_idx),
            "replay_rows": len(replay_idx), "safe_rate": float(t["y_safe"].mean()),
            "unsafe_rate": float(1.0 - t["y_safe"].mean()), "feature_dim": self.feature_dim,
            "ensemble_size": len(self.safe_models),
        }
        return {"ready": True, **self.last_fit_info}

    def _predict_base(self, texts, delta_init_guess):
        sem, tok, kw, meta = self._build_base_features(texts, np.asarray(delta_init_guess, dtype=np.float32))
        with torch.no_grad():
            sem_pred_std = self.semantic(torch.tensor(sem, device=self.device).float()).cpu().numpy().astype(np.float32)
            tok_pred_std = self.token(torch.tensor(tok, device=self.device).long()).cpu().numpy().astype(np.float32)
        sem_pred_rho = np.expm1(self.rho_scaler.inverse_transform(sem_pred_std)).astype(np.float32)
        tok_pred_rho = np.expm1(self.rho_scaler.inverse_transform(tok_pred_std)).astype(np.float32)
        kw_safe_pred = self.keyword_safe.predict_proba(kw)[:, 1].astype(np.float32) if self.keyword_safe is not None else np.zeros(len(texts), dtype=np.float32)
        kw_rho_pred_std = self.keyword_rho.predict(kw).astype(np.float32) if self.keyword_rho is not None else np.zeros(len(texts), dtype=np.float32)
        kw_rho_pred = np.expm1(self.rho_scaler.inverse_transform(kw_rho_pred_std)).astype(np.float32)
        fuse_in = np.concatenate([
            sem_pred_rho[:, None], tok_pred_rho[:, None], kw_rho_pred[:, None], kw_safe_pred[:, None],
            meta, sem[:, :256], kw[:, : min(256, kw.shape[1])],
        ], axis=1).astype(np.float32)
        return sem_pred_rho, tok_pred_rho, kw_rho_pred, kw_safe_pred, fuse_in

    def predict(self, texts: List[str], delta_init_guess: List[float]) -> Dict[str, np.ndarray]:
        if not self.ready:
            z = np.zeros(len(texts), dtype=np.float32)
            p = np.full(len(texts), 0.5, dtype=np.float32)
            return {"sem": z, "tok": z, "kw": z, "fuse": z, "safe": p, "safe_mean": p, "safe_sigma": z, "sigma": z, "margin": z}

        sem_pred_rho, tok_pred_rho, kw_rho_pred, kw_safe_pred, fuse_in = self._predict_base(texts, delta_init_guess)

        with torch.no_grad():
            margin_std = self.fusion_rho(torch.tensor(fuse_in, device=self.device).float()).cpu().numpy().astype(np.float32)
        margin_pred = self.margin_scaler.inverse_transform(margin_std).astype(np.float32)

        safe_member_probs = []
        with torch.no_grad():
            x = torch.tensor(fuse_in, device=self.device).float()
            for mdl in self.safe_models:
                logits = mdl(x).cpu().numpy().astype(np.float32)
                probs = self._sigmoid_np(logits)
                safe_member_probs.append(probs)

        safe_member_probs = np.stack(safe_member_probs, axis=1) if safe_member_probs else np.full((len(texts), 1), 0.5, dtype=np.float32)
        safe_mean = safe_member_probs.mean(axis=1).astype(np.float32)
        safe_sigma = safe_member_probs.std(axis=1).astype(np.float32)

        rho_fuse = np.clip(1.0 - margin_pred, 0.0, None).astype(np.float32)

        return {
            "sem": sem_pred_rho, "tok": tok_pred_rho, "kw": kw_rho_pred,
            "fuse": rho_fuse, "safe": safe_mean, "safe_mean": safe_mean,
            "safe_sigma": safe_sigma, "sigma": safe_sigma, "margin": margin_pred.astype(np.float32),
            "kw_safe": kw_safe_pred.astype(np.float32),
            "safe_members": safe_member_probs.astype(np.float32),
        }
