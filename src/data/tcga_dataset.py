"""TCGA multimodal survival dataset (patient-level).

Protocol, aligned with MCAT / SurvPath so that numbers are comparable to the literature:

* One sample per **patient** (``case_id``). Patches from all of a patient's slides are
  concatenated into a single bag; the query bottleneck handles the variable bag size.
* Discrete survival bins are quantiles of the **uncensored training** patients, with the
  outer edges extended to -inf / +inf and left-closed intervals ``[a, b)``. The same bins
  are re-used unchanged for the validation split (no label leakage from validation).
* Gene expression is scaled with statistics fit on the training split only
  (SurvPath uses MinMax to [-1, 1]; ``standard`` and ``none`` are also available).
* The continuous survival time is carried through so the concordance index is computed on
  it rather than on the 4 coarse bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# endpoint name -> (time column, censorship column) in SurvPath metadata CSVs
ENDPOINTS = {
    "dss": ("survival_months_dss", "censorship_dss"),
    "os": ("survival_months", "censorship"),
    "pfi": ("survival_months_pfi", "censorship_pfi"),
}

_SLIDE_EXTS = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".h5", ".pt")


def slide_stem(slide_id: str) -> str:
    """Strip a known slide / embedding extension: 'TCGA-...-DX1.<uuid>.svs' -> 'TCGA-...-DX1.<uuid>'."""
    name = Path(str(slide_id)).name
    for ext in _SLIDE_EXTS:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


@dataclass
class SurvivalBins:
    """Discrete time bins. ``edges`` has ``num_bins + 1`` entries; outer edges are -inf/+inf."""

    edges: np.ndarray

    @classmethod
    def from_training(cls, times, censorship, num_bins: int) -> "SurvivalBins":
        times = np.asarray(times, dtype=float)
        censorship = np.asarray(censorship, dtype=float)
        uncensored = times[censorship < 1]
        if len(np.unique(uncensored)) < num_bins:  # degenerate cohort: fall back to all patients
            uncensored = times
        _, edges = pd.qcut(uncensored, q=num_bins, retbins=True, labels=False, duplicates="drop")
        edges = np.asarray(edges, dtype=float)
        if len(edges) != num_bins + 1:
            raise ValueError(
                f"Could not form {num_bins} distinct quantile bins from {len(uncensored)} "
                f"uncensored training times (got {len(edges) - 1})."
            )
        edges[0], edges[-1] = -np.inf, np.inf
        return cls(edges=edges)

    @property
    def num_bins(self) -> int:
        return len(self.edges) - 1

    def assign(self, times) -> np.ndarray:
        """Left-closed assignment: bin ``i`` covers ``[edges[i], edges[i+1])``."""
        idx = np.searchsorted(self.edges, np.asarray(times, dtype=float), side="right") - 1
        return np.clip(idx, 0, self.num_bins - 1).astype(np.int64)

    def interior_edges(self) -> np.ndarray:
        return self.edges[1:-1]

    def to_dict(self) -> dict:
        return {"edges": [None if not np.isfinite(e) else float(e) for e in self.edges]}

    @classmethod
    def from_dict(cls, d: dict) -> "SurvivalBins":
        edges = np.asarray([np.nan if e is None else e for e in d["edges"]], dtype=float)
        edges[0], edges[-1] = -np.inf, np.inf
        return cls(edges=edges)


@dataclass
class OmicsScaler:
    """Per-gene scaler fit on the training split. ``minmax`` maps the train range to [-1, 1] (SurvPath)."""

    kind: str = "minmax"
    center: np.ndarray | None = None
    scale: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "OmicsScaler":
        x = np.asarray(x, dtype=np.float32)
        if self.kind == "minmax":
            lo, hi = x.min(0), x.max(0)
            self.center = lo
            self.scale = np.where(hi - lo > 0, hi - lo, 1.0).astype(np.float32)
        elif self.kind == "standard":
            mu, sd = x.mean(0), x.std(0)
            self.center = mu
            self.scale = np.where(sd > 0, sd, 1.0).astype(np.float32)
        elif self.kind == "none":
            self.center = np.zeros(x.shape[1], dtype=np.float32)
            self.scale = np.ones(x.shape[1], dtype=np.float32)
        else:
            raise ValueError(f"Unknown scaler kind: {self.kind}")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center is None:
            raise RuntimeError("OmicsScaler.transform called before fit")
        x = np.asarray(x, dtype=np.float32)
        z = (x - self.center) / self.scale
        if self.kind == "minmax":
            z = 2.0 * z - 1.0
        return z.astype(np.float32)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "center": self.center.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "OmicsScaler":
        return cls(
            kind=d["kind"],
            center=np.asarray(d["center"], np.float32),
            scale=np.asarray(d["scale"], np.float32),
        )


class TCGAMultimodalDataset(Dataset):
    """Patient-level (WSI bag, gene vector, survival label) samples in SurvPath's data layout."""

    def __init__(
        self,
        metadata_csv: str | Path,
        rna_csv: str | Path,
        embeddings_dir: str | Path,
        split_csv: str | Path,
        split: str = "train",
        num_bins: int = 4,
        endpoint: str = "dss",
        label_col: str | None = None,
        censor_col: str | None = None,
        max_patches: int | None = None,
        bins: SurvivalBins | None = None,
        scaler: OmicsScaler | None = None,
        scaler_kind: str = "minmax",
        cache_in_ram: bool = False,
        cache_dtype: torch.dtype = torch.float16,
    ):
        if label_col is None or censor_col is None:
            if endpoint not in ENDPOINTS:
                raise ValueError(f"endpoint must be one of {list(ENDPOINTS)}, got {endpoint!r}")
            label_col, censor_col = ENDPOINTS[endpoint]
        self.split = split
        self.label_col, self.censor_col = label_col, censor_col
        self.embeddings_dir = Path(embeddings_dir)
        self.max_patches = max_patches
        self._cache: dict[str, torch.Tensor] | None = {} if cache_in_ram else None
        self._cache_dtype = cache_dtype

        # ---- slides in this split with usable labels -------------------------------------
        meta = pd.read_csv(metadata_csv)
        meta["case_id"] = meta["case_id"].astype(str)
        split_ids = pd.read_csv(split_csv)[split].dropna().astype(str).tolist()
        meta = meta[meta["case_id"].isin(split_ids)].copy()
        meta = meta.dropna(subset=[label_col, censor_col])
        meta = meta[meta[label_col] > 0]
        meta["stem"] = meta["slide_id"].map(slide_stem)

        available = self._index_embeddings(self.embeddings_dir)
        self.missing_slides = sorted(set(meta["stem"]) - set(available))
        meta = meta[meta["stem"].isin(available)]
        if meta.empty:
            raise RuntimeError(
                f"No slides of split '{split}' have embeddings under {self.embeddings_dir} "
                f"({len(self.missing_slides)} slides listed in metadata are missing)."
            )

        # ---- collapse to patients ---------------------------------------------------------
        groups = meta.groupby("case_id", sort=True)
        if (groups[label_col].nunique() > 1).any() or (groups[censor_col].nunique() > 1).any():
            raise ValueError("Inconsistent survival labels across slides of the same patient")
        patients = groups.agg(
            survival_time=(label_col, "first"),
            censorship=(censor_col, "first"),
            n_slides=("stem", "size"),
        ).reset_index()
        stems_by_case = groups["stem"].apply(list)
        patients["slide_stems"] = patients["case_id"].map(stems_by_case)

        # ---- gene expression --------------------------------------------------------------
        rna = pd.read_csv(rna_csv)
        rna = rna.rename(columns={rna.columns[0]: "case_id"})
        rna["case_id"] = rna["case_id"].astype(str)
        rna = rna.drop_duplicates("case_id").set_index("case_id")
        self.gene_columns: list[str] = [str(c) for c in rna.columns]
        patients = patients[patients["case_id"].isin(rna.index)].reset_index(drop=True)
        gene_raw = rna.loc[patients["case_id"], :].to_numpy(dtype=np.float32)
        gene_raw = np.nan_to_num(gene_raw, nan=0.0)

        # ---- bins and scaler: fit on train, re-use on val ---------------------------------
        if bins is None:
            if split != "train":
                raise ValueError("Pass the training split's `bins` when building a non-training split")
            bins = SurvivalBins.from_training(patients["survival_time"], patients["censorship"], num_bins)
        if scaler is None:
            if split != "train":
                raise ValueError("Pass the training split's `scaler` when building a non-training split")
            scaler = OmicsScaler(kind=scaler_kind).fit(gene_raw)
        self.bins, self.scaler = bins, scaler

        patients["survival_time_bin"] = bins.assign(patients["survival_time"].to_numpy())
        patients["censorship"] = patients["censorship"].astype(np.float32)
        patients["survival_time"] = patients["survival_time"].astype(np.float32)
        self.patients = patients
        self.gene_matrix = scaler.transform(gene_raw)  # (n_patients, n_genes) float32
        self._paths = {s: available[s] for stems in patients["slide_stems"] for s in stems}

    # ------------------------------------------------------------------------------------
    @staticmethod
    def _index_embeddings(root: Path) -> dict[str, Path]:
        if not root.exists():
            return {}
        found: dict[str, Path] = {}
        for p in root.rglob("*"):
            if p.suffix in (".h5", ".pt") and p.is_file():
                found.setdefault(slide_stem(p.name), p)
        return found

    def _load_slide(self, stem: str) -> torch.Tensor:
        if self._cache is not None and stem in self._cache:
            return self._cache[stem].float()
        path = self._paths[stem]
        if path.suffix == ".h5":
            with h5py.File(path, "r") as f:
                feats = torch.from_numpy(np.asarray(f["features"][:], dtype=np.float32))
        else:
            feats = torch.load(path, map_location="cpu", weights_only=True).float()
        feats = feats.reshape(-1, feats.shape[-1])
        if self._cache is not None:
            self._cache[stem] = feats.to(self._cache_dtype)
        return feats

    def load_coords(self, stem: str) -> np.ndarray | None:
        """Patch coordinates for attention heatmaps, if the embedding file stores them."""
        path = self._paths.get(stem)
        if path is None or path.suffix != ".h5":
            return None
        with h5py.File(path, "r") as f:
            for key in ("coords", "coords_patching"):
                if key in f:
                    return np.asarray(f[key][:]).reshape(-1, 2)
        return None

    # ------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        row = self.patients.iloc[idx]
        feats = torch.cat([self._load_slide(s) for s in row["slide_stems"]], dim=0)
        if self.max_patches is not None and feats.shape[0] > self.max_patches:
            keep = torch.randperm(feats.shape[0])[: self.max_patches]
            feats = feats[keep]
        return {
            "case_id": row["case_id"],
            "slide_ids": list(row["slide_stems"]),
            "wsi_features": feats,
            "gene_expression": torch.from_numpy(self.gene_matrix[idx]),
            "survival_time_bin": torch.tensor(int(row["survival_time_bin"]), dtype=torch.long),
            "censorship": torch.tensor(float(row["censorship"]), dtype=torch.float32),
            "survival_time": torch.tensor(float(row["survival_time"]), dtype=torch.float32),
        }

    def surv_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(event_observed bool, time) arrays, e.g. for IPCW estimators."""
        event = self.patients["censorship"].to_numpy() < 1
        time = self.patients["survival_time"].to_numpy(dtype=float)
        return event, time

    def summary(self) -> str:
        ev = int((self.patients["censorship"] < 1).sum())
        multi = int((self.patients["n_slides"] > 1).sum())
        return (
            f"{self.split}: {len(self)} patients ({ev} events, {multi} with >1 slide), "
            f"{len(self.gene_columns)} genes, {len(self.missing_slides)} slides without embeddings"
        )


def collate_multimodal(batch: list[dict]) -> dict:
    """Pad WSI bags to the longest in the batch and return a validity mask."""
    max_patches = max(item["wsi_features"].shape[0] for item in batch)
    wsi_dim = batch[0]["wsi_features"].shape[1]
    B = len(batch)

    wsi_padded = torch.zeros(B, max_patches, wsi_dim)
    wsi_mask = torch.zeros(B, max_patches, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item["wsi_features"].shape[0]
        wsi_padded[i, :n] = item["wsi_features"]
        wsi_mask[i, :n] = True

    return {
        "case_ids": [b["case_id"] for b in batch],
        "slide_ids": [b["slide_ids"] for b in batch],
        "wsi_features": wsi_padded,
        "wsi_mask": wsi_mask,
        "gene_expression": torch.stack([b["gene_expression"] for b in batch]),
        "survival_time_bin": torch.stack([b["survival_time_bin"] for b in batch]),
        "censorship": torch.stack([b["censorship"] for b in batch]),
        "survival_time": torch.stack([b["survival_time"] for b in batch]),
    }
