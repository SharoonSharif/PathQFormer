"""
Download and prepare TCGA data for PathQ-Former.

Downloads:
  1. Clinical + survival data from UCSC Xena
  2. Gene expression (FPKM) from UCSC Xena
  3. Pathway definitions from Reactome + MSigDB

Pre-extracted WSI embeddings (UNI2/CONCH) must be obtained separately
from HuggingFace (gated access): https://huggingface.co/MahmoodLab

Usage:
    python scripts/download_data.py --cancer_type BLCA --output_dir data/
"""

import argparse
import os
import json
from pathlib import Path

import pandas as pd
import numpy as np

XENA_BASE = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download"

CANCER_TYPES = {
    "BLCA": "TCGA-BLCA",
    "BRCA": "TCGA-BRCA",
    "STAD": "TCGA-STAD",
    "COADREAD": ["TCGA-COAD", "TCGA-READ"],
    "HNSC": "TCGA-HNSC",
    "GBMLGG": ["TCGA-GBM", "TCGA-LGG"],
}

SURVIVAL_ENDPOINT = "DSS"  # disease-specific survival


def download_file(url: str, output_path: Path) -> None:
    """Download a file if it doesn't already exist."""
    if output_path.exists():
        print(f"  Already exists: {output_path}")
        return

    import urllib.request
    print(f"  Downloading: {url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    print(f"  Saved to: {output_path}")


def download_clinical_data(output_dir: Path) -> Path:
    """Download pan-cancer clinical data from Xena."""
    url = f"{XENA_BASE}/Survival_SupplementalTable_S1_20171025_xena_sp"
    output_path = output_dir / "tcga_clinical" / "survival_data.tsv"
    download_file(url, output_path)
    return output_path


def download_gene_expression(output_dir: Path) -> Path:
    """Download pan-cancer gene expression (RSEM expected counts, log2(x+1)) from Xena."""
    url = f"{XENA_BASE}/EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
    output_path = output_dir / "tcga_transcriptomics" / "pancan_gene_expression.tsv.gz"
    download_file(url, output_path)
    return output_path


def prepare_clinical_csv(
    survival_path: Path,
    cancer_type: str,
    output_dir: Path,
    num_bins: int = 4,
) -> Path:
    """
    Filter clinical data for a cancer type and compute discrete survival time bins.
    """
    print(f"\nPreparing clinical data for {cancer_type}...")
    df = pd.read_csv(survival_path, sep="\t")

    project_codes = CANCER_TYPES[cancer_type]
    if isinstance(project_codes, str):
        project_codes = [project_codes]

    # Filter to cancer type and DSS endpoint
    mask = df["_PATIENT"].str[:12].apply(
        lambda x: any(x.startswith(code.replace("TCGA-", "TCGA-")[:8]) for code in project_codes)
    )
    df_cancer = df[mask].copy()

    dss_col = f"{SURVIVAL_ENDPOINT}"
    dss_time_col = f"{SURVIVAL_ENDPOINT}.time"

    if dss_col not in df_cancer.columns:
        print(f"  Warning: {dss_col} not found. Available: {df_cancer.columns.tolist()}")
        return None

    df_cancer = df_cancer.dropna(subset=[dss_col, dss_time_col])
    df_cancer = df_cancer[df_cancer[dss_time_col] > 0]

    df_cancer["case_id"] = df_cancer["_PATIENT"]
    df_cancer["censorship"] = 1 - df_cancer[dss_col].astype(int)
    df_cancer["survival_time"] = df_cancer[dss_time_col]

    _, bin_edges = pd.qcut(
        df_cancer["survival_time"], q=num_bins, retbins=True, duplicates="drop",
    )
    df_cancer["survival_time_bin"] = pd.cut(
        df_cancer["survival_time"], bins=bin_edges, labels=False, include_lowest=True,
    )

    output_path = output_dir / "tcga_clinical" / f"{cancer_type}_clinical.csv"
    df_cancer.to_csv(output_path, index=False)
    print(f"  Saved {len(df_cancer)} patients to {output_path}")
    return output_path


def create_pathway_json(output_dir: Path) -> None:
    """
    Create placeholder pathway JSON files.
    In practice, download GMT files from:
      - Reactome: https://reactome.org/download-data (ReactomePathways.gmt)
      - MSigDB: https://www.gsea-msigdb.org/gsea/msigdb (h.all.vX.X.symbols.gmt)
    Then run: build_reactome_pathways() / build_msigdb_hallmarks() from pathway_tokenizer.py
    """
    pathway_dir = output_dir / "pathways"
    pathway_dir.mkdir(parents=True, exist_ok=True)

    readme_path = pathway_dir / "README.txt"
    if not readme_path.exists():
        readme_path.write_text(
            "Download pathway GMT files:\n"
            "1. Reactome: https://reactome.org/download-data -> ReactomePathways.gmt\n"
            "2. MSigDB Hallmarks: https://www.gsea-msigdb.org/gsea/msigdb -> h.all.v2024.1.Hs.symbols.gmt\n\n"
            "Then convert to JSON using:\n"
            "  from src.data.pathway_tokenizer import build_reactome_pathways\n"
            "  build_reactome_pathways('ReactomePathways.gmt', 'reactome.json')\n"
        )
    print(f"  Pathway README created at {readme_path}")


def main():
    parser = argparse.ArgumentParser(description="Download TCGA data for PathQ-Former")
    parser.add_argument("--cancer_type", type=str, default="BLCA", choices=list(CANCER_TYPES.keys()))
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--num_bins", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("Step 1: Downloading clinical / survival data...")
    survival_path = download_clinical_data(output_dir)

    print("Step 2: Downloading gene expression data...")
    download_gene_expression(output_dir)

    print("Step 3: Preparing clinical CSV...")
    prepare_clinical_csv(survival_path, args.cancer_type, output_dir, args.num_bins)

    print("Step 4: Creating pathway file placeholders...")
    create_pathway_json(output_dir)

    print("\nDone! Next steps:")
    print("  1. Download WSI embeddings from HuggingFace (UNI2 or CONCH)")
    print("  2. Download Reactome/MSigDB GMT files and convert to JSON")
    print("  3. Run training: python -m src.training.train --config configs/default.yaml")


if __name__ == "__main__":
    main()
