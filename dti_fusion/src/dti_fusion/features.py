"""Modality encoders.

Drug modality: Morgan fingerprint (ECFP-style, radius/bits from config).
Protein modality: mean-pooled ESM-2 last hidden state, a protein language
model embedding, the standard zero-shot protein representation. Sequences
are truncated to the model's context and the truncation is recorded.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from dti_fusion.config import load_config


def morgan_fingerprint(smiles: str, radius: int, nbits: int):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    if not smiles or not isinstance(smiles, str):
        return None  # None/empty input is unparseable, not a C++ TypeError
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    return gen.GetFingerprint(mol)


def drug_fingerprints(smiles_list, spec: str) -> np.ndarray:
    import re

    m = re.fullmatch(r"morgan-r(\d+)-(\d+)", spec.strip())
    if m is None:
        raise ValueError(
            f"unsupported fingerprint spec {spec!r}, expected "
            "'morgan-r<radius>-<bits>'"
        )
    radius, nbits = int(m.group(1)), int(m.group(2))
    fps, bad = [], []
    for s in smiles_list:
        fp = morgan_fingerprint(s, radius, nbits)
        if fp is None:
            bad.append(s)
            fps.append(np.zeros(nbits, dtype=np.float32))
        else:
            arr = np.zeros(nbits, dtype=np.float32)
            from rdkit import DataStructs

            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(arr)
    if bad:
        print(f"warning: {len(bad)} unparseable SMILES -> zero fingerprint")
    return np.stack(fps)


def protein_embeddings(seqs, model_name: str, max_len: int) -> np.ndarray:
    """Mean-pooled ESM-2 last hidden state, (n, d_model) float32.

    Results persist in the shared ESM_CACHE_DIR store (vendored
    esm_cache) keyed on the truncated sequence actually embedded."""
    from dti_fusion.esm_cache import get_many, put

    effective = [s[:max_len] for s in seqs]
    values, misses = get_many("esm2-embed", model_name, effective)
    if misses:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        with torch.no_grad():
            for i in misses:
                enc = tok(
                    effective[i], return_tensors="pt", truncation=True,
                    max_length=max_len,
                )
                out = model(**enc).last_hidden_state[0]
                mask = enc["attention_mask"][0].bool()
                emb = out[mask].mean(0).numpy()
                values[i] = emb
                put("esm2-embed", model_name, effective[i], emb)
    return np.stack(values).astype(np.float32)


def main(in_parquet: str, out_npz: str):
    cfg = load_config()
    df = pd.read_parquet(in_parquet)
    X_drug = drug_fingerprints(
        df["smiles"].tolist(), cfg["model"]["fingerprint"]
    )
    uniq_seqs = df["sequence"].unique()
    emb = protein_embeddings(
        uniq_seqs.tolist(), cfg["model"]["esm_model"],
        cfg["dataset"]["esm_max_len"],
    )
    seq_emb = {s: e for s, e in zip(uniq_seqs, emb)}
    X_prot = np.stack([seq_emb[s] for s in df["sequence"]])
    n_trunc = int((df["sequence"].str.len() > cfg["dataset"]["esm_max_len"]).sum())
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz, X_drug=X_drug, X_prot=X_prot,
        pkd=df["pkd"].to_numpy(), target_id=df["target_id"].to_numpy(),
        frac_at_cap=np.float64(df["at_cap"].mean()),
        frac_truncated=np.float64(n_trunc / len(df)),
    )
    print(
        f"features: drug {X_drug.shape}, protein {X_prot.shape} "
        f"({n_trunc} rows truncated to {cfg['dataset']['esm_max_len']} aa)"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
