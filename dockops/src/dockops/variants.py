"""Zero-shot variant-effect scoring with ESM-2 (masked-marginal log ratio).

For each mutation, the WT residue at that position is masked and the model
predicts a distribution over amino acids; the score is
log p(mutant) - log p(wildtype). Negative = disfavored vs WT. This is the
standard zero-shot variant-effect technique (ESM-1v lineage), variant
prioritization, not de-novo design.

Model: facebook/esm2_t6_8M_UR50D (8M params), CPU-feasible, downloaded
once by huggingface on first use.
"""

from __future__ import annotations

import json
import sys

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
AA = "ACDEFGHIKLMNPQRSTVWY"


def mutation_score(logprobs, wt_id: int, mut_id: int) -> float:
    """log p(mut) - log p(wt) at a position. Pure function, testable."""
    return float(logprobs[mut_id] - logprobs[wt_id])


class ESM2Scorer:
    def __init__(self, model_name: str = MODEL_NAME):
        import torch  # noqa: F401
        from transformers import AutoTokenizer, EsmForMaskedLM

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmForMaskedLM.from_pretrained(model_name).eval()

    def _logprobs_at(self, seq: str, pos: int):
        """Masked-marginal log-probs over AAs at 0-based residue `pos`."""
        import torch

        masked = seq[:pos] + self.tok.mask_token + seq[pos + 1 :]
        inputs = self.tok(masked, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        mask_idx = (
            inputs["input_ids"][0] == self.tok.mask_token_id
        ).nonzero()[0].item()
        return torch.log_softmax(logits[mask_idx], dim=-1)

    def score(self, seq: str, pos: int, mut: str) -> dict:
        """Score mut at 1-based residue position pos of seq."""
        if not 1 <= pos <= len(seq):
            raise ValueError(f"pos {pos} out of range for len {len(seq)}")
        if mut not in AA:
            raise ValueError(f"not a standard amino acid: {mut!r}")
        i = pos - 1
        wt = seq[i]
        if wt not in AA:
            raise ValueError(f"non-standard residue at pos {pos}: {wt!r}")
        lp = self._logprobs_at(seq, i)
        wt_id = self.tok.convert_tokens_to_ids(wt)
        mut_id = self.tok.convert_tokens_to_ids(mut)
        return {
            "mutation": f"{wt}{pos}{mut}",
            "score": round(mutation_score(lp, wt_id, mut_id), 4),
        }


def main() -> None:
    """Demo: score a few mutations on a small protein (GB1, 56 aa)."""
    seq = (
        "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    )
    mutations = [(1, "A"), (10, "G"), (35, "W"), (53, "P")]
    scorer = ESM2Scorer()
    out = [scorer.score(seq, pos, mut) for pos, mut in mutations]
    print(json.dumps({"model": MODEL_NAME, "results": out}, indent=2))


if __name__ == "__main__":
    main()
