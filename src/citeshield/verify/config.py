"""Verifier configuration. Thresholds are calibrated in eval/results/verifier_calibration.md."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VerifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    nli_model: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    nli_revision: str = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
    # CPU by default: measured 336 ms/pair on CPU vs 18 ms on GPU, but the GPU (8 GB) is
    # already shared by the Ollama model and the retrieval models during local runs.
    device: str = "cpu"
    batch_size: int = Field(8, ge=1)

    # Claim is supported when P(entailment) >= entail_threshold for some usable citation (or
    # all usable citations combined). A cited source contradicts it when P(contradiction) >=
    # contradiction_threshold. Summary and caveat sentences use sentence_threshold.
    # Values: midpoints of the separating margins on the calibration half of
    # eval/verification/claims.jsonl (citeshield-calibrate-verifier). Feasible intervals were
    # wide, (0.038, 0.991) and (0.166, 0.959): the set shows the classifier is reliable on
    # clear-cut claims but does not pin the thresholds down. See verifier_calibration.md.
    entail_threshold: float = Field(0.51, ge=0, le=1)
    contradiction_threshold: float = Field(0.56, ge=0, le=1)
    sentence_threshold: float = Field(0.51, ge=0, le=1)

    # Refuse when the best binding-source rerank score in the evidence is below this.
    # Milestone 3 data (golden set, 53 answerable / 7 unanswerable): answerable top scores
    # >= 0.668, unanswerable <= 0.151. 0.30 sits in that gap, nearer the unanswerable side
    # so answerable questions are not lost. Provisional: small sample, and the agent's own
    # queries differ from the golden questions.
    min_retrieval_score: float = Field(0.30, ge=0, le=1)

    min_quote_segment_chars: int = Field(12, ge=1)
    max_revisions: int = Field(1, ge=0, le=1, description="At most one controlled revision.")
