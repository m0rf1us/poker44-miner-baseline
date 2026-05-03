"""Poker44 miner — heuristic baseline.

Single-file deterministic scorer based on per-chunk behavioral signals
(action-type ratios, street depth, showdown flag, player count). No training
step. Pinned commit so the running scoring logic is publicly verifiable.
"""

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


PINNED_REPO_URL = "https://github.com/m0rf1us/poker44-miner-baseline"
PINNED_REPO_COMMIT = "{{REPO_COMMIT}}"

DUMP_PATH = os.getenv("POKER44_CHUNKS_DUMP_PATH", "")
DUMP_UIDS = {
    s.strip()
    for s in os.getenv("POKER44_DUMP_UIDS", "").split(",")
    if s.strip()
}


class Miner(BaseMinerNeuron):
    """Heuristic Poker44 miner with publicly verifiable scoring."""

    def __init__(self, config=None):
        super().__init__(config=config)
        repo_root = Path(__file__).resolve().parents[1]
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[Path(__file__).resolve()],
            defaults={
                "model_name": "poker44-baseline-v1",
                "model_version": "1.0.2",
                "framework": "python-heuristic",
                "license": "MIT",
                "repo_url": PINNED_REPO_URL,
                "repo_commit": PINNED_REPO_COMMIT,
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Heuristic miner. No training step. "
                    "Uses only runtime chunk features."
                ),
                "training_data_sources": ["none"],
                "private_data_attestation": (
                    "This miner does not train on "
                    "validator-private human data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(
            self.model_manifest
        )
        self.manifest_digest = manifest_digest(self.model_manifest)
        bt.logging.info(
            f"Manifest status: {self.manifest_compliance['status']} | "
            f"missing={self.manifest_compliance['missing_fields']} | "
            f"digest={self.manifest_digest[:12]}"
        )
        bt.logging.info("Poker44 baseline miner started")

    async def forward(
        self, synapse: DetectionSynapse
    ) -> DetectionSynapse:
        chunks = synapse.chunks or []
        scores = [self._score_chunk(chunk) for chunk in chunks]
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        n_true = sum(1 for s in scores if s >= 0.5)
        bt.logging.info(
            f"Scored {len(chunks)} chunks | "
            f"True={n_true} False={len(scores) - n_true}"
        )
        self._maybe_dump(chunks, scores, synapse)
        return synapse

    def _maybe_dump(self, chunks, scores, synapse) -> None:
        if not DUMP_PATH:
            return
        if DUMP_UIDS and str(self.uid) not in DUMP_UIDS:
            return
        try:
            try:
                vhk = synapse.dendrite.hotkey if synapse.dendrite else "?"
            except Exception:
                vhk = "?"
            record = {
                "ts": int(time.time()),
                "uid": int(self.uid),
                "validator_hotkey": vhk,
                "chunks": [
                    {"score": float(scores[i]), "hands": chunks[i]}
                    for i in range(len(chunks))
                ],
            }
            with open(DUMP_PATH, "a") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        except Exception as exc:
            bt.logging.warning(f"chunks dump failed: {exc}")

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(
            action.get("action_type") for action in actions
        )
        meaningful = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )
        call_ratio = action_counts.get("call", 0) / meaningful
        check_ratio = action_counts.get("check", 0) / meaningful
        fold_ratio = action_counts.get("fold", 0) / meaningful
        raise_ratio = action_counts.get("raise", 0) / meaningful
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0
        else:
            player_count_signal = 0.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * cls._clamp01(call_ratio / 0.35)
        score += 0.12 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.10 * cls._clamp01(raise_ratio / 0.20)
        return cls._clamp01(score)

    @classmethod
    def _score_chunk(cls, chunk: list) -> float:
        if not chunk:
            return 0.5
        hand_scores = [cls._score_hand(hand) for hand in chunk]
        return round(
            cls._clamp01(sum(hand_scores) / len(hand_scores)), 6
        )

    async def blacklist(
        self, synapse: DetectionSynapse
    ) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Poker44 baseline miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | "
                f"Incentive: {miner.metagraph.I[miner.uid]}"
            )
            time.sleep(5 * 60)
