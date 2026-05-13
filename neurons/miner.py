"""Poker44 miner — heuristic baseline + optional sigmoid calibration.

Single-file deterministic scorer based on per-chunk behavioral signals
(action-type ratios, street depth, showdown flag, player count).

Optionally applies a sigmoid calibration on top of the raw score:
final = 1 / (1 + exp(-(raw - offset) * scale))
The calibration parameters are loaded from a JSON file pointed to by
POKER44_CALIBRATION_PATH; the file is hot-reloaded on each forward via
mtime check. If the file is missing or unreadable, the raw baseline
score is emitted unchanged.
"""

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

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
CALIBRATION_PATH = os.getenv("POKER44_CALIBRATION_PATH", "")
CALIBRATION_UIDS = {
    s.strip()
    for s in os.getenv("POKER44_CALIBRATION_UIDS", "").split(",")
    if s.strip()
}

STRATEGY_ASSIGNMENT_PATH = os.getenv(
    "POKER44_STRATEGY_ASSIGNMENT_PATH", ""
)
ML_MODEL_PATH = os.getenv("POKER44_ML_MODEL_PATH", "")
SUPPORTED_STRATEGIES = {
    "baseline", "all_zero", "all_half", "max", "ml", "ml13tens",
}


_ML13TENS_SCORER = None


def _ml13tens_predict(chunks: list) -> list:
    """Score each chunk via tomkaba's TorchScript runtime model.

    Lazy-imports so the torch.jit.load only fires when this strategy is
    actually requested. Returns 0.0-filled list on any load/predict failure
    so callers can fall back gracefully.
    """
    global _ML13TENS_SCORER
    if _ML13TENS_SCORER is None:
        try:
            from poker44.miner_heuristics import score_chunk as _sc
            _ML13TENS_SCORER = _sc
            bt.logging.info("[ml13tens] loaded runtime_model.ts")
        except Exception as exc:
            bt.logging.warning(f"[ml13tens] load failed: {exc}")
            return [0.0] * len(chunks)
    return [_ML13TENS_SCORER(c) for c in chunks]


_ML_FEATURE_NAMES = [
    "n_hands", "stack_mean", "stack_min", "stack_max", "stack_unique",
    "stack_count", "action_total", "actions_per_hand",
    "ratio_other", "ratio_fold", "ratio_call", "ratio_check",
    "ratio_raise", "ratio_bet", "ratio_all_in",
    "streets_mean", "streets_std", "showdown_rate",
    "n_players_mean", "pot_pre_mean", "pot_pre_max", "pot_post_mean",
    "pot_post_max", "pot_growth_mean",
    "end_preflop", "end_flop", "end_turn", "end_river",
    "amount_mean", "amount_max", "amount_std", "normalized_amount_mean",
    "action_seq_unique", "fold_to_raise_ratio",
    "check_to_bet_ratio",
    "preflop_action_share", "flop_action_share", "turn_action_share",
    "river_action_share",
    "min_to_max_stack_ratio",
]


class _MLHolder:
    """Hot-reloads pickled ensemble artifact from ML_MODEL_PATH."""

    def __init__(self):
        self._mtime = 0.0
        self._artifact = None

    def get(self):
        if not ML_MODEL_PATH:
            return None
        try:
            mt = os.path.getmtime(ML_MODEL_PATH)
        except OSError:
            return self._artifact
        if mt <= self._mtime and self._artifact is not None:
            return self._artifact
        try:
            import pickle
            with open(ML_MODEL_PATH, "rb") as fh:
                self._artifact = pickle.load(fh)
            self._mtime = mt
            bt.logging.info(
                f"[ml] reloaded artifact from {ML_MODEL_PATH} "
                f"(val_ap={self._artifact.get('val_ap')})"
            )
        except Exception as exc:
            bt.logging.warning(f"[ml] reload failed: {exc}")
        return self._artifact


_ML = _MLHolder()


def _ml_chunk_features(hands: list) -> list:
    stacks = []
    actions_by_street = Counter()
    action_types = Counter()
    streets = []
    pots_pre, pots_post = [], []
    amounts, norm_amounts = [], []
    n_players = []
    end_streets = Counter()
    action_seq = []
    showdowns = 0
    for hand in hands:
        meta = hand.get("metadata") or {}
        outcome = hand.get("outcome") or {}
        players = [p for p in (hand.get("players") or [])
                   if p.get("starting_stack", 0) > 0]
        actions = hand.get("actions") or []
        n_players.append(len(players))
        end_streets[meta.get("hand_ended_on_street", "")] += 1
        if outcome.get("showdown"):
            showdowns += 1
        for p in players:
            stacks.append(round(float(p.get("starting_stack", 0) or 0), 4))
        for a in actions:
            atype = a.get("action_type", "")
            action_types[atype] += 1
            actions_by_street[a.get("street", "")] += 1
            try:
                pots_pre.append(float(a.get("pot_before", 0) or 0))
                pots_post.append(float(a.get("pot_after", 0) or 0))
            except Exception:
                pass
            try:
                amounts.append(float(a.get("amount", 0) or 0))
            except Exception:
                pass
            try:
                norm_amounts.append(float(a.get("normalized_amount_bb", 0) or 0))
            except Exception:
                pass
            action_seq.append(atype)
        streets.append(len(hand.get("streets") or []))

    n_hands = max(1, len(hands))
    total_actions = max(1, sum(action_types.values()))
    pot_growth = [(po - pp) for pp, po in zip(pots_pre, pots_post)]

    def m(xs): return float(sum(xs) / len(xs)) if xs else 0.0
    def mx(xs): return float(max(xs)) if xs else 0.0
    def mn(xs): return float(min(xs)) if xs else 0.0
    def sd(xs):
        if not xs: return 0.0
        a = sum(xs) / len(xs)
        return float((sum((x - a) ** 2 for x in xs) / len(xs)) ** 0.5)

    return [
        n_hands, m(stacks), mn(stacks), mx(stacks),
        len(set(stacks)), len(stacks),
        total_actions, total_actions / n_hands,
        action_types.get("other", 0) / total_actions,
        action_types.get("fold", 0) / total_actions,
        action_types.get("call", 0) / total_actions,
        action_types.get("check", 0) / total_actions,
        action_types.get("raise", 0) / total_actions,
        action_types.get("bet", 0) / total_actions,
        action_types.get("all_in", 0) / total_actions,
        m(streets), sd(streets),
        showdowns / n_hands, m(n_players),
        m(pots_pre), mx(pots_pre),
        m(pots_post), mx(pots_post),
        m(pot_growth),
        end_streets.get("preflop", 0) / n_hands,
        end_streets.get("flop", 0) / n_hands,
        end_streets.get("turn", 0) / n_hands,
        end_streets.get("river", 0) / n_hands,
        m(amounts), mx(amounts), sd(amounts), m(norm_amounts),
        len(set(action_seq)),
        action_types.get("fold", 0) / max(1, action_types.get("raise", 0) + action_types.get("bet", 0)),
        action_types.get("check", 0) / max(1, action_types.get("bet", 0) + action_types.get("raise", 0)),
        actions_by_street.get("preflop", 0) / total_actions,
        actions_by_street.get("flop", 0) / total_actions,
        actions_by_street.get("turn", 0) / total_actions,
        actions_by_street.get("river", 0) / total_actions,
        (mn(stacks) / max(1e-6, mx(stacks))) if stacks else 0.0,
    ]


def _ml_predict(chunks: list) -> list:
    art = _ML.get()
    if art is None or not chunks:
        return [0.0] * len(chunks)
    import numpy as np
    X_full = np.array(
        [_ml_chunk_features(c) for c in chunks], dtype=np.float32
    )
    feat_idx = art.get("feature_indices")
    X = X_full[:, feat_idx] if feat_idx is not None else X_full
    try:
        p_lgb = art["lightgbm"].predict_proba(X)[:, 1]
        p_rf = art["rf"].predict_proba(X)[:, 1]
        Xs = art["scaler"].transform(X)
        p_lr = art["logreg"].predict_proba(Xs)[:, 1]
        w = art.get(
            "ens_weights", {"lightgbm": 0.5, "rf": 0.3, "logreg": 0.2}
        )
        ens = (w["lightgbm"] * p_lgb + w["rf"] * p_rf + w["logreg"] * p_lr)
        bias = float(art.get("bias_shift") or 0.0)
        if bias:
            ens = np.clip(ens + bias, 0.0, 1.0)
        return [float(round(x, 6)) for x in ens]
    except Exception as exc:
        bt.logging.warning(f"[ml] predict failed: {exc}; falling back to zeros")
        return [0.0] * len(chunks)


class _StrategyCache:
    """Hot-reloads {uid: strategy} mapping from a JSON file."""

    def __init__(self):
        self._mtime = 0.0
        self._mapping: dict[str, str] = {}

    def get(self, uid: int) -> str:
        if not STRATEGY_ASSIGNMENT_PATH:
            return "baseline"
        try:
            mt = os.path.getmtime(STRATEGY_ASSIGNMENT_PATH)
        except OSError:
            return self._mapping.get(str(uid), "baseline")
        if mt > self._mtime:
            try:
                with open(STRATEGY_ASSIGNMENT_PATH, "r") as fh:
                    self._mapping = {
                        str(k): str(v) for k, v in json.load(fh).items()
                    }
                self._mtime = mt
                bt.logging.info(
                    f"[strategy] reloaded mapping ({len(self._mapping)} "
                    f"UIDs)"
                )
            except Exception as exc:
                bt.logging.warning(f"[strategy] reload failed: {exc}")
        s = self._mapping.get(str(uid), "baseline")
        if s not in SUPPORTED_STRATEGIES:
            return "baseline"
        return s


_STRATEGY = _StrategyCache()


class _CalibrationCache:
    """Hot-reloads sigmoid params from a JSON file via mtime."""

    def __init__(self):
        self._mtime = 0.0
        self._params: Optional[dict] = None

    def get(self) -> Optional[dict]:
        if not CALIBRATION_PATH:
            return None
        try:
            mt = os.path.getmtime(CALIBRATION_PATH)
        except OSError:
            return self._params
        if mt <= self._mtime and self._params is not None:
            return self._params
        try:
            with open(CALIBRATION_PATH, "r") as fh:
                obj = json.load(fh)
            scale = float(obj.get("scale"))
            offset = float(obj.get("offset"))
            self._params = {"scale": scale, "offset": offset}
            self._mtime = mt
            bt.logging.info(
                f"[calib] reloaded scale={scale} offset={offset}"
            )
        except Exception as exc:
            bt.logging.warning(f"[calib] reload failed: {exc}")
        return self._params


_CALIB = _CalibrationCache()


def _apply_calibration(raw: float) -> float:
    p = _CALIB.get()
    if not p:
        return raw
    try:
        z = (raw - p["offset"]) * p["scale"]
        z = max(-30.0, min(30.0, z))
        return 1.0 / (1.0 + math.exp(-z))
    except Exception:
        return raw


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
                "model_version": "1.0.7",
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
        raw_scores = [self._score_chunk(chunk) for chunk in chunks]
        strategy = _STRATEGY.get(self.uid)
        scores = self._apply_strategy(strategy, chunks, raw_scores)

        calib_eligible = (
            not CALIBRATION_UIDS or str(self.uid) in CALIBRATION_UIDS
        )
        if calib_eligible and _CALIB.get() is not None:
            scores = [_apply_calibration(s) for s in scores]
            calib_state = "on"
        else:
            calib_state = "off"

        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        n_true = sum(1 for s in scores if s >= 0.5)
        bt.logging.info(
            f"Scored {len(chunks)} chunks | "
            f"True={n_true} False={len(scores) - n_true} "
            f"strat={strategy} calib={calib_state}"
        )
        self._maybe_dump(chunks, raw_scores, synapse)
        return synapse

    @staticmethod
    def _apply_strategy(
        strategy: str, chunks: list, raw_scores: list[float]
    ) -> list[float]:
        if strategy == "all_zero":
            return [0.0] * len(raw_scores)
        if strategy == "all_half":
            return [0.5] * len(raw_scores)
        if strategy == "max":
            out: list[float] = []
            for chunk, _ in zip(chunks, raw_scores):
                if not chunk:
                    out.append(0.0)
                    continue
                hs = [Miner._score_hand(h) for h in chunk]
                out.append(round(max(hs), 6) if hs else 0.0)
            return out
        if strategy == "ml":
            preds = _ml_predict(chunks)
            if preds and any(p > 0 for p in preds):
                return preds
            return list(raw_scores)
        if strategy == "ml13tens":
            preds = _ml13tens_predict(chunks)
            if preds and any(p > 0 for p in preds):
                return preds
            return list(raw_scores)
        return list(raw_scores)

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
