# poker44-miner-baseline

Reference heuristic miner for Poker44 subnet (netuid 126) on Bittensor.

Implements a deterministic, explainable bot-risk score per chunk based on
behavioral signals — action-type ratios, street depth, showdown flags and
player count. The goal is a transparent baseline above the random level.

## Usage

This file is meant to be deployed as `neurons/miner.py` inside a
[Poker44-subnet](https://github.com/Poker44/Poker44-subnet) checkout.
It depends on upstream packages `poker44.base.miner`,
`poker44.utils.model_manifest`, and `poker44.validator.synapse`.

## License

MIT.
