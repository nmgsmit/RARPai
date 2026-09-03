# RARPai

Segmentation + metric depth for RARP video. Edited locally, trained on Snellius —
see [CLAUDE.md](CLAUDE.md) for the cluster workflow and [COMMANDS.md](COMMANDS.md)
for the copy-paste job commands.

## Apps you can run (interactive, local — no cluster needed)

| App | Command | What it does |
|-----|---------|--------------|
| Depth measuring GUI | `python scripts/gui_depth_measure.py` | Pick a frame, click two points, get the distance in mm from a trained EndoDAC depth model. Checkpoint dropdown + on-disk depth cache. Flags: `--ckpt`, `--data-dir`, `--no-model` (UI only), `--self-test`. |
| ATLAS-Interactive GUI | `python gui.py` in `../transfer_atlas_mod` | The upstream annotation/inference GUI (separate repo). |

Batch scripts (no GUI) live in `scripts/`; the SUL pipeline is
`scripts/sul_measure.py` (2D) and `scripts/sul_measure_depth.py` (2D vs 3D).
