# PAST-Mixer

Official source release for **PAST-Mixer: Physics-Anchored Residual Learning for Long-Horizon Ballistic Trajectory Prediction From Short Radar Tracks**.

PAST-Mixer fits a generic drag model to a short radar track and learns the anchor's residual error. Finite-difference kinematic descriptors and anchor context (fit residual, future anchor shape, and fitted drag coefficient) condition a pre-normalized temporal-channel mixer with direct horizon projection. A masked physical-unit objective supervises predictions through impact.

## Release scope

This repository contains the six backbones, training and evaluation pipeline, Drag-LS anchor fit and model-based baselines, experiment matrices, tests, and a non-operational synthetic smoke-test generator. The parameterized benchmark generator, upstream radar tracker, generated benchmark data, checkpoints, and result arrays are not included in this source-only release.

The manuscript benchmark uses controlled representative projectile-class simulation and is not field telemetry, an exact firing-table reconstruction, or an operational deployment study. Demo data verify the software interface only and must not be used to reproduce or reinterpret the manuscript results. The first-submission code remains at tag `submission-release`.

## Installation

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
```

Python 3.10 or later is recommended. GPU execution is optional. On Windows, use a short checkout path to avoid path-length limits when writing experiment artifacts.

## Software smoke test

Generate a small generic dataset and run one CPU epoch through the paper's input, anchor-context, and masked-loss path:

```bash
python scripts/generate_demo_data.py --output data/demo --seq-len 16 --pred-len 32
python run.py --dataset_dir data/demo --seq_len 16 --pred_len 32 --dt 0.1 --input_state kinematics --output_mode physics_anchor --anchor_context fit --loss_mode physical_composite --ade_loss_weight 0.5 --fde_loss_weight 0 --select_metric val_rmse_m --d_model 16 --d_ff 32 --e_layers 1 --batch_size 8 --train_epochs 1 --patience 1 --use_gpu 0 --model_id demo --checkpoints outputs/demo/checkpoints --results outputs/demo/results
```

The demo writes variable `valid_len` values to exercise masking. Its anchor files are constant-velocity placeholders from a linear fit to the observed window, with an arbitrary fitted-beta placeholder; they are not Drag-LS fits. Artifacts go under `outputs/demo/`. This is a software check, not a paper benchmark.

Run the CPU unit tests with:

```bash
python -m pytest
```

## Compatible dataset format

Each split is a PyTorch file named `train.pt`, `val.pt`, or `test.pt`. Cartesian coordinates are in meters, ordered `(x, y, height)`:

- `X`: float tensor or NumPy array `(N, L + H, 3)`, clean observed prefix followed by the future target, padded after impact.
- `X_observed`: float tensor or NumPy array `(N, L, 3)`, tracked observed positions.
- `valid_len`: integer tensor or array `(N,)`, number of valid future steps, including impact, between 1 and H. Steps after this count do not enter loss or metrics.
- `protocol` (optional): dictionary, including `default_observation_mode` and sensor overrides when applicable.
- `meta` (optional for training): per-trajectory lists used by evaluation, such as `domain`, `projectile`, `wind`, `t_go_s`, `impact_xyz_m`, and `radar_site_xyz_m`. Metadata filters use `weapon` and `wind` when requested.

Place `protocol_manifest.json` in the dataset directory for run provenance. The loader fits coordinate normalization on the full `train.pt` only. With `physics_anchor` and `anchor_context=fit`, each split also needs float arrays:

- `anchor_drag_<split>.npy`: `(N, H, 3)`, future Drag-LS positions in meters.
- `anchor_fit_window_<split>.npy`: `(N, L, 3)`, fitted positions over the observed window in meters.
- `anchor_beta_<split>.npy`: `(N,)`, fitted drag coefficient. Context standardizes beta using the full training split's fitted coefficients.

The released baseline script builds these arrays from compatible benchmark splits. Its input additionally requires tensor fields `polar` `(N, W + L, 3)` (range in meters, azimuth/elevation in radians) and `detected` `(N, W + L)` (boolean), plus `meta.domain` and `meta.radar_site_xyz_m` `(N, 3)`. The manuscript uses W=300 warm-up samples. This script expects tensor-valued `X`, `X_observed`, and `valid_len`; the training loader also accepts NumPy arrays. It estimates the beta prior from clean training trajectories and selects fit strength on validation; each anchor fit uses only that trajectory's tracked window.

```bash
python tools/evaluate_protocol_v2_physics_baselines.py --data-dir data/protocol_v2 --out outputs/protocol_v2/physics_baselines
```

This evaluates CV, CA, CV+gravity, Drag-LS, and Drag-UKF and writes anchors for all splits. `--fast` skips Drag-UKF. For additional compatible splits, `--anchors-only --report outputs/protocol_v2/physics_baselines/report.json` reuses the stored training prior and validation-selected fit strength. The demo does not supply polar measurements and does not exercise this script.

## Paper protocol and experiments

The resubmission uses L=100 at 0.01 s, H=2000 masked after impact, and remaining time uniformly sampled over 5-20 s. All main-table networks use kinematics, physics anchors, fit context, and RMSE + 0.5 ADE with FDE weight 0. PAST-Mixer uses width 256, feed-forward width 512, 3 blocks, learning rate 1e-3, batch size 32, at most 80 epochs, patience 15, and seeds 2021-2025. Minimum validation RMSE selects checkpoints.

See [`configs/paper_protocol.json`](configs/paper_protocol.json) for settings and sensor/tracker specifications, and [`configs/experiments/main_v3_20261102.py`](configs/experiments/main_v3_20261102.py) for the final job matrices and tuned baseline sizes. V3 imports v2 and sets the final objective. CLI defaults retain the source's legacy settings; use the v3 matrix for paper runs.

Print a matrix with:

```bash
python configs/experiments/main_v3_20261102.py main
```

Stages are `main` (fixed-size recipe study), `abl`, `tune`, `best` (tuned baselines), `rand`, `win`, and `stress <condition> [fair|rand]`. Stress and window stages expect separately prepared compatible datasets; their generator is not released.

Build the 30-job main-table comparison (five PAST-Mixer seeds and five seeds for each tuned baseline), run it, and summarize saved physical-coordinate predictions:

```bash
python -c "import json,sys; from pathlib import Path; sys.path.insert(0,'configs/experiments'); from main_v3_20261102 import jobs; p=Path('outputs/paper/config/jobs.json'); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps([j for j in jobs('main') if j['name'].startswith('main_R5c_PASTMixer_')]+jobs('best')),encoding='utf-8')"
python tools/run_queue.py --jobs outputs/paper/config/jobs.json --out outputs/paper --gpus 0 --per_gpu 1
python tools/summarize_protocol_v2.py --out outputs/paper --data-dir data/protocol_v2 --physics outputs/protocol_v2/physics_baselines --prefix main_R5c_PASTMixer,best_ --bootstrap 2000 --pairs "R5c:PASTMixer,B4_DragLS;R5c:PASTMixer,R5c:TSMixer"
```

The queue writes logs, checkpoints, results, and status under its output directory and skips jobs with existing metrics. For CPU runs, append `--use_gpu 0` to each job's argument list before queuing. Summaries report seed mean and sample standard deviation, with paired trajectory/seed bootstrap intervals when requested. Benchmark runs require the compatible data and anchors described above.

## Manuscript evidence

The abstract reports ADE 38.3 m versus Drag-LS 65.6 m (42% reduction) and the best purely learned predictor's 69.7 m (45% reduction). Physics anchoring reduces every tested backbone's ADE by 44%-53%. PAST-Mixer's 90th-percentile impact error is 94.9 m and TTI error is 0.187 s; all five paired ADE confidence intervals exclude zero. It retains the lowest ADE in every lead-time bin and improves on Drag-LS across all nine radar perturbations. These manuscript results are not regenerated by the source-only demo.

## Citation

The manuscript is under review. See [`CITATION.cff`](CITATION.cff) for citation metadata; the final bibliographic record will be added after publication.

## License

The source code is released under the [MIT License](LICENSE).
