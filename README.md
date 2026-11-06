# PAST-Mixer

Official source release for **PAST-Mixer: Physics-Anchored Residual Learning for Long-Horizon Ballistic Trajectory Prediction From Short Radar Tracks**.

PAST-Mixer fits a generic drag model to a short radar track and learns the anchor's residual error. Finite-difference kinematic descriptors and anchor context (fit residual, future anchor shape, and fitted drag coefficient) condition a pre-normalized temporal-channel mixer with direct horizon projection. A masked physical-unit objective supervises predictions through impact.

## Release scope

This repository contains the six backbones, training and evaluation pipeline, Drag-LS anchor fit and model-based baselines, experiment matrices, tests, and a non-operational synthetic smoke-test generator. The parameterized benchmark generator, upstream radar tracker, generated benchmark data, checkpoints, and result arrays are not included in this source-only release.

The manuscript reports two benchmarks with different truth aerodynamics: a high-fidelity main benchmark (International Standard Atmosphere and a transonic drag rise) and a nominal-physics second benchmark (exponential atmosphere). Both use the same split layout below and the same tools. The manuscript benchmark uses controlled representative projectile-class simulation and is not field telemetry, an exact firing-table reconstruction, or an operational deployment study. Demo data verify the software interface only and must not be used to reproduce or reinterpret the manuscript results. The first-submission code remains at tag `submission-release`.

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
python tools/evaluate_protocol_v2_physics_baselines.py --data-dir data/protocol_hf --out outputs/protocol_hf/physics_baselines
python tools/evaluate_protocol_v2_physics_baselines.py --data-dir data/protocol_v2 --out outputs/protocol_v2/physics_baselines
```

This evaluates CV, CA, CV+gravity, Drag-LS, Drag-UKF, and Drag-LS-Std and writes Drag-LS anchors for all splits. Drag-LS-Std is the Drag-LS fit with a standard drag function of Mach number (G1 or G7) and an ISA atmosphere; the table and prior strength are selected on validation, and its prior acts on the window-end drag level. `--fast` skips Drag-UKF and `--no-anchors` writes baseline predictions only. For additional compatible splits, `--anchors-only --report outputs/protocol_hf/physics_baselines/report.json` reuses the stored training prior and validation-selected fit strength; adding `--anchor-model b6` writes anchors from the selected Drag-LS-Std fit instead. The demo does not supply polar measurements and does not exercise this script.

## Paper protocol and experiments

The resubmission uses L=100 at 0.01 s, H=2000 masked after impact, and remaining time uniformly sampled over 5-20 s. All main-table networks use kinematics, physics anchors, fit context, and RMSE + 0.5 ADE with FDE weight 0. PAST-Mixer uses width 256, feed-forward width 512, 3 blocks, learning rate 1e-3, batch size 32, at most 80 epochs, patience 15, and seeds 2021-2025. Minimum validation RMSE selects checkpoints.

See [`configs/paper_protocol.json`](configs/paper_protocol.json) for settings and sensor/tracker specifications, and [`configs/experiments/main_v3_20261102.py`](configs/experiments/main_v3_20261102.py) for the final job matrices and tuned baseline sizes. V3 imports v2 and sets the final objective. CLI defaults retain the source's legacy settings; use the v3 matrix for paper runs.

Print a matrix with:

```bash
python configs/experiments/main_v3_20261102.py main
```

The matrices use `data/protocol_v2` (nominal benchmark) by default. Set the environment variable `BTP_DATA` to another protocol root, e.g. `data/protocol_hf`, to run every stage on that benchmark; derived datasets use its suffixes (`protocol_hf_hold`, `protocol_hf_holdaero`), and a `protocol_hf` root selects the baseline sizes tuned on the high-fidelity benchmark.

Stages are `main` (fixed-size recipe study), `abl`, `tune`, `best` (tuned baselines), `rand`, `win`, and `stress <condition> [fair|rand]`. Stress and window stages expect separately prepared compatible datasets; their generator is not released.

Build the 30-job main-table comparison (five PAST-Mixer seeds and five seeds for each tuned baseline), run it, and summarize saved physical-coordinate predictions:

```bash
python -c "import json,sys; from pathlib import Path; sys.path.insert(0,'configs/experiments'); from main_v3_20261102 import jobs; p=Path('outputs/paper/config/jobs.json'); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps([j for j in jobs('main') if j['name'].startswith('main_R5c_PASTMixer_')]+jobs('best')),encoding='utf-8')"
python tools/run_queue.py --jobs outputs/paper/config/jobs.json --out outputs/paper --gpus 0 --per_gpu 1
python tools/summarize_protocol_v2.py --out outputs/paper --data-dir data/protocol_v2 --physics outputs/protocol_v2/physics_baselines --prefix main_R5c_PASTMixer,best_ --bootstrap 2000 --pairs "R5c:PASTMixer,B4_DragLS;R5c:PASTMixer,R5c:TSMixer"
```

The queue writes logs, checkpoints, results, and status under its output directory and skips jobs with existing metrics. For CPU runs, append `--use_gpu 0` to each job's argument list before queuing. Summaries report seed mean and sample standard deviation, with paired trajectory/seed bootstrap intervals when requested. Benchmark runs require the compatible data and anchors described above.

Stages `hold` and `holdaero` train the leave-one-class-out models on datasets built by `tools/make_holdout_data.py` (`holdaero`: aerodynamically randomized training and validation splits, with the test split taken from `--test-source`); `tools/summarize_protocol_v2.py --pool-holdout <root> --pool-tag hold|holdaero [--runs <dir>]` pools them with class-stratified bootstrap intervals and includes Drag-LS-Std when present. `tools/conformal_protocol_v2.py [--hold-tag hold|holdaero]` computes split-conformal impact regions from saved R5c PAST-Mixer checkpoints (inference only): disks centered on the seed-ensemble impact and scaled by the seed spread, calibrated on the validation split.

## Manuscript evidence

On the high-fidelity benchmark PAST-Mixer reaches 38.3 m ADE, 33% below the strongest physical baseline, Drag-LS-Std (57.1 m), and improves on it under all nine radar degradations; its impact error is 38.0 m median and 98.6 m at the 90th percentile, with 0.170 s TTI error. Physics anchoring reduces the ADE of each of five backbones by 49%-56%. Among six backbones tuned with equal budgets, PAST-Mixer has the lowest ADE on both benchmarks (38.3 m on each). Under leave-one-class-out training the anchored networks reach about half the physical error on the withheld artillery classes. These manuscript results are not regenerated by the source-only demo.

## Citation

The manuscript is under review. See [`CITATION.cff`](CITATION.cff) for citation metadata; the final bibliographic record will be added after publication.

## License

The source code is released under the [MIT License](LICENSE).
