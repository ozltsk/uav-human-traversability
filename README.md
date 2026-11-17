# Micro-UAV monocular mapping for indoor human traversability

Flight software, mapping and passability code, and the evaluation scripts of the article
"From Simulation to Real Flight: Experimental Evaluation of Micro-UAV Monocular Mapping for Indoor Human
Traversability" (submitted to *Drones*, MDPI).

The flight records, the tape-measured ground truth and the photographs of the scenes are a separate dataset on
Zenodo: **DOI to be added**.

## What is here

| Folder | Content |
|---|---|
| `rmtt_field/` | Flight software for a fleet of DJI RoboMaster TT (Tello Talent, SDK 3.0) drones: survey missions, recording of video, telemetry and range data, an emulator with 34 automatic checks, the field protocol (`PROTOCOL.md`, in Ukrainian), export of surveys for mapping |
| `ac1_evacmap/` | Mapping: relative depth (Depth Anything V2 Small) and floor mask (SegFormer-B0, ADE20K), metric scale from the floor, 3D log-odds occupancy map |
| `ac2_evacroute/` | Passability for three postures (rules M3 and M4, TravNet M5 and M5r), A* routing, generator of the synthetic training scenes |
| `field_analysis/` | Evaluation against the measured ground truth: maps, passage states, route safety, controls, baseline, confirmatory series |
| `runs/ac2_models/`, `runs/ac2_models_rmtt/` | Weights of TravNet: M5 (original synthetic model) and M5r (real-camera model) |
| `drones2026/E6_frozen.json` | Processing settings and SHA-256 sums of 16 code and model files, frozen on 10 November 2026 before the confirmatory series |

Folder names are those of the working project. Do not rename them: the frozen file refers to these paths.
Comments and messages in the code are in Ukrainian.

## Installation

Python 3.12, no GPU needed.

```
pip install -r requirements.txt
```

The depth and segmentation models are downloaded from Hugging Face on first use
(`depth-anything/Depth-Anything-V2-Small-hf`, `nvidia/segformer-b0-finetuned-ade-512-512`).

## Reproducing the results

Unpack the Zenodo dataset so that `data/field_20261116/` contains `runs/`, `ground_truth/`, `calib/` and
`journal.csv`. The first-series scripts read the ground truth of scenes S1–S5 from
`data/field_20261104/ground_truth/`; copy the files of S1–S5 there (see the dataset README).

Check that the code is the frozen one:

```
python field_analysis/freeze_config.py --verify
```

Export the usable surveys and run the perception (about 13 s per frame on a laptop CPU):

```
python rmtt_field/export_field.py data/field_20261116 --out runs/real_20261116
python ac1_evacmap/app.py predict --log runs/real_20261116/<survey>
```

`export_field.py` writes both series into `runs/real_20261116`. The first-series scripts expect the 18 surveys of
S1–S5 in `runs/real_20261104`; copy or move those folders there.

Confirmatory series (Section 4.7, Table 11), run from `field_analysis/`:

```
python e6_eval.py
```

`e6_eval.py` evaluates every pairing of the A and B flights of S9 and S10. The article counts units that share no
flight: the k-th usable A flight of a drone with its k-th usable B flight (26 units).

First series (Sections 4.2–4.5), controls and baseline (Section 4.6), run from `field_analysis/`:

```
python field_eval.py --logs ../runs/real_20261104 --gt ../data/field_20261104/ground_truth --out ../runs/real_20261104/eval18_eps012_th08 --theta 0.8 --max-eps 0.012
python route_safety2.py
python baseline_costmap.py
python tape_map.py
```

`field_eval.py` reads the list of surveys from `_selected.txt` in the `--logs` folder (one folder name per line).
The reference configuration of the article is the floor-fit threshold 0.012 and θ = 0.8
(see `drones2026/E6_frozen.json`); `route_safety2.py` reads its output folder `eval18_eps012_th08`.

## Flying

Read `rmtt_field/PROTOCOL.md` and `rmtt_field/README.md` first. `python rmtt_field/test_field.py` runs the
emulator checks without a drone. The software commands real aircraft: keep the propeller guards on and the
emergency keys at hand.

## Licence

Code: MIT. Dataset: CC BY 4.0 (see Zenodo).
