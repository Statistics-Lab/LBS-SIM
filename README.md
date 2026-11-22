# Estimation and Testing of Heterogeneous Log-Birnbaum–Saunders Partially Linear Single-Index Models

Reproducible code for the manuscript submitted to *Statistics in Medicine*.

The repository implements the two partially linear single-index models based on the
log-Birnbaum–Saunders (LBS) distribution under right censoring described in the paper:

- **Model 1** — a mean-based partially linear single-index model with a constant shape parameter.
- **Model 2** — a joint partially linear single-index model in which both the mean and the
  shape parameter depend on covariates through single-index functions.

Both models approximate the unknown single-index functions with B-splines and estimate the
parameters by a two-stage Newton–Raphson algorithm. Parameter confidence intervals and the
likelihood-ratio test for linearization are obtained by parametric bootstrap.

## Repository structure

```
.
├── data/
│   ├── GBSG2_data.csv        # preprocessed GBSG2 data used by the real-data analyses
│   └── GBSG2_raw.xlsx        # raw GBSG2 data (same source as above, before dummy coding)
├── application/
│   ├── application_model1_homoscedastic.py  # real-data analysis, Model 1
│   └── application_model2_heteroscedastic.py  # real-data analysis, Model 2
├── simulation_study/
│   ├── sim_model1_n{n}_cens{c}.py     # simulation, Model 1 (n = 200/400/600, c = 10/30/50)
│   ├── sim_model2_n{n}_cens{c}.py     # simulation, Model 2 (same grid)
│   ├── plot_sim_model1.py                   # summary figures for Model 1 (Figure 1)
│   └── plot_sim_model2.py                   # summary figures for Model 2 (Figure 2)
└── README.md
```

## Data

The German Breast Cancer Study Group 2 (GBSG2) data set records 686 node-positive breast
cancer patients with complete prognostic information, of whom 387 are censored (an overall
censoring rate of 56.41%). The response is the logarithm of the recurrence-free survival time
(in days); the censoring indicator is 1 for recurrence and 0 for censoring.

The data are publicly available through the Python package
[`scikit-survival`](https://scikit-survival.readthedocs.io/) (`load_gbsg2`). The file
`data/GBSG2_data.csv` is the version used by the real-data scripts: `age` and `tsize` are
normalized and enter the single-index function, while the remaining categorical predictors
are dummy-coded and enter the linear part. The raw file `data/GBSG2_raw.xlsx` is provided for
reference. Both files are included in the repository for convenience and are not intended to
be submitted separately as journal data files.

## Dependencies

The code was written for Python 3 and requires the following packages:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `scienceplots` (optional; used only in a commented-out plotting style)
- `xlsxwriter` (for writing the result workbooks)

## How to run

All scripts locate their inputs and outputs relative to their own location, so the
repository can be cloned and run from any directory.

### 1. Real-data analysis

```bash
python application/application_model1_homoscedastic.py
python application/application_model2_heteroscedastic.py
```

Each script reads `data/GBSG2_data.csv`, fits the corresponding model, and writes the
estimated single-index curves, bootstrap confidence bands, and parameter estimates to a
`results/` folder created next to the script.

### 2. Simulation study

Run the estimation scripts first. For each model, the scripts cover the sample sizes
`n = 200, 400, 600` and target censoring rates `10%, 30%, 50%`, with `m = 1000` replications
per combination:

```bash
python simulation_study/sim_model1_n200_cens10.py
# ... (one script per n/censoring-rate combination; see the file list)
```

Each script writes its outputs (single-index curve estimates, parameter estimates, and
evaluation metrics) to a `results/` folder created next to the script.

### 3. Summary figures

After the simulation scripts for a given model have finished, run the corresponding plotting
script to produce the summary figures (Figure 1 for Model 1, Figure 2 for Model 2):

```bash
python simulation_study/plot_sim_model1.py
python simulation_study/plot_sim_model2.py
```

The plotting scripts locate the result folders produced by the simulation scripts
automatically and save the figures to `simulation_study/figures/`.
