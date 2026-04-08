from pathlib import Path


# Paths
EXPERIMENT = "default" 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / EXPERIMENT
RAW_DIR = RESULTS_DIR / "raw"
PROCESSED_DIR = RESULTS_DIR / "processed"
FIG_DIR = RESULTS_DIR / "figures"
LOG_DIR = PROJECT_ROOT / "logs"

for _p in [RESULTS_DIR, RAW_DIR, PROCESSED_DIR, FIG_DIR, LOG_DIR]:
    _p.mkdir(parents=True, exist_ok=True)


# Model & calculator

MODEL_NAME = "uma-s-1p1"
TASK_NAME = "omat"
DEVICE = "cpu"   


# Alloy systems

BASE_ELEMENT = "W"
SOLUTES = ["Ta", "Nb", "Mo"]

# 0.00, 0.05, ..., 1.00
COMPOSITIONS = [i / 100 for i in range(0, 101, 5)]


# Structure settings

A0_DICT = {
    "W": 3.17,
    "Ta": 3.30,
    "Nb": 3.30,
    "Mo": 3.15,
}

SUPERCELL = (4, 4, 4)
RANDOM_SEED = 42


# Relax settings

RELAX_FMAX = 0.01
RELAX_STEPS = 800


# Deformation settings

HYDRO_MAX_STRAIN = 0.01
HYDRO_N_DEFORM = 9

NORMAL_MAX_STRAIN = 0.01
NORMAL_N_DEFORM = 9

SHEAR_MAX_STRAIN = 0.04
SHEAR_N_DEFORM = 9


# Polynomial fitting

POLY_ORDER = 2


# Plot / repeated random seeds

USE_MULTIPLE_SEEDS = True
SEED_LIST = [42, 52, 62]