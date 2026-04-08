# w-alloy-elastic-properties
Master ML project
This project investigates the elastic properties of W-based alloys 
(W-Ta, W-Nb, W-Mo) using atomistic simulations with UMA potentials(task name = "OMAT", predictor = "uma-s-1p1").

## Objectives
- Compute elastic constants (C11, C12, C44)
- Derive bulk modulus (B) and shear modulus (G)
- Analyze B/G ratio trends with alloy composition
- Compare against linear melting temperature (Vegard's law)

## Methods
- Structure generation using ASE
- Relaxation with FIRE optimizer
- Elastic tensor fitting (strain-energy method)
- Post-processing and visualization

## Results
- B/G vs composition trends
- B/G vs melting temperature correlation

## Project Structure
### Core Modules (src/)
- `alloy_builder.py`  
  Generate alloy structures (random substitution, supercells)
- `elastic.py`  
  Main workflow for elastic tensor calculation
- `relax_utils.py`  
  Structure relaxation (FIRE optimizer)
- `deformation_utils.py`  
  Apply strain tensors
- `energy_utils.py`  
  Energy evaluation using UMA model
- `fit_utils.py`  
  Polynomial fitting for energy-strain relations
- `property_utils.py`  
  Compute B, G, and B/G ratio
- `io_utils.py`  
  Data saving/loading
- `tm_utils.py`  
  Linear melting temperature estimation (Vegard's law)
### Running Experiments (src/)
- `python src/run_single_case.py`
  Single Case
- `python src/run_all_alloys.py`
  All alloys
### Experiment Design
Strain Range Tests
- `python src/config.py` - largestrain_test1-
- `python src/config_small_strain_test2.py`- smallstrain_test2
Strain Methods Test
- `python src/config_delete_size_test3.py` - delete_size_test3 - `python src/deformation_utils_3.py`
Fitting Methods Test
- `python src/config_fit_test4.py` - fit_test4 - `python src/fit_utils_4.py`

Now the output figures use `config_5.py`, `deformtion_utils_3.py`, `fit_utils_4.py`
