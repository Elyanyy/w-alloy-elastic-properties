from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator


def get_uma_calculator(model_name="uma-s-1p1", task_name="omat", device="cpu"):
    """
    Initialize UMA calculator.

    Notes
    -----
    Depending on your local fairchem version, get_predict_unit() may or may not
    accept 'device'. If it errors, remove device=device below.
    """
    try:
        predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
    except TypeError:
        predictor = pretrained_mlip.get_predict_unit(model_name)

    calc = FAIRChemCalculator(predictor, task_name=task_name)
    return calc


def relax_structure(
    atoms,
    calculator,
    fmax=0.01,
    steps=800,
):
    """
    Relax both atomic positions and cell.
    """
    atoms = atoms.copy()
    atoms.calc = calculator

    ecf = FrechetCellFilter(atoms)
    opt = FIRE(ecf)
    opt.run(fmax=fmax, steps=steps)

    return atoms


def get_reference_state(
    atoms,
    calculator,
    fmax=0.01,
    steps=800,
):
    """
    Relax structure and extract reference energy / volume.
    """
    relaxed_atoms = relax_structure(
        atoms=atoms,
        calculator=calculator,
        fmax=fmax,
        steps=steps,
    )

    e0 = relaxed_atoms.get_potential_energy()
    v0 = relaxed_atoms.get_volume()

    return {
        "relaxed_atoms": relaxed_atoms,
        "E0": e0,
        "V0": v0,
    }