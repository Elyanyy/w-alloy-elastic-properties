import numpy as np
from ase.build import bulk


def build_pure_bcc(
    element: str = "W",
    a0: float = 3.17,
    supercell=(4, 4, 4),
):
    """
    Build a bcc supercell using ASE bulk().
    """
    atoms = bulk(element, crystalstructure="bcc", a=a0, cubic=True)
    atoms = atoms.repeat(supercell)
    return atoms


def build_random_substitutional_alloy(
    base_element: str,
    solute_element: str,
    solute_fraction: float,
    a0: float,
    supercell=(4, 4, 4),
    seed: int = 42,
):
    """
    Build a substitutional W-X alloy by randomly replacing base atoms.

    Parameters
    ----------
    base_element : str
        Base element, e.g. "W"
    solute_element : str
        Solute element, e.g. "Ta"
    solute_fraction : float
        Fraction of solute atoms, between 0 and 1
    a0 : float
        Initial lattice parameter
    supercell : tuple
        Supercell replication
    seed : int
        Random seed

    Returns
    -------
    atoms : ase.Atoms
        Alloy structure
    """
    if not (0.0 <= solute_fraction <= 1.0):
        raise ValueError("solute_fraction must be between 0 and 1.")

    atoms = build_pure_bcc(
        element=base_element,
        a0=a0,
        supercell=supercell,
    )

    n_atoms = len(atoms)
    n_solute = int(round(solute_fraction * n_atoms))

    rng = np.random.default_rng(seed)
    indices = np.arange(n_atoms)

    if n_solute > 0:
        chosen = rng.choice(indices, size=n_solute, replace=False)
        symbols = atoms.get_chemical_symbols()
        for idx in chosen:
            symbols[idx] = solute_element
        atoms.set_chemical_symbols(symbols)

    return atoms


def get_composition_dict(atoms):
    """
    Return a composition dictionary from ASE Atoms.
    """
    symbols = atoms.get_chemical_symbols()
    comp = {}
    for s in symbols:
        comp[s] = comp.get(s, 0) + 1
    total = len(symbols)
    for k in comp:
        comp[k] = comp[k] / total
    return comp


def get_actual_solute_fraction(atoms, solute_element: str):
    """
    Return the actual solute fraction in the built alloy.
    """
    comp = get_composition_dict(atoms)
    return comp.get(solute_element, 0.0)