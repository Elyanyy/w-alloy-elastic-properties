from simstate_utils import simstate_to_ase_atoms


def compute_total_energy(atoms, calculator):
    """
    Compute total energy using ASE + UMA calculator.
    """
    atoms = atoms.copy()
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    return energy


def compute_energies_for_records(records, calculator):
    """
    For each deformation record, convert state -> ASE atoms and compute energy.
    """
    out = []

    for rec in records:
        atoms = simstate_to_ase_atoms(rec["state"])
        energy = compute_total_energy(atoms, calculator)

        row = dict(rec)
        row["energy"] = float(energy)
        out.append(row)

    return out


def add_energy_density(records_with_energy, E0, V0):
    """
    Add strain energy density: (E - E0) / V0
    """
    out = []
    for rec in records_with_energy:
        row = dict(rec)
        row["energy_density"] = (row["energy"] - E0) / V0
        out.append(row)
    return out