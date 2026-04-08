import numpy as np
import torch
from ase import Atoms
from torch_sim.state import SimState


def ase_atoms_to_simstate(atoms_relaxed, device="cpu", dtype=torch.float64):
    """
    Convert ASE Atoms -> TorchSim SimState
    """
    positions = np.asarray(atoms_relaxed.get_positions(), dtype=np.float64)
    cell = np.asarray(atoms_relaxed.get_cell(), dtype=np.float64)
    masses = np.asarray(atoms_relaxed.get_masses(), dtype=np.float64)
    atomic_numbers = np.asarray(atoms_relaxed.get_atomic_numbers(), dtype=np.int64)
    pbc = np.asarray(atoms_relaxed.get_pbc(), dtype=bool)

    positions_t = torch.tensor(positions, dtype=dtype, device=device)
    masses_t = torch.tensor(masses, dtype=dtype, device=device)
    atomic_numbers_t = torch.tensor(atomic_numbers, dtype=torch.long, device=device)
    pbc_t = torch.tensor(pbc, dtype=torch.bool, device=device)
    cell_t = torch.tensor(cell.T[None, :, :], dtype=dtype, device=device)

    state = SimState(
        positions=positions_t,
        cell=cell_t,
        masses=masses_t,
        pbc=pbc_t,
        atomic_numbers=atomic_numbers_t,
    )
    return state


def simstate_to_ase_atoms(state: SimState):
    """
    Convert TorchSim SimState -> ASE Atoms

    Notes
    -----
    state.cell is stored as column-vector style with shape (1, 3, 3),
    so we transpose back to ASE row-vector cell.
    """
    positions = state.positions.detach().cpu().numpy()
    cell = state.cell.squeeze(0).mT.detach().cpu().numpy()
    masses = state.masses.detach().cpu().numpy()
    atomic_numbers = state.atomic_numbers.detach().cpu().numpy()
    pbc = state.pbc.detach().cpu().numpy()

    atoms = Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=pbc,
    )
    atoms.set_masses(masses)
    return atoms