import torch
from torch_sim.state import SimState
from elastic import get_strain


def get_cart_deformed_cell(state: SimState, axis: int = 0, size: float = 1.0) -> SimState:
    """
    Deform a unit cell and scale atomic positions accordingly.

    axis:
        0,1,2 -> x,y,z normal deformation
        3,4,5 -> yz,xz,xy shear deformation
    """
    row_vector_cell = state.row_vector_cell.squeeze()
    positions = state.positions

    if not (0 <= axis <= 5):
        raise ValueError("Axis must be between 0 and 5")
    if row_vector_cell.shape != (3, 3):
        raise ValueError("Cell must be a 3x3 tensor")
    if positions.shape[-1] != 3:
        raise ValueError("Positions must have shape (n_atoms, 3)")

    L = torch.eye(3, dtype=state.dtype, device=state.device)

    if axis < 3:
        L[axis, axis] += size
    elif axis == 3:
        L[1, 2] = size
        L[2, 1] = size
        L[0, 0] = 1.0 / (1.0 - size**2)
    elif axis == 4:
        L[0, 2] = size
        L[2, 0] = size
        L[1, 1] = 1.0 / (1.0 - size**2)
    else:
        L[0, 1] = size
        L[1, 0] = size
        L[2, 2] = 1.0 / (1.0 - size**2)

    old_inv = torch.linalg.inv(row_vector_cell)
    frac_coords = torch.matmul(positions, old_inv)

    new_cell = torch.matmul(row_vector_cell, L)
    new_positions = torch.matmul(frac_coords, new_cell)

    return SimState(
        positions=new_positions,
        cell=new_cell.mT.unsqueeze(0),
        masses=state.masses,
        pbc=state.pbc,
        atomic_numbers=state.atomic_numbers,
    )


def get_hydrostatic_deformed_cell(state: SimState, size: float) -> SimState:
    """
    Apply isotropic hydrostatic deformation.
    """
    row_vector_cell = state.row_vector_cell.squeeze()
    positions = state.positions

    F = torch.eye(3, dtype=state.dtype, device=state.device) * (1.0 + size)

    old_inv = torch.linalg.inv(row_vector_cell)
    frac_coords = torch.matmul(positions, old_inv)

    new_cell = torch.matmul(row_vector_cell, F)
    new_positions = torch.matmul(frac_coords, new_cell)

    deformed_state = SimState(
        positions=new_positions,
        cell=new_cell.mT.unsqueeze(0),
        masses=state.masses,
        pbc=state.pbc,
        atomic_numbers=state.atomic_numbers,
    )
    return deformed_state


def generate_hydrostatic_deformations(reference_state, max_strain=0.04, n_deform=9):
    """
    Generate hydrostatic deformations for bulk modulus fitting.
    """
    strains = torch.linspace(
        -max_strain,
        max_strain,
        n_deform,
        dtype=reference_state.dtype,
        device=reference_state.device,
    )
    strains = strains[torch.abs(strains) > 1e-12]

    records = []
    for strain in strains:
        deformed_state = get_hydrostatic_deformed_cell(
            state=reference_state,
            size=float(strain.item())
        )

        record = {
            "family": "hydrostatic",
            "axis": None,
            "applied_size": float(strain.item()),
            "state": deformed_state,
        }
        records.append(record)

    return records


def generate_energy_strain_deformations(
    reference_state,
    normal_axes=(0,),
    shear_axes=(5,),
    max_strain_normal=0.04,
    max_strain_shear=0.08,
    n_deform=9,
):
    """
    Generate normal and shear deformations for C11 / C44 fitting.
    """
    records = []

    normal_strains = torch.linspace(
        -max_strain_normal,
        max_strain_normal,
        n_deform,
        dtype=reference_state.dtype,
        device=reference_state.device,
    )
    normal_strains = normal_strains[torch.abs(normal_strains) > 1e-12]

    for axis in normal_axes:
        for strain in normal_strains:
            deformed_state = get_cart_deformed_cell(
                state=reference_state,
                axis=axis,
                size=float(strain.item())
            )

            record = {
                "family": "normal",
                "axis": axis,
                "applied_size": float(strain.item()),
                "state": deformed_state,
            }
            records.append(record)

    shear_strains = torch.linspace(
        -max_strain_shear,
        max_strain_shear,
        n_deform,
        dtype=reference_state.dtype,
        device=reference_state.device,
    )
    shear_strains = shear_strains[torch.abs(shear_strains) > 1e-12]

    for axis in shear_axes:
        for strain in shear_strains:
            deformed_state = get_cart_deformed_cell(
                state=reference_state,
                axis=axis,
                size=float(strain.item())
            )

            record = {
                "family": "shear",
                "axis": axis,
                "applied_size": float(strain.item()),
                "state": deformed_state,
            }
            records.append(record)

    return records


def attach_strain_to_records(records, reference_state):
    """
    Compute actual Voigt strain relative to the reference state.
    """
    for rec in records:
        strain_voigt = get_strain(
            deformed_state=rec["state"],
            reference_state=reference_state
        )
        rec["strain_voigt"] = strain_voigt.detach().cpu().numpy()

    return records


def generate_all_deformation_records(
    reference_state,
    hydro_max_strain=0.01,
    hydro_n_deform=9,
    normal_axes=(0,),
    shear_axes=(5,),
    normal_max_strain=0.01,
    shear_max_strain=0.01,
    ns_n_deform=9,
):
    """
    Generate hydro + normal + shear deformation records and attach actual strains.
    """
    hydro_records = generate_hydrostatic_deformations(
        reference_state=reference_state,
        max_strain=hydro_max_strain,
        n_deform=hydro_n_deform,
    )

    ns_records = generate_energy_strain_deformations(
        reference_state=reference_state,
        normal_axes=normal_axes,
        shear_axes=shear_axes,
        max_strain_normal=normal_max_strain,
        max_strain_shear=shear_max_strain,
        n_deform=ns_n_deform,
    )

    all_records = hydro_records + ns_records
    all_records = attach_strain_to_records(all_records, reference_state)

    return all_records