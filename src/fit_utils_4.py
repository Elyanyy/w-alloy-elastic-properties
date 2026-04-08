import numpy as np


def fit_family(records, family, strain_component_index, poly_order=2):
    """
    Fit one deformation family using E vs strain^2 (linear fit).

    Returns
    -------
    result : dict
    """
    subset = [r for r in records if r["family"] == family]
    if len(subset) < 3:
        raise ValueError(f"Not enough records for family={family}")

    x = []
    y = []
    applied_sizes = []
    for rec in subset:
        ev = rec["strain_voigt"]
        x.append(float(ev[strain_component_index]))
        y.append(float(rec["energy_density"]))
        applied_sizes.append(rec["applied_size"]) 
    if family == "shear":
        print("\n=== DEBUG SHEAR ===")
        print("applied_size =", applied_sizes)
        print("strain_voigt =", x)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        raise ValueError(f"Not enough fit points for family={family}. x={x}")

    coeffs = np.polyfit(x, y, 2)
    a2 = coeffs[0]

    return {
        "family": family,
        "x": x,
        "y": y,
        "a2": a2,
    }


def extract_B_from_hydro_a2(a2):
    """
    Hydrostatic:
        ΔE/V0 = 9/2 * B * δ^2
    so:
        B = a2 / 4.5
    """
    return a2 / 4.5


def extract_C11_from_normal_a2(a2):
    """
    Normal:
        ΔE/V0 = 1/2 * C11 * δ^2
    so:
        C11 = 2 * a2
    """
    return 2.0 * a2


def extract_C44_from_shear_a2(a2):
    """
    Shear:
        ΔE/V0 = 1/2 * C44 * (2δ)^2 = 2 * C44 * δ^2
    so:
        C44 = a2 / 2
    """
    return a2/2


def compute_C12_from_B_C11(B, C11):
    """
    B = (C11 + 2*C12)/3
    => C12 = (3B - C11)/2
    """
    return (3.0 * B - C11) / 2.0


def fit_all_elastic_constants(records, poly_order=4):
    """
    Fit hydrostatic / normal / shear families and extract
    B, C11, C12, C44 in eV/A^3.
    """
    hydro_fit = fit_family(
        records=records,
        family="hydrostatic",
        strain_component_index=0,
        poly_order=poly_order,
    )

    normal_fit = fit_family(
        records=records,
        family="normal",
        strain_component_index=0,
        poly_order=poly_order,
    )

    shear_fit = fit_family(
        records=records,
        family="shear",
        strain_component_index=5,
        poly_order=poly_order,
    )

    B = extract_B_from_hydro_a2(hydro_fit["a2"])
    C11 = extract_C11_from_normal_a2(normal_fit["a2"])
    C44 = extract_C44_from_shear_a2(shear_fit["a2"])
    C12 = compute_C12_from_B_C11(B, C11)

    return {
        "hydro_fit": hydro_fit,
        "normal_fit": normal_fit,
        "shear_fit": shear_fit,
        "B_eVA3": B,
        "C11_eVA3": C11,
        "C12_eVA3": C12,
        "C44_eVA3": C44,
    }