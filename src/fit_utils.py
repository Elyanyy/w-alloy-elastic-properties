import numpy as np


def fit_polynomial(x, y, order=4):
    """
    Fit y(x) with np.polyfit and return coefficients in descending order.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    coeffs = np.polyfit(x, y, deg=order)
    return coeffs


def get_quadratic_coeff_from_polyfit(coeffs):
    """
    Extract the coefficient of x^2 from descending-order polynomial coefficients.
    Example:
        order=4 -> [a4, a3, a2, a1, a0]
        return a2
    """
    deg = len(coeffs) - 1
    idx = deg - 2
    return coeffs[idx]


def fit_family(records, family, strain_component_index, poly_order=4):
    """
    Fit one deformation family using the selected Voigt strain component.

    Parameters
    ----------
    records : list[dict]
    family : str
        'hydrostatic' / 'normal' / 'shear'
    strain_component_index : int or None
        0 for e1, 5 for e6, etc.
        For hydrostatic, we usually use e1 (same as e2=e3).
    poly_order : int

    Returns
    -------
    result : dict
    """
    subset = [r for r in records if r["family"] == family]
    if len(subset) < 3:
        raise ValueError(f"Not enough records for family={family}")

    x = []
    y = []
    for rec in subset:
        ev = rec["strain_voigt"]
        x.append(float(ev[strain_component_index]))
        y.append(float(rec["energy_density"]))

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    coeffs = fit_polynomial(x, y, order=poly_order)
    a2 = get_quadratic_coeff_from_polyfit(coeffs)

    return {
        "family": family,
        "x": x,
        "y": y,
        "coeffs": coeffs,
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
    return a2/2.0


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