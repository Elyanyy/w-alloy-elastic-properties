from constants import MELTING_POINTS_K


def linear_melting_point(base_element: str, solute_element: str, solute_fraction: float):
    """
    Linear rule of mixtures:
        Tm = (1-x)*Tm(base) + x*Tm(solute)
    """
    tm_base = MELTING_POINTS_K[base_element]
    tm_solute = MELTING_POINTS_K[solute_element]
    return (1.0 - solute_fraction) * tm_base + solute_fraction * tm_solute