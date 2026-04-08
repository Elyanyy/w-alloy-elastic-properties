from constants import EV_A3_TO_GPA


def ev_a3_to_gpa(x):
    return x * EV_A3_TO_GPA


def compute_vrh_shear_modulus(C11, C12, C44):
    """
    Cubic crystal:
        G_V = (C11 - C12 + 3*C44)/5
        G_R = 5(C11 - C12)C44 / [4C44 + 3(C11 - C12)]
        G_H = (G_V + G_R)/2
    """
    GV = (C11 - C12 + 3.0 * C44) / 5.0
    denom = 4.0 * C44 + 3.0 * (C11 - C12)

    if abs(denom) < 1e-14:
        GR = float("nan")
    else:
        GR = 5.0 * (C11 - C12) * C44 / denom

    GH = 0.5 * (GV + GR)
    return GV, GR, GH


def compute_B_over_G(B, G):
    if abs(G) < 1e-14:
        return float("nan")
    return B / G


def check_born_stability(C11, C12, C44):
    """
    Cubic Born stability:
        1) C11 - C12 > 0
        2) C11 + 2*C12 > 0
        3) C44 > 0
    """
    c1 = (C11 - C12) > 0.0
    c2 = (C11 + 2.0 * C12) > 0.0
    c3 = C44 > 0.0

    return {
        "born_c1": c1,
        "born_c2": c2,
        "born_c3": c3,
        "born_stable": bool(c1 and c2 and c3),
    }


def enrich_elastic_results(elastic_dict):
    """
    Add G, B/G, GPa conversions, and stability.
    """
    B = elastic_dict["B_eVA3"]
    C11 = elastic_dict["C11_eVA3"]
    C12 = elastic_dict["C12_eVA3"]
    C44 = elastic_dict["C44_eVA3"]

    GV, GR, GH = compute_vrh_shear_modulus(C11, C12, C44)
    B_over_G = compute_B_over_G(B, GH)
    stability = check_born_stability(C11, C12, C44)

    out = dict(elastic_dict)

    out["GV_eVA3"] = GV
    out["GR_eVA3"] = GR
    out["G_eVA3"] = GH
    out["B_over_G"] = B_over_G

    out["B_GPa"] = ev_a3_to_gpa(B)
    out["C11_GPa"] = ev_a3_to_gpa(C11)
    out["C12_GPa"] = ev_a3_to_gpa(C12)
    out["C44_GPa"] = ev_a3_to_gpa(C44)
    out["GV_GPa"] = ev_a3_to_gpa(GV)
    out["GR_GPa"] = ev_a3_to_gpa(GR)
    out["G_GPa"] = ev_a3_to_gpa(GH)

    out.update(stability)
    return out