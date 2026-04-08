from config_5 import (
    MODEL_NAME,
    TASK_NAME,
    DEVICE,
    BASE_ELEMENT,
    A0_DICT,
    SUPERCELL,
    RANDOM_SEED,
    RELAX_FMAX,
    RELAX_STEPS,
    HYDRO_MAX_STRAIN,
    HYDRO_N_DEFORM,
    NORMAL_MAX_STRAIN,
    NORMAL_N_DEFORM,
    SHEAR_MAX_STRAIN,
    SHEAR_N_DEFORM,
    POLY_ORDER,
    RAW_DIR,
    PROCESSED_DIR,
    SEED_LIST
)

from alloy_builder import (
    build_random_substitutional_alloy,
    get_actual_solute_fraction,
)
from relax_utils import get_uma_calculator, get_reference_state
from simstate_utils import ase_atoms_to_simstate
from deformation_utils_3 import generate_all_deformation_records
from energy_utils import compute_energies_for_records, add_energy_density
from fit_utils_4 import fit_all_elastic_constants
from property_utils import enrich_elastic_results
from tm_utils import linear_melting_point
from io_utils import save_records_csv, save_json


def run_single_case(
    solute_element: str,
    nominal_solute_fraction: float,
    seed: int = RANDOM_SEED,
):
    """
    Run one W-X alloy case from build -> relax -> deformation -> fit -> properties.
    """
    print("=" * 80)
    print(f"Running case: W-{solute_element}, nominal x = {nominal_solute_fraction:.2f}, seed = {seed}")

    a0_mix = (
        (1.0 - nominal_solute_fraction) * A0_DICT[BASE_ELEMENT]
        + nominal_solute_fraction * A0_DICT[solute_element]
    )

    atoms0 = build_random_substitutional_alloy(
        base_element=BASE_ELEMENT,
        solute_element=solute_element,
        solute_fraction=nominal_solute_fraction,
        a0=a0_mix,
        supercell=SUPERCELL,
        seed=seed,
    )

    actual_x = get_actual_solute_fraction(atoms0, solute_element)

    calc = get_uma_calculator(
        model_name=MODEL_NAME,
        task_name=TASK_NAME,
        device=DEVICE,
    )

    ref = get_reference_state(
        atoms=atoms0,
        calculator=calc,
        fmax=RELAX_FMAX,
        steps=RELAX_STEPS,
    )

    relaxed_atoms = ref["relaxed_atoms"]
    E0 = ref["E0"]
    V0 = ref["V0"]

    reference_state = ase_atoms_to_simstate(
        relaxed_atoms,
        device=DEVICE,
    )

    records = generate_all_deformation_records(
        reference_state=reference_state,
        hydro_max_strain=HYDRO_MAX_STRAIN,
        hydro_n_deform=HYDRO_N_DEFORM,
        normal_axes=(0,),
        shear_axes=(5,),
        normal_max_strain=NORMAL_MAX_STRAIN,
        shear_max_strain=SHEAR_MAX_STRAIN,
        ns_n_deform=max(NORMAL_N_DEFORM, SHEAR_N_DEFORM),
    )
    records_e = compute_energies_for_records(records, calculator=calc)
    records_ed = add_energy_density(records_e, E0=E0, V0=V0)

    elastic_raw = fit_all_elastic_constants(
        records=records_ed,
        poly_order=POLY_ORDER,
    )

    elastic_full = enrich_elastic_results(elastic_raw)

    tm_linear = linear_melting_point(
        base_element=BASE_ELEMENT,
        solute_element=solute_element,
        solute_fraction=actual_x,
    )

    summary = {
    "base_element": BASE_ELEMENT,
    "solute_element": solute_element,
    "nominal_solute_fraction": nominal_solute_fraction,
    "actual_solute_fraction": actual_x,
    "seed": seed,
    "a0_mix": a0_mix,
    "E0_eV": E0,
    "V0_A3": V0,
    "Tm_linear_K": tm_linear,
}

    summary.update(elastic_full)


    EV_A3_TO_GPA = 160.21766208

    summary["C11_GPa"] = summary["C11_eVA3"] * EV_A3_TO_GPA
    summary["C12_GPa"] = summary["C12_eVA3"] * EV_A3_TO_GPA
    summary["C44_GPa"] = summary["C44_eVA3"] * EV_A3_TO_GPA
    stem = f"W_{solute_element}_x{nominal_solute_fraction:.2f}_seed{seed}"

    save_records_csv(
        records_ed,
        RAW_DIR / f"{stem}_records.csv",
    )

    save_json(
        summary,
        PROCESSED_DIR / f"{stem}_summary.json",
    )

    print("Done.")
    print(f"B  = {summary['B_GPa']:.3f} GPa")
    print(f"G  = {summary['G_GPa']:.3f} GPa")
    print(f"B/G = {summary['B_over_G']:.4f}")
    print(f"Tm = {summary['Tm_linear_K']:.2f} K")
    print(f"Born stable = {summary['born_stable']}")


    return {
        "summary": summary,
        "records": records_ed,
        "relaxed_atoms": relaxed_atoms,
    }


if __name__ == "__main__":
    for solute in ["Nb", "Ta"]:
        print("=" * 80)
        print(f"Testing pure {solute}")

        result = run_single_case(
            solute_element=solute,
            nominal_solute_fraction=1.0,
            seed=42
        )

        summary = result["summary"]
        print("shear_fit x =", summary["shear_fit"]["x"])
        print("shear_fit y =", summary["shear_fit"]["y"])
        print("shear_fit a2 =", summary["shear_fit"]["a2"])
        print("C44_GPa =", summary["C44_GPa"])
        print("G_GPa =", summary["G_GPa"])
        print("B_GPa =", summary["B_GPa"])
        print("B/G =", summary["B_over_G"])