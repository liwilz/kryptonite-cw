"""Energy and emissions: the CPU energy CodeCarbon measured for every stored result, by
experiment.

    python energy_report.py"""

import numpy as np

from utils import CV, FIN, LC, MEASURED, WTS, _tracker, text_table

# UK electricity consumed, including grid losses: 0.14396 kgCO2e/kWh for data year 2025 (DESNZ,
# 2026 Government GHG conversion factors for company reporting), the factor the LoRA notebook uses
GRID_G_PER_KWH = 143.96


def carbon_report():
    """Energy measured by CodeCarbon for every stored result, by experiment, and the emissions at
    GRID_G_PER_KWH."""
    rows = [r for st in (CV, LC, WTS, FIN) for r in st.rows.values()]
    missing = [r for r in rows if "energy" not in r]
    rows = [r for r in rows if r.get("energy")]
    modes = {r["energy"]["cpu_mode"] for r in rows}
    if missing or not modes or not modes <= set(MEASURED):
        print(
            f"Not reported: {len(missing)} results have no energy record and the CPU modes are"
            f" {sorted(map(str, modes))}; energy is reported only when every result was measured"
            f" by the CPU's counters ({', '.join(MEASURED)})."
        )
        return
    names = dict(
        BASE="Baseline (degree 2)",
        KPOLY="Their kernel",
        RF="Random forest",
        FOLD="MLP 4×512",
        MLP="MLP 4×512",
        ABL="MLP ablations",
        WTS="MANBAT: bit weights",
        WK="MANBAT",
        UK="Equal-weight kernel",
        LC="Learning curves",
        LCT="Learning curves",
        CONTROL="Controls",
        FINAL="Test scores",
        TEST="Test scores",
        HIDDEN="Hidden-set refits",
    )

    def name(r):
        return names.get(r.get("exp"), str(r.get("exp")))

    tot = {}
    for r in rows:
        t = tot.setdefault(name(r), np.zeros(2))
        t += [r.get("secs", 0) / 3600, r["energy"]["cpu_kwh"] * 1000]
    total = np.sum(list(tot.values()), axis=0)

    def line(k, v):
        return [k, f"{v[0]:.2f}", f"{v[1]:.1f}", f"{v[1] * GRID_G_PER_KWH / 1000:.1f}"]

    out = [line(k, v) for k, v in tot.items()] + [line("Total", total)]
    text_table(["", "hours", "CPU Wh", "g CO2e"], out, "Measured CPU energy by experiment")
    cpus = {r["energy"].get("cpu") for r in rows} - {None}
    if not cpus:  # results stored before the CPU name was recorded: ask CodeCarbon now
        tr = _tracker()[0]
        tr.start_task()
        cpus = {tr.stop_task().cpu_model}
    print(
        f"CPU: {', '.join(sorted(cpus))}; energy read from its own counters"
        f" ({', '.join(sorted(modes))}).\nEvery model runs on the CPU only, so no GPU energy is"
        " counted; RAM and screen are not measured.\ng CO2e at"
        f" {GRID_G_PER_KWH} g per kWh (DESNZ, UK 2025 average, including grid losses)."
    )


if __name__ == "__main__":
    carbon_report()
