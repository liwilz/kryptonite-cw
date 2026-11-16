"""Energy and emissions: the CPU energy CodeCarbon measured for every stored result, by
experiment.

    python energy_report.py"""

import numpy as np

from utils import CV, FIN, FOLD_TRAIN, K, LC, MEASURED, NS, SEED, WTS, _tracker, text_table

GRID_G_PER_KWH = 125  # NESO, "Britain's Electricity Explained: 2024 Review": GB average for 2024


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
        BASE="Polynomial LR on all bits",
        LR="Polynomial LR on all bits",
        KPOLY="Kernel LR on all bits",
        RF="Random forest on all bits",
        FOLD="MLP 4×512",
        MLP="MLP 4×512",
        LC="Learning curves",
        UK="Kernel on all bits, equal weights",
        WK="Kernel on all bits, forest weights",
        CONTROL="Final pipeline: controls and locked test",
        FINAL="Final pipeline: controls and locked test",
        HIDDEN="Hidden-set refits",
        ABL="MLP ablations",
    )

    def name(r):
        if r.get("exp") == "WTS":
            return "Bit weights"
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
        f" {GRID_G_PER_KWH} g per kWh (NESO, GB 2024 average)."
    )

    def cv_cost(pick):
        return sum(r["energy"]["cpu_kwh"] * 1000 for r in rows if pick(r))

    for n in NS:
        wk = cv_cost(
            lambda r: r.get("n") == n
            and (
                r.get("exp") == "WTS"
                and r.get("seed", SEED) == SEED
                and r.get("rows") == FOLD_TRAIN[n]
                or r.get("exp") == "WK"
            )
        )
        mlp = cv_cost(lambda r: r.get("n") == n and r.get("exp") in ("FOLD", "MLP"))
        print(
            f"n = {n}: {K}-fold CV of the weighted kernel (weights + its grid) {wk:.1f} Wh;"
            f" of the MLP {mlp:.1f} Wh"
        )


if __name__ == "__main__":
    carbon_report()
