import pandas as pd
import statsmodels.formula.api as smf
from .config import PROCESSED, RESULTS

def adjusted_means(df, trait):
    d = df.loc[df["Trait"] == trait].dropna(subset=["Value"]).copy()
    if d["Genotype"].nunique() < 3:
        raise ValueError("At least three genotypes are needed for this starter analysis.")
    # Starter fixed-effects model: environment, genotype, environment:genotype,
    # and replication nested in environment. Adapt to actual experimental design.
    d["EnvRep"] = d["Environment"].astype(str) + ":" + d["Replication"].astype(str)
    fit = smf.ols(
        "Value ~ C(Environment) + C(EnvRep) + C(Genotype) + C(Environment):C(Genotype)",
        data=d
    ).fit()
    # Estimated marginal means are approximated here by predictions averaged
    # over observed environment/replication combinations for each genotype.
    grid = d[["Environment", "EnvRep"]].drop_duplicates()
    rows = []
    for gid in sorted(d["Genotype"].unique()):
        g = grid.copy()
        g["Genotype"] = gid
        pred = fit.predict(g)
        rows.append({"Genotype": gid, "Trait": trait, "Adjusted_Value": float(pred.mean())})
    return pd.DataFrame(rows), fit

def main():
    raw = pd.read_csv(PROCESSED / "phenotypes_raw.csv")
    RESULTS.mkdir(parents=True, exist_ok=True)
    outputs = []
    for trait in sorted(raw["Trait"].dropna().unique()):
        means, _ = adjusted_means(raw, trait)
        outputs.append(means)
    result = pd.concat(outputs, ignore_index=True)
    result.to_csv(RESULTS / "adjusted_genotype_means.csv", index=False)
    print(result.head())
    print("Adjusted genotype means saved. Confirm this model matches your field design.")
if __name__ == "__main__":
    main()
