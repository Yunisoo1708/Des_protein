"""Link the best-reward search candidate per design (by search.reward) into <out>/inference_samples/<design>."""
import os, sys, pandas as pd
run, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(f"{run}/inference_metrics/per_example_metrics.csv")
d["design"] = d["example_id"].str.extract(r"^(.*)_bsc\d+_\d+$")[0]
best = d.sort_values("search.reward", ascending=False).groupby("design").head(1)
assert best.design.is_unique and len(best) == d.design.nunique()
os.makedirs(f"{out}/inference_samples", exist_ok=True)
for _, r in best.iterrows():
    os.symlink(f"{run}/inference_samples/{r.example_id}", f"{out}/inference_samples/{r.design}")
best[["design", "example_id", "search.reward"]].to_csv(f"{out}/selected_candidates.csv", index=False)
print(f"{len(d)} candidates, {d.design.nunique()} designs, linked {len(best)} best; "
      f"reward of linked: median {best['search.reward'].median():.1f}, all-candidate median {d['search.reward'].median():.1f}")
