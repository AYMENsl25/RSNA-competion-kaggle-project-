"""Labels v3 = 0.5 * consensus report labels + 0.5 * teacher OOF (quantile-mapped onto the consensus distribution).

python make_labels_v3.py <teacher_output_dir> <labels_k1.json> <out_labels_v3.json>

Gold-58 check (never trained on): teacher = mean of the 4 fold models' gold predictions, consensus as in run.py.
Writes labels_v3.json in the labels_k1.json format, so the student runner can use it unchanged. Gate: v3 > consensus on gold.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
root, lab_path, out_path = sys.argv[1:4]
gold_cons = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_consensus.csv")
lab = json.load(open(lab_path))
cons = pd.DataFrame({u: [np.nan if v is None else v for v in y] for u, y in lab["y"].items()}, index=LAB).T
oof = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(root, "**", "oof_fold*.csv"), recursive=True)]).set_index("StudyInstanceUID")[LAB]
assert len(oof) >= 0.95 * len(cons), (len(oof), len(cons))


def qmap(t, ref):
    """rank of t mapped onto the empirical quantiles of ref (keeps the label scale, teacher ordering)."""
    r = t.rank(pct=True)
    return pd.Series(np.quantile(ref.dropna().values, r.clip(0, 1).values), index=t.index)


teach = pd.DataFrame({c: qmap(oof[c], cons[c]) for c in LAB})
c_al = cons.reindex(teach.index)
v3 = (0.5 * c_al + 0.5 * teach).where(c_al.notna(), teach).clip(0.01, 0.99)

# gold check: the fold models never saw gold; consensus gold values come from the public tables like run.py
gy = pd.DataFrame(lab["gold_y"], index=LAB).T
last = {}
for f in sorted(glob.glob(os.path.join(root, "**", "gold_ep*.npy"), recursive=True)):
    d = os.path.dirname(f)
    e = int(os.path.basename(f)[7:-4])
    if e >= last.get(d, (-1, None))[0]:
        last[d] = (e, f)
G = pd.DataFrame(np.mean([np.load(f) for _, f in last.values()], 0), index=lab["gold"], columns=LAB)
cg = pd.read_csv(gold_cons, index_col=0).reindex(G.index)[LAB] if os.path.exists(gold_cons) else None
rows = {"teacher_mean4": [roc_auc_score(gy.loc[G.index, c], G[c]) for c in LAB]}
if cg is not None:
    tg = pd.DataFrame({c: qmap(G[c], cons[c]) for c in LAB})
    rows["consensus"] = [roc_auc_score(gy.loc[G.index, c], cg[c].fillna(0.5)) for c in LAB]
    rows["v3"] = [roc_auc_score(gy.loc[G.index, c], (0.5 * cg[c].fillna(tg[c]) + 0.5 * tg[c])) for c in LAB]
t = pd.DataFrame(rows, index=LAB)
t.loc["MEAN"] = t.mean()
print(t.round(3).to_string())
out = dict(lab, y={u: [None if np.isnan(x) else round(float(x), 4) for x in v3.loc[u].values] for u in v3.index},
           sources=lab.get("sources", []) + ["teacher_oof_t1"])
json.dump(out, open(out_path, "w"))
print("wrote", out_path, len(out["y"]), "studies")
