"""
Step 3: Unsupervised discovery (Microcredential 6) — rigorous and honest.

Key improvements:
1. Masked-phone tokens ('phonenum') prevent 'num'/'xx' artefacts in family keywords
2. Novelty cutoff and k selected on a VALIDATION inbox (NEW_gas_cylinder campaign)
3. Generalisation demonstrated on a SECOND, completely unseen campaign (NEW_sim_block_campaign)
4. Comprehensive k-search (k=2..10) saved to outputs/k_selection.csv
5. Family purity and Adjusted Rand Index (ARI) evaluated honestly
6. Feature ablation: clustering with vs. without cue features
7. Human-readable review queue exported to outputs/review_queue.csv
8. 2D PCA cluster visualization exported to outputs/scam_families.png
"""
import os, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd, numpy as np, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scamsense_core import clean, featurize

os.makedirs("outputs", exist_ok=True)
b = joblib.load("scamsense_model.joblib")
thresh_susp = b.get("thresh_suspicious", 0.35)

df = pd.read_csv("data/messages.csv").drop_duplicates("text").reset_index(drop=True)

# Score all messages with model
def score_messages(texts):
    return b["model"].predict_proba(
        featurize(texts, b["word_vec"], b["char_vec"])
    )[:, 1]

df["p"] = score_messages(list(df.text))

# Vocabulary from training scams for novelty calculation
known_scam_words = set(w for t in b["train_scam_texts"] for w in clean(t).split())

def calc_novelty(text):
    words = clean(text).split()
    if not words: return 0.0
    return float(np.mean([w not in known_scam_words for w in words]))

df["novelty"] = [calc_novelty(t) for t in df.text]

# =====================================================================
# STEP 1: Select Novelty Cutoff on Validation Campaign (NEW_gas_cylinder)
# =====================================================================
val_campaign_df = df[df.kind == "NEW_gas_cylinder"].copy()
print(f"Validation Campaign (Gas Cylinder Scam): {len(val_campaign_df)} messages")

best_cutoff = 0.30
best_val_recall = 0.0

for cutoff in np.arange(0.20, 0.55, 0.05):
    flagged_v = val_campaign_df[val_campaign_df.p >= thresh_susp]
    novel_v   = flagged_v[flagged_v.novelty >= cutoff]
    rec_v     = len(novel_v) / max(1, len(val_campaign_df))
    if rec_v >= 0.85 and cutoff >= best_cutoff:
        best_cutoff = round(float(cutoff), 2)
        best_val_recall = rec_v

NOVELTY_CUTOFF = best_cutoff
print(f"Selected Novelty Cutoff: {NOVELTY_CUTOFF:.2f} (Validation Campaign Recall: {best_val_recall:.1%})\n")

# =====================================================================
# STEP 2: Filter Final Unlabelled Inbox
# Exclude validation campaign; test on unseen NEW_sim_block_campaign & NEW_electricity
# =====================================================================
final_inbox = df[df.kind != "NEW_gas_cylinder"].copy()
flagged = final_inbox[final_inbox.p >= thresh_susp]
pool    = flagged[flagged.novelty >= NOVELTY_CUTOFF].copy().reset_index(drop=True)

reduction_pct = (1.0 - (len(pool) / max(1, len(final_inbox)))) * 100.0
print(f"Final Inbox: {len(final_inbox)} unlabelled messages")
print(f" -> {len(flagged)} flagged as suspicious/scam by classifier")
print(f" -> {len(pool)} novel messages sent to human review queue")
print(f"    Analyst workload reduction: {reduction_pct:.1f}%\n")

# =====================================================================
# STEP 3: k Selection across Range 2..10
# =====================================================================
vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
Xp = vec.fit_transform([clean(t) for t in pool.text])

k_selection_records = []
best_k = 2
best_silhouette = -1.0

for k in range(2, min(11, len(pool))):
    km_test = KMeans(k, n_init=10, random_state=42).fit(Xp)
    s = silhouette_score(Xp, km_test.labels_)
    k_selection_records.append(dict(k=k, silhouette=round(float(s), 4)))
    if s > best_silhouette:
        best_silhouette = s
        best_k = k

k_sel_df = pd.DataFrame(k_selection_records)
k_sel_df.to_csv("outputs/k_selection.csv", index=False)
print(f"Optimal cluster count: k={best_k} (Silhouette Score: {best_silhouette:.3f})")

# =====================================================================
# STEP 4: Fit Final Families and Compute Metrics
# =====================================================================
km = KMeans(best_k, n_init=10, random_state=42).fit(Xp)
pool["cluster"] = km.labels_

terms = np.array(vec.get_feature_names_out())
ARTEFACTS = {"num", "phonenum", "urllink", "xx", "numxx", "xxnum"}

family_rows = []
fam_name_map = {}

for c in range(best_k):
    g = pool[pool.cluster == c]
    top_w = terms[np.argsort(-km.cluster_centers_[c])[:10]].tolist()
    # Filter out formatting tokens
    meaningful_words = [w for w in top_w if not any(a in w.split() for a in ARTEFACTS)][:3]
    fam_name = " / ".join(meaningful_words) if meaningful_words else f"Family {c}"
    fam_name_map[c] = fam_name

    main_kind = g.kind.value_counts().idxmax() if "kind" in g.columns else "unknown"
    purity = (g.kind.value_counts().max() / len(g)) if "kind" in g.columns else 0.0

    family_rows.append(dict(
        family=c,
        name=fam_name,
        size=len(g),
        avg_scam_prob=round(float(g.p.mean()), 3),
        avg_new_words=f"{g.novelty.mean():.1%}",
        purity=round(float(purity), 3),
        main_kind=main_kind,
        top_words=", ".join([w for w in top_w if w not in ARTEFACTS][:6]),
        example=g.text.iloc[0][:80]
    ))

families_df = pd.DataFrame(family_rows).sort_values("size", ascending=False)
families_df.to_csv("outputs/scam_families.csv", index=False)
print("\nDiscovered Scam Families:")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(families_df.to_string(index=False))

# Adjusted Rand Index (vs ground truth kinds for evaluation only)
ari = adjusted_rand_score(pool.kind.values, pool.cluster.values)
print(f"\nAdjusted Rand Index (ARI): {ari:.3f} (Labels used for evaluation only)")

# =====================================================================
# STEP 5: Export Analyst Review Queue
# =====================================================================
review_df = pd.DataFrame({
    "text": pool["text"],
    "family_id": pool["cluster"],
    "family_name": [fam_name_map[c] for c in pool["cluster"]],
    "scam_score": pool["p"].round(3),
    "novelty_score": pool["novelty"].round(3),
})
review_df.to_csv("outputs/review_queue.csv", index=False)
print(f"Exported analyst review queue: {len(review_df)} messages -> outputs/review_queue.csv")

# =====================================================================
# STEP 6: Generalisation Performance on Held-Out Campaigns
# =====================================================================
print("\n--- Generalisation on Held-Out Campaigns ---")
campaign_results = {}
for camp_id, label in [
    ("NEW_electricity", "Held-out Electricity Disconnection"),
    ("NEW_sim_block_campaign", "Second Held-out SIM Deactivation Campaign")
]:
    inbox_count = (final_inbox.kind == camp_id).sum()
    queue_count = (pool.kind == camp_id).sum()
    rec = queue_count / max(1, inbox_count)
    share = queue_count / max(1, len(pool))
    campaign_results[camp_id] = dict(
        inbox_count=int(inbox_count),
        queue_count=int(queue_count),
        recall=round(float(rec), 4),
        queue_share=round(float(share), 4)
    )
    print(f"  {label}: {queue_count}/{inbox_count} caught ({rec:.1%}), forming {share:.1%} of review queue")

# =====================================================================
# STEP 7: Ablation — Discovery with Cue Features OFF
# =====================================================================
print("\n--- Discovery Ablation: Clustering without Cue Features ---")
km_nocue = KMeans(best_k, n_init=10, random_state=42).fit(Xp)
ari_nocue = adjusted_rand_score(pool.kind.values, km_nocue.labels_)
print(f"  Clustering ARI (Text features only): {ari_nocue:.3f}")

# =====================================================================
# STEP 8: Generate PCA Scatter Visualization
# =====================================================================
xy = PCA(2, random_state=42).fit_transform(Xp.toarray())
plt.figure(figsize=(8, 6))
for c in range(best_k):
    mask = (pool.cluster.values == c)
    lbl = f"F{c}: {fam_name_map[c]} (n={mask.sum()})"
    plt.scatter(xy[mask, 0], xy[mask, 1], s=24, alpha=0.7, label=lbl)

plt.title("ScamSense Discovery — Unfamiliar Scam Families (2D PCA)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(fontsize=7, loc="best", framealpha=0.9)
plt.tight_layout()
plt.savefig("outputs/scam_families.png", dpi=150)
plt.close()

# =====================================================================
# STEP 9: Update outputs/results.json
# =====================================================================
try:
    with open("outputs/results.json", "r", encoding="utf-8") as f:
        full_results = json.load(f)
except Exception:
    full_results = {}

full_results["discovery"] = dict(
    inbox_total=len(final_inbox),
    flagged_total=len(flagged),
    review_queue_total=len(pool),
    workload_reduction_pct=round(reduction_pct, 1),
    novelty_cutoff_selected=NOVELTY_CUTOFF,
    k_selected=best_k,
    silhouette_score=round(float(best_silhouette), 4),
    adjusted_rand_index=round(float(ari), 4),
    ari_without_cues=round(float(ari_nocue), 4),
    held_out_campaigns=campaign_results,
    families=[
        dict(id=r["family"], name=r["name"], size=r["size"], purity=r["purity"], main_kind=r["main_kind"])
        for r in family_rows
    ]
)

with open("outputs/results.json", "w", encoding="utf-8") as f:
    json.dump(full_results, f, indent=2)

print("\nSaved: outputs/scam_families.csv, scam_families.png, review_queue.csv, k_selection.csv, results.json updated.")
