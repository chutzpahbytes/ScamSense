import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
"""
Step 2: Train, compare two models, evaluate rigorously, save.
Includes:
- Train / Val / Test split by sentence template (leakage-free)
- Threshold selection on validation set
- 5-fold GroupKFold cross-validation by template (mean ± std)
- Feature ablation study (Word vs Word+Char vs Full Model)
- Realistic prevalence evaluation (5% scam scenario)
- Evaluation on independent realistic test set
- Export to outputs/results.json and outputs/confusion_matrix.png
"""
import pandas as pd, numpy as np, joblib, os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from scipy.sparse import hstack
from scamsense_core import clean, featurize, cue_matrix, explain

BASELINE = dict(accuracy=0.939, precision=1.0, recall=0.897, f1=0.946, unseen_caught=1.0)

df = pd.read_csv("data/messages.csv")
print(f"Total rows: {len(df)}  |  duplicates: {df.duplicated('text').sum()}")
df = df.drop_duplicates("text")

# Held-out campaigns — NEVER touched during training or hyperparameter tuning
unseen_elec = df[df.kind == "NEW_electricity"]
unseen_gas  = df[df.kind == "NEW_gas_cylinder"]
unseen_sim  = df[df.kind == "NEW_sim_block_campaign"]

known = df[~df.kind.isin(["NEW_electricity", "NEW_gas_cylinder", "NEW_sim_block_campaign"])].copy()

# --- Split by template: train / validation / test ---
val_templates, test_templates = set(), set()
for kind, g in known.groupby("kind"):
    ids = sorted(g.template.unique())
    if len(ids) == 1:
        # If only 1 template exists, put into train
        pass
    elif len(ids) == 2:
        test_templates.add(ids[1])
    elif len(ids) == 3:
        test_templates.add(ids[2])
        val_templates.add(ids[1])
    else:
        # 4+ templates: hold out every 3rd for test, every 4th for validation
        for i, t in enumerate(ids):
            if (i + 1) % 3 == 0:
                test_templates.add(t)
            elif (i + 1) % 4 == 0:
                val_templates.add(t)

test  = known[known.template.isin(test_templates)].copy()
val   = known[known.template.isin(val_templates)].copy()
train = known[~known.template.isin(test_templates | val_templates)].copy()

print(f"Dataset split: Train {len(train)} | Val {len(val)} | Test {len(test)}")
print(f"Held-out campaigns: Electricity {len(unseen_elec)} | Gas {len(unseen_gas)} | SIM-block {len(unseen_sim)}")

# --- Fit vectorizers on train only (zero leakage) ---
word_vec = TfidfVectorizer(ngram_range=(1,2), min_df=2)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=3)
ctrain = [clean(t) for t in train.text]
word_vec.fit(ctrain); char_vec.fit(ctrain)

Xtr = featurize(train.text, word_vec, char_vec)
Xva = featurize(val.text,   word_vec, char_vec)
Xte = featurize(test.text,  word_vec, char_vec)
Xun = featurize(unseen_elec.text, word_vec, char_vec)

ytr, yva, yte, yun = train.label.values, val.label.values, test.label.values, unseen_elec.label.values

# --- Fit Candidate Models ---
lr = LogisticRegression(max_iter=2000, C=3, class_weight="balanced", random_state=42)
lr.fit(Xtr, ytr)
dt = DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=1)
dt.fit(Xtr, ytr)

# --- Determine Discriminating Words for Interpretability ---
# Appears in >= 3 scam training documents and < 3% of safe ones
cv = CountVectorizer(ngram_range=(1,2), min_df=2, vocabulary=word_vec.vocabulary_)
train_scam_texts = list(train[train.label == 1].text)
train_safe_texts = list(train[train.label == 0].text)
Xs = cv.transform([clean(t) for t in train_scam_texts])
Xsf = cv.transform([clean(t) for t in train_safe_texts])
scam_doc_count = (Xs > 0).sum(axis=0).A1
safe_doc_count = (Xsf > 0).sum(axis=0).A1
n_safe_tr = max(1, len(train_safe_texts))

scam_discriminating = set(
    w for w, i in word_vec.vocabulary_.items()
    if scam_doc_count[i] >= 3 and (safe_doc_count[i] / n_safe_tr) < 0.03
)
print(f"Scam-discriminating terms identified: {len(scam_discriminating)}")

# --- Select Thresholds on Validation Set ---
probs_val = lr.predict_proba(Xva)[:, 1]
best_thresh_scam = 0.70
best_f1_val = 0.0

for t in np.arange(0.50, 0.86, 0.01):
    preds_v = (probs_val >= t).astype(int)
    if preds_v.sum() == 0: continue
    pr = precision_score(yva, preds_v, zero_division=0)
    rc = recall_score(yva, preds_v, zero_division=0)
    f1_val = f1_score(yva, preds_v, zero_division=0)
    # Require validation precision >= 98% while optimizing F1
    if pr >= 0.98 and f1_val > best_f1_val:
        best_f1_val = f1_val
        best_thresh_scam = round(float(t), 2)

best_thresh_susp = 0.35
print(f"Optimal thresholds (selected on validation set): SCAM >= {best_thresh_scam:.2f}, SUSPICIOUS >= {best_thresh_susp:.2f}")

# --- Test Set Evaluation ---
probs_te = lr.predict_proba(Xte)[:, 1]
preds_te = (probs_te >= best_thresh_scam).astype(int)

acc  = accuracy_score(yte, preds_te)
prec = precision_score(yte, preds_te, zero_division=0)
rec  = recall_score(yte, preds_te, zero_division=0)
f1   = f1_score(yte, preds_te, zero_division=0)
unseen_caught = (lr.predict_proba(Xun)[:, 1] >= best_thresh_scam).mean()

print(f"\nTest Set Metrics (Logistic Regression):")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  Unseen Electricity Campaign Caught: {unseen_caught:.1%}")

# --- Decision Tree Comparison on Test Set ---
dt_preds = (dt.predict_proba(Xte)[:, 1] >= best_thresh_scam).astype(int)
dt_unseen = (dt.predict_proba(Xun)[:, 1] >= best_thresh_scam).mean()

# --- GroupKFold Cross-Validation (5 Folds by Sentence Template) ---
gkf = GroupKFold(n_splits=5)
cv_p, cv_r, cv_f = [], [], []
X_known = featurize(known.text, word_vec, char_vec)
y_known = known.label.values
groups  = known.template.values

for tr_idx, va_idx in gkf.split(X_known, y_known, groups):
    m_cv = LogisticRegression(max_iter=2000, C=3, class_weight="balanced", random_state=42)
    m_cv.fit(X_known[tr_idx], y_known[tr_idx])
    pp = m_cv.predict_proba(X_known[va_idx])[:, 1]
    pd_pred = (pp >= best_thresh_scam).astype(int)
    cv_p.append(precision_score(y_known[va_idx], pd_pred, zero_division=0))
    cv_r.append(recall_score(y_known[va_idx], pd_pred, zero_division=0))
    cv_f.append(f1_score(y_known[va_idx], pd_pred, zero_division=0))

cv_prec_mean, cv_prec_std = float(np.mean(cv_p)), float(np.std(cv_p))
cv_rec_mean,  cv_rec_std  = float(np.mean(cv_r)), float(np.std(cv_r))
cv_f1_mean,   cv_f1_std   = float(np.mean(cv_f)), float(np.std(cv_f))
print(f"\n5-Fold GroupKFold CV (grouped by template):")
print(f"  Precision: {cv_prec_mean:.3f} ± {cv_prec_std:.3f}")
print(f"  Recall:    {cv_rec_mean:.3f} ± {cv_rec_std:.3f}")
print(f"  F1 Score:  {cv_f1_mean:.3f} ± {cv_f1_std:.3f}")

# --- Ablation Table ---
Xtr_w  = word_vec.transform(ctrain)
Xtr_c  = char_vec.transform(ctrain)
Xtr_cu = cue_matrix(list(train.text))

Xte_w  = word_vec.transform([clean(t) for t in test.text])
Xte_c  = char_vec.transform([clean(t) for t in test.text])
Xte_cu = cue_matrix(list(test.text))

ablation_configs = [
    ("Word TF-IDF only", Xtr_w, Xte_w),
    ("Word + Char TF-IDF", hstack([Xtr_w, Xtr_c]).tocsr(), hstack([Xte_w, Xte_c]).tocsr()),
    ("Full Model (Word + Char + Cues)", hstack([Xtr_w, Xtr_c, Xtr_cu]).tocsr(), hstack([Xte_w, Xte_c, Xte_cu]).tocsr()),
]

ablation_rows = []
for abl_name, Xa_tr, Xa_te in ablation_configs:
    m = LogisticRegression(max_iter=2000, C=3, class_weight="balanced", random_state=42)
    m.fit(Xa_tr, ytr)
    pp = m.predict_proba(Xa_te)[:, 1]
    ap = (pp >= best_thresh_scam).astype(int)
    ablation_rows.append(dict(
        configuration=abl_name,
        accuracy=round(float(accuracy_score(yte, ap)), 4),
        precision=round(float(precision_score(yte, ap, zero_division=0)), 4),
        recall=round(float(recall_score(yte, ap, zero_division=0)), 4),
        f1=round(float(f1_score(yte, ap, zero_division=0)), 4),
    ))

abl_df = pd.DataFrame(ablation_rows)
print("\nAblation Results:\n", abl_df.to_string(index=False))

# --- Realistic Prevalence Evaluation (5% Scam Prevalence) ---
tp = int(((preds_te == 1) & (yte == 1)).sum())
fn = int(((preds_te == 0) & (yte == 1)).sum())
fp = int(((preds_te == 1) & (yte == 0)).sum())
tn = int(((preds_te == 0) & (yte == 0)).sum())

tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

# 1,000 incoming messages in a realistic inbox with 5% scam rate (50 scam, 950 safe)
N = 1000; prev = 0.05
n_scam_real, n_safe_real = int(N * prev), int(N * (1 - prev))
tp_real = tpr * n_scam_real
fp_real = fpr * n_safe_real
prec_real = tp_real / (tp_real + fp_real) if (tp_real + fp_real) > 0 else 0.0
rec_real  = tpr
fa_per_1k = round(fpr * 1000)

print(f"\nRealistic 5% Prevalence Evaluation:")
print(f"  Precision at 5% prevalence: {prec_real:.4f}")
print(f"  Recall at 5% prevalence:    {rec_real:.4f}")
print(f"  False alarms per 1,000 safe messages: {fa_per_1k}")

# --- Save Model Bundle ---
bundle = dict(
    model=lr, word_vec=word_vec, char_vec=char_vec,
    train_scam_texts=train_scam_texts,
    thresh_scam=best_thresh_scam, thresh_suspicious=best_thresh_susp,
    scam_discriminating_words=scam_discriminating,
)
joblib.dump(bundle, "scamsense_model.joblib")
print(f"Saved model bundle: scamsense_model.joblib ({os.path.getsize('scamsense_model.joblib')/1024:.0f} KB)")

# --- Save Comparison and Confusion Matrix ---
os.makedirs("outputs", exist_ok=True)
comparison_df = pd.DataFrame([
    dict(Model="Logistic Regression", Accuracy=round(acc, 3), Precision=round(prec, 3),
         Recall=round(rec, 3), F1=round(f1, 3), UnseenScamCaught=round(float(unseen_caught), 3)),
    dict(Model="Decision Tree", Accuracy=round(accuracy_score(yte, dt_preds), 3),
         Precision=round(precision_score(yte, dt_preds, zero_division=0), 3),
         Recall=round(recall_score(yte, dt_preds, zero_division=0), 3),
         F1=round(f1_score(yte, dt_preds, zero_division=0), 3),
         UnseenScamCaught=round(float(dt_unseen), 3)),
])
comparison_df.to_csv("outputs/model_comparison.csv", index=False)
abl_df.to_csv("outputs/ablation.csv", index=False)

ConfusionMatrixDisplay(confusion_matrix(yte, preds_te), display_labels=["Safe", "Scam"]).plot(cmap="Blues")
plt.title("ScamSense — Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# --- Evaluate on Independent realistic_test.csv (Task 1 & Task 3 Comparison) ---
real_test_path = "data/realistic_test.csv"
realistic_metrics = {}
if os.path.exists(real_test_path):
    df_real = pd.read_csv(real_test_path)
    real_results = []
    for _, row in df_real.iterrows():
        p_val, v_val, r_val = explain(row["text"], bundle)
        real_results.append(dict(
            text=row["text"], label=row["label"], kind=row["kind"],
            prob=round(p_val, 4), verdict=v_val,
            correct=int((1 if v_val in ("SCAM", "SUSPICIOUS") else 0) == row["label"]),
            strict_correct=int((1 if v_val == "SCAM" else 0) == row["label"])
        ))
    real_res_df = pd.DataFrame(real_results)
    real_res_df.to_csv("outputs/new_realistic.csv", index=False)
    
    strict_preds_real = [1 if r["verdict"] == "SCAM" else 0 for r in real_results]
    flagged_preds_real = [1 if r["verdict"] in ("SCAM", "SUSPICIOUS") else 0 for r in real_results]
    
    realistic_metrics = dict(
        strict=dict(
            accuracy=round(float(accuracy_score(df_real.label, strict_preds_real)), 4),
            precision=round(float(precision_score(df_real.label, strict_preds_real, zero_division=0)), 4),
            recall=round(float(recall_score(df_real.label, strict_preds_real, zero_division=0)), 4),
            f1=round(float(f1_score(df_real.label, strict_preds_real, zero_division=0)), 4),
        ),
        flagged=dict(
            accuracy=round(float(accuracy_score(df_real.label, flagged_preds_real)), 4),
            precision=round(float(precision_score(df_real.label, flagged_preds_real, zero_division=0)), 4),
            recall=round(float(recall_score(df_real.label, flagged_preds_real, zero_division=0)), 4),
            f1=round(float(f1_score(df_real.label, flagged_preds_real, zero_division=0)), 4),
        )
    )
    print(f"\nIndependent Realistic Test Set (n={len(df_real)}):")
    print(f"  Strict (SCAM=1):    Acc={realistic_metrics['strict']['accuracy']:.3f}, Rec={realistic_metrics['strict']['recall']:.3f}, Prec={realistic_metrics['strict']['precision']:.3f}")
    print(f"  Flagged (SCAM/SUSP): Acc={realistic_metrics['flagged']['accuracy']:.3f}, Rec={realistic_metrics['flagged']['recall']:.3f}, Prec={realistic_metrics['flagged']['precision']:.3f}")

# Read baseline realistic metrics if available
baseline_realistic_metrics = {}
if os.path.exists("outputs/baseline_realistic.csv"):
    b_df = pd.read_csv("outputs/baseline_realistic.csv")
    b_strict = [1 if v == "SCAM" else 0 for v in b_df["verdict"]]
    b_flagged = [1 if v in ("SCAM", "SUSPICIOUS") else 0 for v in b_df["verdict"]]
    baseline_realistic_metrics = dict(
        strict=dict(
            accuracy=round(float(accuracy_score(b_df.label, b_strict)), 4),
            precision=round(float(precision_score(b_df.label, b_strict, zero_division=0)), 4),
            recall=round(float(recall_score(b_df.label, b_strict, zero_division=0)), 4),
            f1=round(float(f1_score(b_df.label, b_strict, zero_division=0)), 4),
        ),
        flagged=dict(
            accuracy=round(float(accuracy_score(b_df.label, b_flagged)), 4),
            precision=round(float(precision_score(b_df.label, b_flagged, zero_division=0)), 4),
            recall=round(float(recall_score(b_df.label, b_flagged, zero_division=0)), 4),
            f1=round(float(f1_score(b_df.label, b_flagged, zero_division=0)), 4),
        )
    )

# --- Save Complete Results to outputs/results.json ---
results = dict(
    dataset=dict(
        total_rows=len(df),
        unique=len(known) + len(unseen_elec) + len(unseen_gas) + len(unseen_sim),
        train=len(train),
        val=len(val),
        test=len(test),
        unseen_electricity=len(unseen_elec),
        unseen_gas=len(unseen_gas),
        unseen_sim_block=len(unseen_sim),
    ),
    baseline_known=BASELINE,
    test_set=dict(
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        unseen_caught=round(float(unseen_caught), 4)
    ),
    thresholds=dict(
        scam=best_thresh_scam,
        suspicious=best_thresh_susp
    ),
    cv=dict(
        folds=5,
        precision_mean=round(cv_prec_mean, 4), precision_std=round(cv_prec_std, 4),
        recall_mean=round(cv_rec_mean, 4), recall_std=round(cv_rec_std, 4),
        f1_mean=round(cv_f1_mean, 4), f1_std=round(cv_f1_std, 4)
    ),
    ablation=ablation_rows,
    realistic_prevalence=dict(
        scam_pct=5,
        precision=round(prec_real, 4),
        recall=round(rec_real, 4),
        false_alarms_per_1k_safe=fa_per_1k
    ),
    decision_tree=dict(
        accuracy=round(float(accuracy_score(yte, dt_preds)), 4),
        precision=round(float(precision_score(yte, dt_preds, zero_division=0)), 4),
        recall=round(float(recall_score(yte, dt_preds, zero_division=0)), 4),
        f1=round(float(f1_score(yte, dt_preds, zero_division=0)), 4),
        unseen_caught=round(float(dt_unseen), 4)
    ),
    independent_realistic_test=dict(
        sample_size=len(df_real) if os.path.exists(real_test_path) else 0,
        baseline_score=baseline_realistic_metrics,
        new_score=realistic_metrics
    )
)

with open("outputs/results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSuccessfully saved outputs/results.json")
