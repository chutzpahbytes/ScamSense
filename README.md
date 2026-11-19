# ScamSense — Explainable, Offline Hinglish Scam-SMS Detector

Built as the capstone project for the **IBM SkillsBuild AI Practitioner pathway** (7 microcredentials).

---

## 1. What ScamSense Does

1. **Instant Scam Checking:** Paste an SMS or WhatsApp message into the desktop app or web interface. It outputs:
   - A verdict: `SCAM`, `SUSPICIOUS`, or `LOOKS SAFE`
   - A scam risk score (percentage)
   - Actionable, plain-language reasons (e.g. *"It claims a refund is waiting for you"*, *"It asks for / mentions OTP, PIN, CVV or card details"*, *"It directs you to an official channel"*)
2. **Unsupervised Discovery:** Takes an unlabelled message inbox, filters for high-risk messages with **unfamiliar wording** (novelty filter), and clusters them into distinct families with k-means so an analyst reviews a short queue (~85% smaller) rather than thousands of messages.
3. **Completely Offline & Lightweight:** Classical ML pipeline powered by scikit-learn. Model bundle is **141 KB**. Zero GPU or cloud dependencies at run time.
4. **Web & Vercel Ready:** Includes a clean Flask API (`api/index.py`), mobile-friendly UI (`templates/index.html`), and `vercel.json` for 1-click free deployment.

---

## 2. Before vs After Performance Summary

All metrics are dynamically generated and verified from `outputs/results.json`:

| Metric | Baseline ("Before") | Improved ("After") | Notes / Improvement |
|---|---|---|---|
| **Unique Messages** | 1,385 | **2,262** | Added 6 new scam kinds, 4 hard-negative safe kinds |
| **Split Strategy** | Train / Test by template | **Train / Val / Test by template** | Validation set used for threshold selection |
| **Verdict Thresholds** | Fixed (0.70 / 0.35) | **Validation-Selected (0.50 / 0.35)** | Tuned for ≥98% precision on validation set |
| **Test Set Accuracy** | 93.9% | **100.0%** | Evaluated on held-out sentence templates |
| **Test Set Precision** | 100.0% | **100.0%** | Zero false alarms on test set safe messages |
| **Test Set Recall** | 89.7% | **100.0%** | All test set scam templates caught |
| **5-Fold GroupKFold CV** | Not evaluated | **Prec: 97.9% ± 4.2%<br>Rec: 88.6% ± 9.7%<br>F1: 92.8% ± 6.5%** | Folds grouped strictly by sentence template |
| **Realistic Prevalence (5%)** | Claimed imbalance | **Prec: 100%<br>False alarms: 0 / 1,000 safe** | Mathematically adjusted to 5% scam prevalence |
| **Realistic Test Set (n=80)** | Rec: 80.0% (strict)<br>F1: 88.9% | **Rec: 95.0% (strict)<br>F1: 96.2%** | Independent test with zero shared templates |
| **Income Tax Phishing Miss** | 4% (LOOKS SAFE) ❌ | **96% (SCAM) ✅** | Cues added for refund & bank detail traps |
| **Family Impersonation** | 41% (SUSPICIOUS) | **84% (SCAM) ✅** | Dedicated impersonation & urgency cues |
| **Discovery Queue Reduction** | 80.8% (1,385 → 266) | **84.6% (2,214 → 341)** | Human analyst reviews 341 msgs instead of 2,214 |
| **Discovery Silhouette (k)** | 0.49 (k=6) | **0.530 (k=10)** | k selected across range 2..10 via validation inbox |
| **Adjusted Rand Index (ARI)** | Not evaluated | **0.614** | Evaluates clustering against true scam categories |
| **Held-Out Campaigns Caught** | 100% (Electricity: 54/54) | **100% (Electricity: 44/44)<br>100% (SIM Block: 42/42)** | Verified on two distinct held-out scam campaigns |

---

## 3. How to Run

### Option A: Windows 1-Click (No Coding Required)
1. Double-click `SETUP_AND_RUN.bat`.
2. It installs requirements, executes `run_all.py` (data generation -> training -> discovery), and launches the desktop app.
3. For subsequent launches, open a terminal in this folder and run:
   ```cmd
   python app.py
   ```
   *(Or `python app.py --console` for terminal text mode)*.

### Option B: Free Web Deployment on Vercel
1. Push this folder to a GitHub repository.
2. Go to [Vercel](https://vercel.com/) -> Add New Project -> Import your repo.
3. Vercel automatically detects `vercel.json` and `api/index.py`. Click **Deploy**.
4. Test locally with:
   ```bash
   python api/index.py
   ```
   and visit `http://localhost:5000`.

### Option C: Manual CLI Execution
```bash
pip install -r requirements.txt
python run_all.py
python test_explain.py
python app.py
```

---

## 4. File Organization

```
scamsense/
├── 1_make_dataset.py        Step 1: Generates synthetic Hinglish SMS dataset
├── 2_train.py               Step 2: Template split, GroupKFold CV, train, ablation, save
├── 3_discover.py            Step 3: Unsupervised discovery (novelty filter + k-means + PCA)
├── scamsense_core.py        Shared logic: cleaning, cues, featurize, explain()
├── app.py                   Tkinter desktop app with Examples, Feedback, and Batch-Check
├── run_all.py               Runs steps 1, 2, and 3 sequentially
├── test_explain.py          Unit test suite (15 assertions covering clear and tricky cases)
├── SETUP_AND_RUN.bat        One-click Windows launcher
├── requirements.txt         Python dependencies (scikit-learn, pandas, numpy, scipy, matplotlib, joblib, flask)
├── vercel.json              Vercel serverless deployment config
├── api/
│   └── index.py             Flask REST API for web queries
├── templates/
│   └── index.html           Responsive web UI with live test chips
├── data/
│   ├── messages.csv         Generated synthetic training & testing dataset
│   ├── realistic_test.csv   80 hand-crafted real-world Indian SMS/WhatsApp messages
│   └── user_feedback.csv   Local feedback logged by users reporting wrong verdicts
└── outputs/
    ├── ablation.csv         Feature ablation study (Word vs Word+Char vs Full Model)
    ├── model_comparison.csv Logistic Regression vs Decision Tree benchmark
    ├── confusion_matrix.png Test-set confusion matrix chart
    ├── k_selection.csv      Silhouette score across cluster counts k=2..10
    ├── scam_families.csv    Summary of discovered scam families (size, purity, top words)
    ├── scam_families.png    2D PCA scatter plot of discovered families
    ├── review_queue.csv     Exported queue for human analyst investigation
    ├── baseline_realistic.csv Baseline model predictions on realistic_test.csv
    ├── new_realistic.csv    Improved model predictions on realistic_test.csv
    └── results.json         Source of truth for all quantitative metrics
```

---

## 5. Ablation Study

Each feature block was isolated and trained with the same classifier and split (`outputs/ablation.csv`):

| Configuration | Test Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Word TF-IDF only** | 77.04% | 72.11% | 100.0% | 83.80% |
| **Word + Char TF-IDF** | 79.35% | 74.19% | 100.0% | 85.19% |
| **Full Model (Word + Char + 15 Cues)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

- **Character n-grams (3–5 char_wb)** compensate for Hinglish spelling variations (e.g. *karein*, *karen*, *krein*).
- **Rule-based Cues** provide the final precision boost and directly power the plain-language explanations.

---

## 6. Mapping to IBM SkillsBuild AI Practitioner Pathway (7 Microcredentials)

| # | Microcredential | Implementation in ScamSense |
|---|---|---|
| **1** | **AI Fundamentals** | Interpretable model architecture (coefficients -> explanation); ethics of false alarms; regional dialect awareness. |
| **2** | **ML Methods & Tools** | End-to-end reproducible pipeline: synthetic generation, cleaning, GroupKFold CV, deployment (`SETUP_AND_RUN.bat`, Vercel). |
| **3** | **Data Analytics** | Duplicate pruning, regex cleaning, Hinglish normalisation, leakage-free template split. |
| **4** | **Supervised ML** | Logistic Regression vs Decision Tree comparison; validation-selected thresholding; precision/recall trade-off. |
| **5** | **Natural Language Processing** | Token normalization (`phonenum`, `urllink`, `num`), word TF-IDF (1–2 ngrams), character n-grams (`char_wb`), regex cues. |
| **6** | **Unsupervised ML** | Vocabulary-based novelty scoring, k-means clustering with validation-selected cutoff & k, PCA dimensionality reduction, ARI eval. |
| **7** | **AI Application** | Fraud mitigation with human-in-the-loop triage, offline privacy-first design, desktop & web user interfaces. |

---

## 7. How ScamSense Differs from State of the Art

| Dimension | Industry Filters (Truecaller, Google Messages) | ScamSense |
|---|---|---|
| **Explanations** | Black-box "Spam" label | Plain-language reasons: specific cues triggered + key discriminating words |
| **Privacy & Footprint** | Cloud processing or deep OS integration | 100% offline, 141 KB model, zero network traffic at runtime |
| **Hinglish Awareness** | Tuned predominantly on standard English | Hinglish normalisation dictionary + sub-word character n-grams |
| **New Scam Triage** | Waits for retraining on newly labelled crowdsourced reports | Unsupervised novelty filter groups emerging patterns for human review immediately |
| **Simplicity** | Proprietary backend infrastructure | Standard Python + scikit-learn, runs on any basic laptop |

---

## 8. Honest Limitations

1. **Synthetic Training Data:** While the pipeline structure is production-grade, the primary training set is generated via templates. Real-world SMS diversity is significantly broader.
2. **Hand-Crafted Cues:** Hand-written rules boost precision on known attack vectors (e.g. KYC, electricity, refunds) but can be circumvented by fraudsters altering their vocabulary.
3. **Cluster Mixing:** While unsupervised clustering achieves an Adjusted Rand Index of 0.614, some scam categories overlap (e.g., UPI traps and fake utility alerts both demand payment, occasionally clustering together).
4. **Validation-Guided Thresholds:** Thresholds (0.50 / 0.35) and novelty cutoffs (0.40) were selected using validation subsets. While this prevents test leakage, performance may vary on completely divergent corpora.
5. **Human-in-the-Loop Requirement:** ScamSense is designed as an advisory decision aid, not an infallible authority. Users should always verify suspicious messages through the official institution app or website.
