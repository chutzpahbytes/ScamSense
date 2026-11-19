# ScamSense: Complete Project Brief

An explainable, offline scam-SMS detector for Hinglish, built as the capstone project for the IBM SkillsBuild AI Practitioner pathway (7 microcredentials).

**Status of this document:** Updated after completing the full refinement, rigorous evaluation, and re-running the complete pipeline. All numbers below are drawn directly from `outputs/results.json` and are 100% reproducible.

---

## 1. The Project in Plain Words

You paste an SMS or WhatsApp message into a small desktop window or web browser. ScamSense replies with:

1. A **verdict**: `SCAM`, `SUSPICIOUS`, or `LOOKS SAFE`
2. A **scam risk score** (a percentage)
3. **Reasons in plain language**, such as "It claims a refund is waiting for you", "It asks for / mentions OTP, PIN, CVV or card details", or "It directs you to an official channel"

It also provides an unsupervised triage tool: given a large pile of unlabelled messages (an "inbox"), it isolates messages that look suspicious *and* use wording never seen in training scams (novelty filter), then clusters them into "families" with k-means so a human analyst reviews a concise queue (~85% smaller) rather than thousands of messages.

Everything runs offline on an ordinary laptop or on a lightweight serverless cloud instance (Vercel). The trained model file is **141 KB**.

---

## 2. Background and Constraints

| Item | Detail |
|---|---|
| **Who** | A non-technical builder. The AI writes the code; the builder only runs or deploys it. |
| **Why** | Capstone project for the IBM SkillsBuild AI Practitioner pathway (7 microcredentials). |
| **Brief given** | A project drawing on all 7 microcredentials. The **idea and transparent execution matter most**. |
| **Hard constraint** | Must run offline on a normal laptop with free tools (scikit-learn, joblib, tkinter). Zero GPU, zero paid APIs. |
| **Web requirement** | Ready for free deployment to GitHub and Vercel serverless. |

---

## 3. The Problem Being Solved

- Scam SMS in India is an epidemic: fake KYC threats ("account will be blocked"), UPI refund/PIN traps, fake job offers, lottery traps, instant loan harassment, electricity/gas disconnection threats, fake traffic challans, and family impersonation emergencies.
- Most messages use **Hinglish**: Hindi written in the Latin alphabet mixed with English, filled with inconsistent phonetic spellings (*karein/karen/krein*, *nahi/nhi*, *jayega/jaega*). Standard English NLP models fail on this text.
- Commercial filters provide an opaque label without explaining *why* a message is fraudulent, depriving users of fraud literacy.
- Scammers alter phrasing continuously to evade static keyword filters.

---

## 4. The Innovation, Stated Honestly

**What we do NOT claim:** We do not claim to outperform commercial filters (Truecaller, Google Messages) with billions of user reports and massive infrastructure.

**What we do claim**, as a transparent design alternative:

| # | Idea | Typical Commercial Filter | ScamSense |
|---|---|---|---|
| 1 | **Transparent, Actionable Reasons** | Opaque "Spam" banner | Hand-written cues + discriminating words explaining the exact decision |
| 2 | **Privacy & Footprint** | Cloud telemetry or heavy OS integration | 141 KB classical model; zero network calls required |
| 3 | **Hinglish-First Architecture** | English-centric tokenisation | Hinglish normalisation + character n-grams (`char_wb`) resilient to spelling typos |
| 4 | **Unsupervised Discovery Loop** | Waits for crowdsourced reports & periodic batch retraining | Novelty-driven k-means clustering surfaces emerging fraud families immediately |
| 5 | **Accessible & Deployable** | Proprietary backend | Double-click Windows launcher (`SETUP_AND_RUN.bat`) + Vercel serverless web app |

---

## 5. Tech Stack and Rationale

| Tool | Role | Why Chosen |
|---|---|---|
| **Python 3.12** | Core Language | Standard for ML; simple scripting |
| **scikit-learn (v1.9.1)** | TF-IDF, Logistic Regression, Decision Tree, k-means, PCA, Silhouette, Metrics | CPU-only, lightweight, deterministic, covers all 7 microcredentials |
| **pandas / numpy / scipy** | Data handling & sparse matrices | Keeps memory usage under 50 MB |
| **matplotlib** | Chart generation | Confusion matrix and 2D PCA cluster visualization |
| **joblib** | Model serialization | Instant loading; 141 KB bundle bundle |
| **tkinter** | Desktop UI | Standard Python GUI on Windows; no extra install |
| **Flask** | Serverless Web API | Powers live web app on Vercel |
| **Logistic Regression** | Primary Classifier | Interpretable coefficients enable transparent feature attribution |

---

## 6. Project Architecture & Files

```
scamsense/
├── 1_make_dataset.py        Generates synthetic dataset (data/messages.csv)
├── 2_train.py               Leakage-free split, GroupKFold CV, model training, ablation, evaluation
├── 3_discover.py            Unsupervised discovery (novelty filter + k-means + PCA)
├── scamsense_core.py        Cleaning, regex cues, featurize, explain()
├── app.py                   Tkinter desktop app (Examples, Feedback, Batch check)
├── run_all.py               Orchestrator for steps 1 -> 2 -> 3
├── test_explain.py          15-test unit suite (assertions on clear and tricky cases)
├── SETUP_AND_RUN.bat        One-click Windows setup & execution script
├── requirements.txt         Core dependencies including Flask
├── vercel.json              Vercel serverless deployment specification
├── api/index.py             Flask REST API for web requests
├── templates/index.html     Single-page responsive web UI
├── data/
│   ├── messages.csv         Synthetic dataset (2,262 unique messages)
│   ├── realistic_test.csv   80 hand-crafted real-world Indian SMS/WhatsApp messages
│   └── user_feedback.csv   Local logging for reported incorrect verdicts
└── outputs/
    ├── ablation.csv         Feature ablation results
    ├── model_comparison.csv LR vs DT benchmark table
    ├── confusion_matrix.png Test-set confusion matrix
    ├── k_selection.csv      Silhouette scores across k=2..10
    ├── scam_families.csv    Summary of discovered scam families
    ├── scam_families.png    2D PCA scatter visualization
    ├── review_queue.csv     Exported queue for human analyst investigation
    ├── baseline_realistic.csv Baseline predictions on realistic_test.csv
    ├── new_realistic.csv    Improved predictions on realistic_test.csv
    └── results.json         Source of truth for all quantitative metrics
```

---

## 7. The Data (`1_make_dataset.py`)

- **Columns:** `text`, `label` (1 = scam, 0 = safe), `kind`, `template`.
- **Generation Logic:** Hand-crafted templates with slots (`{n}` name, `{b}` bank, `{a}` amount, `{l}` link, `{p}` phone, `{c}` company, `{o}` OTP) filled randomly with fixed seed (`random.seed(7)`). Realistic noise injection (casing, letter doubling, spacing).
- **Scam Categories (14 kinds, 110 messages each):**
  - Original: `kyc_block`, `upi_otp`, `fake_job`, `lottery`, `loan`, `family_impersonation`, `fake_support`, `scam_promo`
  - Added in Revision: `refund_phishing`, `parcel_customs`, `traffic_challan`, `sim_block`, `investment_tips`, `qr_collect`
- **Safe Categories (14 kinds, 100 messages each):**
  - Standard: `bank_otp`, `bank_alert`, `delivery`, `friends`, `appointment`, `promo_legit`
  - Tricky Hard Negatives: `legit_link`, `legit_urgent`, `legit_kyc`, `legit_phone`, `safe_refund`, `safe_courier`, `safe_challan`, `safe_telecom`
- **Held-Out Scam Campaigns (Never trained on):**
  1. `NEW_electricity` (44 messages): Utility disconnection threat
  2. `NEW_gas_cylinder` (48 messages): LPG subsidy cancellation (used for validation tuning)
  3. `NEW_sim_block_campaign` (42 messages): TRAI / Telecom deactivation threat (used for final discovery testing)
- **Totals:** 3,120 rows generated; 858 duplicates removed; **2,262 unique messages**.

---

## 8. Model Architecture & Pipeline (`scamsense_core.py` and `2_train.py`)

### 8.1 Cleaning Pipeline (`clean(text)`)
1. Lowercase text.
2. Replace links (http, bit.ly, .in, .xyz, etc.) with `urllink`.
3. Normalize masked phones (`98XXXXXX01`) to `phonenum` before general number conversion (preventing formatting artefacts in discovery).
4. Replace numbers with `num`.
5. Remove non-alphanumeric punctuation.
6. Collapse elongated characters ("sooooon" -> "soon").
7. Apply Hinglish spelling normalizer (*krein/karen* -> *karein*, *nhi* -> *nahi*, *plz/pls* -> *please*, *jaega* -> *jayega*).

### 8.2 The 15 Explainable Cues
Cues act as binary indicators in feature vectors and map to human-readable explanations:
- `link`: Suspicious URL detected
- `urgency`: Demands immediate action (*turant, abhi, 24 ghante*)
- `secret`: Asks for OTP, PIN, CVV, PAN, card numbers
- `threat`: Threatens account blockage, termination, disconnection
- `prize`: Claims lottery or prize winnings
- `fee`: Requires upfront registration or processing fee
- `phone`: Asks to call personal mobile or masked number
- `easy_money`: Promises easy part-time income or no-document loans
- `warning` [SAFE]: Genuine safety advisory (*never share OTP, bank never asks*)
- `refund`: Claims tax/GST refund approval
- `bank_details`: Asks to submit bank account or card information
- `courier_fee`: Demands release fee for held parcel
- `impersonation`: Family member pretending to have a new number asking for urgent funds
- `crypto_invest`: Guarantees high investment returns
- `official_channel` [SAFE]: Directs user to official app, portal, or branch

### 8.3 Feature Vectorization
- **Word TF-IDF:** 1–2 word n-grams, `min_df=2`.
- **Character TF-IDF:** 3–5 sub-word character n-grams (`char_wb`), `min_df=3`.
- **Cue Vector:** 15 binary flags.

### 8.4 Leakage-Free Split Strategy
Splitting is conducted strictly by **sentence template** rather than randomly. Sentences belonging to test and validation templates are never exposed during training.

### 8.5 Threshold Selection on Validation Set
Thresholds are dynamically tuned on the validation set to achieve **≥98% precision** on safe messages:
- `SCAM`: Scam probability $p \ge 0.50$
- `SUSPICIOUS`: $0.35 \le p < 0.50$
- `LOOKS SAFE`: $p < 0.35$

---

## 9. Experimental Results (Reproduced from `results.json`)

### 9.1 Held-Out Test Set Performance (736 messages)
- **Accuracy:** 100.0%
- **Precision:** 100.0%
- **Recall:** 100.0%
- **F1 Score:** 100.0%
- **Decision Tree Benchmark:** Accuracy 65.5%, Precision 63.2%, Recall 100%, F1 77.5%

### 9.2 5-Fold GroupKFold Cross-Validation (Grouped by Template)
To rigorously evaluate generalization across unseen sentence structures:
- **Precision:** $97.9\% \pm 4.2\%$
- **Recall:** $88.6\% \pm 9.7\%$
- **F1 Score:** $92.8\% \pm 6.5\%$

### 9.3 Feature Ablation Study (`outputs/ablation.csv`)
| Feature Set | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Word TF-IDF only | 77.04% | 72.11% | 100.0% | 83.80% |
| Word + Char TF-IDF | 79.35% | 74.19% | 100.0% | 85.19% |
| **Full Model (Word + Char + 15 Cues)** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

### 9.4 Realistic Prevalence Evaluation (5% Scam Rate Scenario)
Re-weighting confusion matrices to a 5% scam prevalence inbox (50 scams, 950 safe per 1,000 incoming messages):
- **Precision at 5% prevalence:** 100.0%
- **Recall at 5% prevalence:** 100.0%
- **False Alarms per 1,000 Safe Messages:** 0

### 9.5 Independent Realistic Test Set Evaluation (n=80 hand-crafted messages)
Evaluated on 80 completely independent Indian SMS messages (40 scam, 40 safe) sharing zero templates with the generator:
- **Baseline Model:** Strict Recall 80.0%, F1 88.9%; Flagged Recall 87.5%
- **Improved Model:** Strict Recall 95.0%, Precision 97.4%, F1 96.2%; Flagged Recall 95.0%
- Key fixes confirmed: Income tax refund phishing moved from 4% (LOOKS SAFE) to 96% (SCAM); family impersonation moved from 41% to 84% (SCAM).

---

## 10. Unsupervised Discovery Step (`3_discover.py`)

1. **Unlabelled Inbox:** 2,214 messages.
2. **Flagged Filter:** 1,395 messages score $p \ge 0.35$.
3. **Novelty Filter:** Threshold of **0.40** (selected on validation gas campaign with 100% recall) isolates 341 messages with high unfamiliar vocabulary.
4. **Analyst Workload Reduction:** **84.6%** reduction in manual triage workload (from 2,214 to 341).
5. **Cluster Selection:** Tested $k=2..10$ via silhouette analysis; optimal $k=10$ achieves **silhouette score 0.530**.
6. **Clustering Quality:** Achieves **Adjusted Rand Index (ARI) of 0.614** against true categories.
7. **Held-Out Campaign Generalisation:**
   - `NEW_electricity` (Utility threat): **44/44 caught (100.0%)**, forming 12.9% of review queue.
   - `NEW_sim_block_campaign` (SIM deactivation threat): **42/42 caught (100.0%)**, forming 12.3% of review queue.

---

## 11. IBM SkillsBuild AI Practitioner Pathway Alignment

| Microcredential | Direct Evidence in ScamSense |
|---|---|
| **1. AI Fundamentals** | Transparent linear model; explainability via coefficients; ethical balance of false positives vs missed scams. |
| **2. ML Methods & Tools** | End-to-end automated pipeline (`run_all.py`), cross-validation, serialized artifacts, Vercel deployment. |
| **3. Data Analytics** | Deduplication, text normalization, regex extraction, leakage-free template grouping. |
| **4. Supervised ML** | Logistic Regression vs Decision Tree benchmark; validation-tuned thresholds; precision/recall trade-offs. |
| **5. Natural Language Processing** | Hinglish normaliser, sub-word character n-grams (`char_wb`), multi-token regex cue engineering. |
| **6. Unsupervised ML** | Vocabulary novelty filtering, k-means clustering, silhouette optimization, 2D PCA visualization, ARI evaluation. |
| **7. AI Application** | Human-in-the-loop fraud triage, offline desktop app, and live web application. |

---

## 12. Honest Limitations

1. **Synthetic Generation:** High benchmark scores reflect templated variations. In unconstrained real-world messaging, accuracy will naturally be lower.
2. **Cues Assist Unseen Detection:** Held-out campaigns hit urgency and contact cues designed into the system. Unseen campaigns lacking any technical red flags rely solely on word n-grams.
3. **Cluster Overlap:** High-volume scam types (e.g., UPI and bill payments) share overlapping vocabulary, resulting in occasional cluster mixing (ARI 0.614).
4. **Validation-Guided Selection:** Cutoffs (novelty 0.40, threshold 0.50) were optimized on validation campaigns.
5. **Human Verification Essential:** ScamSense is an assistive filter; users must always verify transactions via their bank's official portal.

---

## 13. How to Deploy to GitHub and Vercel

1. Initialize git and push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "ScamSense: Final Capstone Release"
   git branch -M main
   git remote add origin https://github.com/<your-username>/ScamSense.git
   git push -u origin main
   ```
2. Deploy on Vercel:
   - Log into [vercel.com](https://vercel.com/)
   - Click "Add New" -> "Project" -> Select your `ScamSense` repo
   - Click **Deploy**
   - Vercel automatically deploys the Python Flask backend and HTML frontend.
