"""
ScamSense core: cleaning, cue detection, model features, explanations.
Everything here runs offline using scikit-learn only.
"""
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

# ---------- 1. Text cleaning (Microcredentials 3 and 5) ----------
# Small Hinglish normaliser: collapses common spelling variants to one form.
HINGLISH = {
    "krein": "karein", "karen": "karein", "kre": "kare", "krna": "karna",
    "nhi": "nahi", "nai": "nahi", "plz": "please", "pls": "please",
    "hogya": "ho gaya", "hoga": "hoga", "jaega": "jayega", "jayegi": "jayega",
    "kro": "karo", "bhjein": "bhejein"
}
LINK = re.compile(r"(https?://\S+|\b\w[\w\-]*\.(?:in|xyz|co|com|ly|me|org|net)\b\S*)", re.I)

def clean(text):
    t = text.lower()
    t = LINK.sub(" urllink ", t)
    # Masked phones (e.g. 98XXXXXX01, 98xxxx01) -> 'phonenum' BEFORE numbers get mapped to 'num'
    t = re.sub(r"\b\d{2,}[xX]{2,}\d{2,}\b", " phonenum ", t)
    t = re.sub(r"\d[\d,\.]*", " num ", t)            # numbers -> 'num'
    t = re.sub(r"[^a-z\s]", " ", t)                   # drop punctuation
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)              # sooooon -> soon
    words = [HINGLISH.get(w, w) for w in t.split()]   # Hinglish normalisation
    return " ".join(words)

# ---------- 2. Explainable cues (plain-language reasons) ----------
CUES = {   # name: (regex, plain-language reason)
 "link":             (r"http|bit\.ly|tinyurl|cutt\.ly|\.xyz|\.in/|\.co/|\.org/|\.net/", "contains a suspicious link"),
 "urgency":          (r"urgent|immediately|abhi|turant|today|aaj|tonight|24 ghante|within|hours|warna|jaldi|now\b", "pushes you to act urgently"),
 "secret":           (r"\botp\b|\bpin\b|\bcvv\b|card number|\bpan\b", "asks for / mentions OTP, PIN, CVV or card details"),
 "threat":           (r"block|suspend|band ho|expire|terminate|cancel|disconnect|deactivat", "threatens to block or cut a service"),
 "prize":            (r"congratulation|badhai|\bwon\b|jeeta|lucky|prize", "claims you won a prize"),
 "fee":              (r"processing fee|registration fee|registration fees|fees? rs|fee pay|fee bhar|clearance fee", "asks you to pay a fee upfront"),
 "phone":            (r"\b\d{2}[Xx]{2,}\d{2,}\b|\b[6-9]\d{9}\b|\b[6-9]\d{2}[-\s]\d{3}[-\s]\d{4}\b|\+91[-\s]?[6-9]\d{9}\b", "asks you to call a personal number"),
 "easy_money":       (r"work from home|part time|per day|daily|roz rs|no experience|no cibil|bina document", "promises easy money / jobs / loans"),
 "warning":          (r"do not share|never share|na karein|official helpline|t&c apply|bank never asks", "contains a genuine safety warning"),
 "refund":           (r"refund.{0,25}(approved|processed|pending|credited|claim)|income.{0,10}tax.{0,15}refund|gst.{0,10}refund|itr.{0,10}refund", "claims a refund is waiting for you"),
 "bank_details":     (r"(submit|share|enter|update|provide|send|daalein).{0,35}(bank details|account number|card details|card number|ifsc|bank account)", "asks you to submit or share bank or card details"),
 "courier_fee":      (r"(customs|held at|redelivery|release your parcel|clearance fee|delivery fee).{0,40}(pay|fee|charge|rs|rupee)", "asks you to pay a fee to release a parcel"),
 "impersonation":    (r"(naya number|new number|phone (toot|kharab|tut|gira)|save kar lo|mera number).{0,50}(bhej|paytm|gpay|transfer|paise|rupee|rs\b)", "appears to impersonate a family member or friend asking for money"),
 "crypto_invest":    (r"(guaranteed.{0,20}(return|profit|earn)|double your money|daily profit|weekly return|telegram group|whatsapp group|stock tips).{0,40}(invest|earn|profit|money|rs|lakh)", "promises guaranteed investment returns"),
 "official_channel": (r"official (app|website|portal|counter)|nearest branch|visit (your )?branch|parivahan\.gov\.in|paytm\.com|phonepe\.com|sbi\.co\.in|hdfcbank\.com|airtel\.in|jio\.com", "directs you to an official channel (a safe signal)"),
}
CUE_NAMES = list(CUES)
_CUE_RE = {k: re.compile(v[0], re.I) for k, v in CUES.items()}

def cue_vector(text):
    return np.array([1.0 if _CUE_RE[k].search(text) else 0.0 for k in CUE_NAMES])

def cue_matrix(texts):
    return csr_matrix(np.vstack([cue_vector(t) for t in texts]))

# ---------- 3. Model features ----------
def featurize(texts, word_vec, char_vec):
    cleaned = [clean(t) for t in texts]
    return hstack([word_vec.transform(cleaned), char_vec.transform(cleaned), cue_matrix(texts)]).tocsr()

# ---------- 4. Verdict + reasons ----------
# Generic stop words to never show in "Key words" explanation
GENERIC_STOP = {
    "customer", "update", "number", "rs", "kal", "abhi", "dear", "sir", "please",
    "hai", "hain", "kar", "karein", "aapka", "aapke", "mera", "mere", "num", "xx",
    "urllink", "phonenum", "it", "is", "a", "an", "the", "for", "your", "you", "in",
    "on", "and", "or", "this", "aur", "ye", "wo", "ek", "are", "be", "at", "by", "se",
    "to", "na", "do", "ki", "ke", "ka", "ko", "ho", "hi", "me", "karo", "kare", "nahi",
    "from", "with", "have", "will", "all", "so", "if", "not", "date", "call", "now"
}

def explain(text, bundle, top_words=3):
    """Returns (scam_probability, verdict, list_of_reasons)."""
    model = bundle["model"]
    wv, cv = bundle["word_vec"], bundle["char_vec"]
    thresh_scam = bundle.get("thresh_scam", 0.70)
    thresh_susp = bundle.get("thresh_suspicious", 0.35)
    scam_disc   = bundle.get("scam_discriminating_words", set())

    X = featurize([text], wv, cv)
    p = float(model.predict_proba(X)[0, 1])
    verdict = "SCAM" if p >= thresh_scam else ("SUSPICIOUS" if p >= thresh_susp else "LOOKS SAFE")

    coef  = model.coef_[0]
    n_w   = len(wv.vocabulary_)
    n_c   = len(cv.vocabulary_)
    contrib = X.multiply(coef).toarray()[0]

    reasons = []
    cue_c = contrib[n_w + n_c:]
    cv_flags = cue_vector(text)

    if verdict != "LOOKS SAFE":
        # 1. Cue reasons first (plain language)
        for i in np.argsort(-cue_c):
            if cue_c[i] > 0.05 and cv_flags[i] == 1:
                reasons.append("It " + CUES[CUE_NAMES[i]][1])

        # 2. Key discriminating words that drove score up
        inv = {v: k for k, v in wv.vocabulary_.items()}
        wc  = contrib[:n_w]
        words = []
        for i in np.argsort(-wc):
            if wc[i] <= 0.05 or len(words) >= top_words:
                break
            w = inv[i]
            # Check discriminating set or fallback filter
            words_in_w = w.split()
            if any(x in GENERIC_STOP for x in words_in_w):
                continue
            if scam_disc:
                if w in scam_disc and not any(w in x or x in w for x in words):
                    words.append(w)
            else:
                if not any(w in x or x in w for x in words):
                    words.append(w)
        if words:
            reasons.append("Key words: " + ", ".join(f'"{w}"' for w in words))

        # 3. Actionable advice for SUSPICIOUS
        if verdict == "SUSPICIOUS":
            reasons.append("Do not click or reply. Check with the sender through the official app or website.")

        if not reasons:
            reasons.append("Scam patterns detected in phrasing and structure")
    else:
        # LOOKS SAFE: explain why it looks safe
        safe_reasons = []
        for i in np.argsort(cue_c):
            if cue_c[i] < -0.05 and cv_flags[i] == 1:
                safe_reasons.append("It " + CUES[CUE_NAMES[i]][1])

        # Check positive safe cues (warning, official_channel)
        for cname in ["warning", "official_channel"]:
            if cname in CUE_NAMES:
                idx = CUE_NAMES.index(cname)
                if cv_flags[idx] == 1:
                    r_text = "It " + CUES[cname][1]
                    if r_text not in safe_reasons:
                        safe_reasons.append(r_text)

        if safe_reasons:
            reasons = safe_reasons[:2]
        else:
            reasons = ["No strong scam signals found; typical of regular messaging"]

    return p, verdict, reasons
