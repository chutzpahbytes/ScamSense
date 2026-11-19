"""
ScamSense unit tests — run with: python test_explain.py
No external framework needed. Loads the trained model and runs assert checks.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
from scamsense_core import explain

STOP = {"customer","update","number","rs","kal","abhi","dear","sir","please","hai","hain",
        "kar","karein","aapka","aapke","mera","mere","num","xx","urllink","it","is","a","the",
        "nahi","karo","kare","your","you","and","or","for","to","in","of"}

def has_stop_only_reason(reasons):
    for r in reasons:
        words = r.lower().replace("it ", "").split()
        if words and all(w.strip(".,!?\"'") in STOP for w in words):
            return True
    return False

bundle = joblib.load(os.path.join(os.path.dirname(__file__), "scamsense_model.joblib"))

tests = [
    # (message, expected_verdict_or_verdicts, description)
    ("742891 is your OTP for HDFC Bank. Valid 10 minutes. Do not share with anyone. -HDFC Bank", "LOOKS SAFE", "Real OTP"),
    ("Dear customer, aapka account band ho jayega. KYC update karein: sbi-kyc.in", "SCAM", "KYC scam"),
    ("Income tax refund approved. Submit bank details at incometax-refund.in", "SCAM", "Tax refund phishing"),
    ("Rs 5000 credited to your SBI account on 19-Nov. -SBI", "LOOKS SAFE", "Real bank credit alert"),
    ("Bhai kal milte hain? Kuch urgent baat karni thi.", "LOOKS SAFE", "Friend chat with 'urgent'"),
    ("Congratulations! Aapne Rs 50000 ka lucky draw jeeta. Processing fee Rs 499 bharein.", "SCAM", "Lottery scam"),
    ("Jio: aapki SIM aaj raat block ho jayegi. KYC update karein: jio-kyc.in", "SCAM", "SIM block scam"),
    ("Your parcel is held at customs. Pay Rs 49 clearance fee: amzn-delivery-fee.in/release", "SCAM", "Parcel customs fee scam"),
    ("Hi, aapka Amazon order out for delivery hai. Expected by 6 PM. No payment at delivery.", "LOOKS SAFE", "Genuine delivery"),
    ("Instant loan Rs 50000 bina documents ke! Approved in 5 min. Apply now bit.ly/loan99", "SCAM", "Loan scam"),
    ("Earn Rs 5000 daily! Guaranteed returns. Join Telegram group now: t.me/invest99", "SCAM", "Investment scam"),
    ("Mummy phone kharab ho gaya, ye naya number hai. Aaj urgent paise chahiye Rs 2000.", ("SCAM", "SUSPICIOUS"), "Family impersonation"),
    ("HDFC: Rs 2000 debited from your a/c. If not done by you, call 1800-XXX-XXXX official helpline.", "LOOKS SAFE", "Real bank debit alert"),
    ("Traffic Police: pending challan Rs 500. Pay at parivahan.gov.in official portal only.", "LOOKS SAFE", "Genuine challan notice"),
    ("Work from home job! Earn Rs 15000/day. No experience. WhatsApp 9876543210", "SCAM", "Fake job"),
]

passed = 0
failed = 0
for msg, expected, desc in tests:
    try:
        p, verdict, reasons = explain(msg, bundle)
        assert reasons, f"FAIL [{desc}]: reasons list is empty"
        assert not has_stop_only_reason(reasons), f"FAIL [{desc}]: reason contains only stop words: {reasons}"
        expected_tuple = (expected,) if isinstance(expected, str) else tuple(expected)
        assert verdict in expected_tuple, f"FAIL [{desc}]: expected {expected}, got {verdict} ({p:.0%})\n  Reasons: {reasons}"
        print(f"  PASS [{desc}]: {verdict} ({p:.0%})")
        passed += 1
    except AssertionError as e:
        print(f"  {e}")
        failed += 1

print(f"\n{'All ' + str(passed) + ' tests passed!' if failed == 0 else str(failed) + ' test(s) FAILED, ' + str(passed) + ' passed.'}")
if failed > 0:
    sys.exit(1)
