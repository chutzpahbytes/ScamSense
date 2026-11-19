import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
"""
Step 1: Make a synthetic Hinglish SMS dataset.
(Replace data/messages.csv with a real SMS dataset later if you get one - the rest of the project still works.)
Columns: text, label (1 = scam, 0 = safe), kind (what type of message it is)
"""
import random, csv, os
random.seed(7)

names   = ["Rahul","Priya","Amit","Neha","Vikram","Sunita","Ravi","Pooja","Karan","Anjali"]
banks   = ["SBI","HDFC","ICICI","PNB","Axis Bank","Bank of Baroda"]
amounts = ["2,000","5,000","9,999","15,000","25,000","49,500","1,00,000"]
links   = ["bit.ly/kyc-upd8","tinyurl.com/b4nk-help","sbi-kyc-update.in/login","http://rewards-claim.xyz/win","cutt.ly/jobpay1","paytm-refund.co/claim"]
phones  = ["98XXXXXX01","70XXXXXX55","81XXXXXX22","99XXXXXX10"]
cos     = ["Amazon","Flipkart","Meesho","Google","TCS","Infosys"]

def pick(x): return random.choice(x)
def noise(t):
    r = random.random()
    if r < 0.15: t = t.upper()
    elif r < 0.30: t = t.lower()
    if random.random() < 0.2: t = t.replace("a","aa",1)
    if random.random() < 0.2: t = t.replace(" ","  ",1)
    return t

SCAM = {
 "kyc_block": [
  "Dear customer, your {b} account will be BLOCKED today. Update KYC immediately: {l}",
  "{b} alert: aapka KYC expire ho gaya hai. Abhi update karein warna account band ho jayega {l}",
  "Urgent! Your {b} net banking is suspended. Verify PAN now {l}",
  "Dear user aapka account 24 ghante me block hoga. KYC update karne ke liye click karein {l}",
 ],
 "upi_otp": [
  "Sir aapka {a} ka refund aa gaya hai. UPI PIN daalkar receive karein {l}",
  "Paytm: You received Rs {a}. Share OTP with our agent to credit the amount. Call {p}",
  "Aapke account se Rs {a} debit hone wala hai. Cancel karne ke liye apna OTP bhejein {p}",
  "{b}: Verify your card number and CVV to avoid charges of Rs {a} {l}",
 ],
 "fake_job": [
  "Work from home job! Roz Rs {a} kamaye. Sirf WhatsApp message karein {p}",
  "Hi {n}, {c} hiring part time. Earn Rs {a} per day, no experience needed. Join {l}",
  "Congratulations! Aapko {c} me data entry job mil gayi hai. Registration fees Rs 500 pay karein {l}",
  "Part time job offer: like YouTube videos aur kamaye Rs {a} daily. Contact {p}",
 ],
 "lottery": [
  "Congratulations {n}! Aapne Rs {a} ka lucky draw jeeta hai. Claim karne ke liye {l}",
  "You have WON an iPhone from {c}! Claim your prize in 2 hours {l}",
  "Badhai ho! {c} lucky customer me aapka number select hua. Prize paane ke liye processing fee bharein {l}",
 ],
 "loan": [
  "Instant loan Rs {a} bina documents ke! Approved in 5 min. Apply now {l}",
  "Pre-approved loan of Rs {a} for you. No CIBIL check. Click {l}",
  "Aapka loan Rs {a} sanction ho gaya hai. Processing fee pay karke paisa paayein {l}",
 ],
}
# Held out of training: a "new campaign" the model has never seen. Used for the discovery demo.
NEW_CAMPAIGN = [
  "Bijli bill pending hai. Aaj raat 9:30 tak payment nahi hua to connection kat jayega. Officer se baat karein {p}",
  "Electricity department: aapka bill Rs {a} baki hai. Abhi call karein {p} warna line disconnect hogi",
  "Dear consumer, bijli connection disconnect hone wala hai tonight. Contact officer {p} for immediate update",
  "Bill overdue Rs {a}. Power supply aaj raat band hogi. Helpline {p}",
]

# Held-out campaigns — NOT included in train/test, used only for generalisation tests
NEW_GAS_CYLINDER = [
  "LPG connection: aapka gas cylinder booking cancel ho sakta hai. Subsidiy bachane ke liye abhi verify karein: {l}",
  "HP Gas: aapki LPG subsidy band hone wali hai. KYC link karne ke liye click karein {l}",
  "Pradhan Mantri Ujjwala Yojana: aapka cylinder allocation rok diya gaya hai. Update karein {l}",
  "Gas agency: cylinder delivery ke liye Rs {a} advance pay karein {l} warna booking cancel hogi",
]
NEW_SIM_BLOCK_CAMPAIGN = [
  "TRAI: aapka mobile number 24 ghante me band ho jayega due to illegal activity. Call officer: {p}",
  "Department of Telecommunications: your number will be disconnected. Last chance to verify: {p}",
  "Legal notice: aapke number se fraudulent activities detect ki gayi hain. Avoid disconnection: {p}",
  "TRAI officer: your SIM linked to 2 criminal cases. Avoid arrest, call now: {p}",
]

SAFE = {
 "bank_otp": [
  "{o} is your OTP for {b} transaction. Do not share it with anyone. -{b}",
  "Your OTP is {o}. Valid for 10 minutes. Never share OTP, bank never asks for it.",
  "{o} aapka {b} login OTP hai. Ise kisi ke saath share na karein.",
 ],
 "bank_alert": [
  "Rs {a} debited from your {b} a/c on 12-Nov. If not you, call 1800-XXX-XXXX (official helpline).",
  "{b}: Rs {a} credited to your account. Avl bal updated. Thank you for banking with us.",
  "Aapke {b} account me Rs {a} salary credit hui hai.",
 ],
 "delivery": [
  "Your {c} order has been shipped and will arrive by Friday. Track in the app.",
  "Aapka parcel aaj deliver hoga. Delivery partner: {n}. OTP delivery ke time dein.",
  "{c}: Your package is out for delivery today.",
 ],
 "friends": [
  "Bhai kal match dekhne chalein? Shaam ko 7 baje milte hain",
  "Mummy ne bola ki aaj dinner pe aa jao, paneer bana hai",
  "Hey {n}, meeting 4 baje shift ho gayi hai. Notes bhej dena please",
  "Happy birthday {n}! Party kab de raha hai?",
  "Assignment ka deadline kal hai, tune submit kiya kya?",
  "Kal college nahi aunga, tabiyat thodi kharab hai",
 ],
 "appointment": [
  "Reminder: your appointment with Dr. Sharma is tomorrow at 11 AM.",
  "Aapka electricity bill Rs {a} generate ho gaya hai. Due date 25th. Pay via official app or counter.",
  "Your recharge of Rs 299 is successful. Validity 28 days.",
 ],
 "promo_legit": [
  "{c} Sale starts tonight! Up to 60% off on electronics. T&C apply.",
  "Get 10% cashback on your next recharge. Offer valid till Sunday. Open the app to know more.",
  "Aaj ka offer: 2 pizzas pe 1 free, sirf aaj ke liye. Order in the app.",
 ],
}

# ---- Harder cases: subtle scams and legit messages that LOOK suspicious ----
SCAM.update({
 "family_impersonation": [
  "Hi, mera naya number hai save kar lo. Ek urgent kaam hai, Rs {a} paytm kar do, kal wapas kar dunga",
  "Mummy phone toot gaya, ye mera new number hai. Rs {a} bhej do abhi zaroori hai",
 ],
 "fake_support": [
  "Sir aapke account me kuch problem hai, is number pe sampark karein {p}",
  "Hello, this is {b} customer care. Card update pending hai, please call back {p}",
  "{b}: aapka reward points Rs {a} expire ho rahe hain. Redeem karein {l}",
 ],
 "scam_promo": [
  "{c} free gift! Sirf Rs 99 delivery charge pay karke claim karein {l}",
  "Exclusive offer from {c}: gift card Rs {a} for you. Verify details {l}",
 ],
})
SAFE.update({
 "legit_link": [
  "Track your {c} order here: {c}.in/track/84721. Delivery by Friday.",
  "Your {b} statement is ready. View it in the official app or at {b}.co.in",
  "Join the college placement webinar tomorrow: meet.google.com/abc-defg-hij",
 ],
 "legit_urgent": [
  "Your {b} credit card payment of Rs {a} is due today. Pay via the official app to avoid late fee.",
  "Reminder: electricity bill due tomorrow. Pay at the counter or official app.",
  "Aaj raat 11 baje tak form submit kar dena, last date hai. -Admin office",
 ],
 "legit_kyc": [
  "{b}: To complete KYC please visit your nearest branch with original ID proof.",
  "Aapka KYC verify ho gaya hai. Koi action ki zarurat nahi. -{b}",
 ],
 "legit_phone": [
  "Delivery partner {n} will call you shortly from {p} for your {c} parcel.",
  "Hi, this is {n} from {c} HR. Your interview is scheduled Monday 11 AM. Reply YES to confirm.",
  "Your EMI of Rs {a} is due on 5th. Auto-debit will happen from your {b} account.",
 ],
})

# ---- New scam template dicts ----
SCAM.update({
 "refund_phishing": [
  "Income tax refund of Rs {a} approved. Submit your bank details at incometax-refund.in to receive.",
  "GST refund Rs {a} credited pending. Provide your account number and IFSC to {l}",
  "Dear {n}, your bank refund of Rs {a} is on hold. Update details now: {l}",
 ],
 "parcel_customs": [
  "Your parcel is held at customs. Pay Rs 49 clearance fee: {l}",
  "Delivery attempt failed. Pay Rs {a} redelivery fee to reschedule: {l}",
  "{c}: package held due to customs duty Rs {a}. Pay to release: {l}",
 ],
 "traffic_challan": [
  "Traffic Police: your vehicle has a pending challan of Rs {a}. Pay now to avoid court case: {l}",
  "E-challan notice: Rs {a} due for traffic violation. Last date today. Click {l}",
  "Parivahan: unpaid challan Rs {a} on your vehicle. Pay immediately {l}",
 ],
 "sim_block": [
  "Jio: aapki SIM aaj raat block ho jayegi. KYC update karein: {l}",
  "Airtel: your SIM will be deactivated in 24 hours. Update verification: {l}",
  "Vi: SIM deactivation alert. Click to prevent: {l} Urgent",
 ],
 "investment_tips": [
  "Earn Rs {a} daily! Guaranteed returns. Join our Telegram group: {l}",
  "Crypto investment: 300% guaranteed profit in 7 days. Invest Rs {a}. WhatsApp {p}",
  "Stock tips that guarantee Rs {a}/month. Limited seats. Join now: {l}",
 ],
 "qr_collect": [
  "Scan this QR to receive Rs {a} refund from {b}",
  "Dear {n}, aapka {a} receive karne ke liye ye QR scan karein aur amount enter karein",
  "UPI collect request of Rs {a} from agent. Scan to approve and receive your cashback",
 ],
})

# ---- New safe template dicts ----
SAFE.update({
 "safe_refund": [
  "Rs {a} refunded to your {b} account for order #84721. Amount will reflect in 3-5 business days. No action needed.",
  "Refund of Rs {a} processed for your {c} return. It will be credited to your original payment method. -Customer Care",
  "Aapke order ka refund Rs {a} aapke account me aa gaya hai. Check your {b} account.",
 ],
 "safe_courier": [
  "{c}: your order #48291 is out for delivery. Expected by 6 PM. No payment required at delivery.",
  "Track your {c} parcel at {c}.in/track/8472 — arriving Friday. Delivery partner will call you.",
  "Aapka parcel {c} ki taraf se aaj deliver hoga. OTP delivery ke waqt share karein.",
 ],
 "safe_challan": [
  "E-challan for vehicle MH01AB1234: Rs {a}. Pay ONLY at parivahan.gov.in or official counter. Do not pay to any agent.",
  "Traffic challan notice. Amount Rs {a}. Pay at official portal only. No third-party payment accepted.",
  "Challan Rs {a} due. Visit nearest RTO or parivahan.gov.in to pay. Valid till 30-Nov.",
 ],
 "safe_telecom": [
  "Jio: your recharge of Rs 299 is successful. Validity 28 days. No action needed.",
  "Airtel: plan renewed. 1.5GB/day, unlimited calls for 28 days active from today.",
  "Vi: your current plan is active till 15-Dec. Balance Rs {a}. Recharge at Vi app or store.",
 ],
})

def fill(t):
    return t.format(n=pick(names), b=pick(banks), a=pick(amounts), l=pick(links),
                    p=pick(phones), c=pick(cos), o=random.randint(100000,999999))

rows = []
def gen(kind, temps, label, n):
    for _ in range(n):
        i = random.randrange(len(temps))
        rows.append((noise(fill(temps[i])), label, kind, f"{kind}-{i}"))
for kind, temps in SCAM.items(): gen(kind, temps, 1, 110)
for kind, temps in SAFE.items(): gen(kind, temps, 0, 100)
gen("NEW_electricity", NEW_CAMPAIGN, 1, 60)
gen("NEW_gas_cylinder", NEW_GAS_CYLINDER, 1, 60)
gen("NEW_sim_block_campaign", NEW_SIM_BLOCK_CAMPAIGN, 1, 60)
random.shuffle(rows)

os.makedirs("data", exist_ok=True)
with open("data/messages.csv","w",newline="",encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["text","label","kind","template"]); w.writerows(rows)
print(f"Saved {len(rows)} messages to data/messages.csv")
