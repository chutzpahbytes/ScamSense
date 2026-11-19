"""
ScamSense app - paste an SMS, get a verdict and plain-language reasons. Works fully offline.
Run:  python app.py        (opens a window; if no window is available it falls back to a text prompt)
"""
import joblib, os, sys
from scamsense_core import explain

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    bundle = joblib.load(os.path.join(HERE, "scamsense_model.joblib"))
except FileNotFoundError:
    # Show friendly message before anything else
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("ScamSense", "Model file not found!\n\nPlease run SETUP_AND_RUN.bat first to build the model, then try again.")
        root.destroy()
    except Exception:
        print("ERROR: scamsense_model.joblib not found. Run SETUP_AND_RUN.bat first.")
    import sys; sys.exit(1)

def report(text):
    p, verdict, reasons = explain(text, bundle)
    return p, verdict, reasons

def run_console():
    print("ScamSense (offline). Paste a message and press Enter. Type 'q' to quit.\n")
    while True:
        msg = input("Message> ").strip()
        if msg.lower() in ("q", "quit", "exit"): break
        if not msg: continue
        p, v, r = report(msg)
        print(f"\n  Verdict: {v}   (scam score {p:.0%})")
        for line in r: print("   -", line)
        print()

def run_window():
    import tkinter as tk
    from tkinter import ttk, messagebox
    root = tk.Tk(); root.title("ScamSense - offline scam SMS checker"); root.geometry("780x620")
    COL = {"SCAM": "#c62828", "SUSPICIOUS": "#ef6c00", "LOOKS SAFE": "#2e7d32"}
    ttk.Label(root, text="Paste the SMS / WhatsApp message here:", font=("Segoe UI", 11)).pack(anchor="w", padx=14, pady=(14, 4))
    box = tk.Text(root, height=7, wrap="word", font=("Segoe UI", 11)); box.pack(fill="x", padx=14)

    # Examples section
    examples_frame = ttk.LabelFrame(root, text="Quick Examples (click to load)", padding=4)
    examples_frame.pack(fill="x", padx=14, pady=(4,0))
    EXAMPLES = [
        ("🔴 KYC Scam", "Dear customer, aapka SBI account aaj band ho jayega. KYC update karein: sbi-kyc-update.in/login Urgent!"),
        ("🔴 Tax Refund Scam", "Income tax refund of Rs 15,490 approved. Submit your bank details at incometax-refund.in"),
        ("🔴 Lottery Scam", "Congratulations! Aapne Rs 50,000 lucky draw jeeta. Claim fee Rs 499 pay karein. Call 9876543210"),
        ("🟢 Real OTP", "742891 is your OTP for HDFC Bank. Valid 10 min. Never share OTP. -HDFC"),
        ("🟢 Real Alert", "Rs 5,000 credited to your SBI account on 19-Nov. Available balance updated. -SBI"),
        ("🟢 Friend Chat", "Bhai kal match dekhne chalein? Shaam 7 baje milte hain."),
        ("⚠️ Family Money Ask", "Mummy phone kharab ho gaya, ye naya number hai. Aaj urgent paise chahiye Rs 2000, kal wapas karunga."),
        ("⚠️ Parcel Fee", "Your parcel is held at customs. Pay Rs 49 clearance fee: amzn-delivery-fee.in/release"),
    ]
    btn_frame = ttk.Frame(examples_frame)
    btn_frame.pack(fill="x")
    for label, text in EXAMPLES:
        def make_loader(t=text):
            def load():
                box.delete("1.0", "end")
                box.insert("1.0", t)
            return load
        ttk.Button(btn_frame, text=label, command=make_loader()).pack(side="left", padx=2, pady=2)

    verdict = tk.Label(root, text="", font=("Segoe UI", 22, "bold")); verdict.pack(pady=(14, 0))
    score = tk.Label(root, text="", font=("Segoe UI", 11)); score.pack()
    why = tk.Label(root, text="", font=("Segoe UI", 11), justify="left", wraplength=700); why.pack(anchor="w", padx=18, pady=10)

    # last_result dict — shared between check() and report_wrong()
    last_result = {"text": "", "verdict": ""}

    def check():
        t = box.get("1.0", "end").strip()
        if not t: return
        p, v, r = report(t)
        last_result["text"] = t
        last_result["verdict"] = v
        verdict.config(text=v, fg=COL[v])
        score.config(text=f"Scam score: {p:.0%}")
        why.config(text="Why:\n" + "\n".join("  \u2022 " + x for x in r))

    def clear():
        box.delete("1.0", "end"); verdict.config(text=""); score.config(text=""); why.config(text="")

    def report_wrong():
        if not last_result["text"]:
            messagebox.showinfo("ScamSense", "Check a message first.")
            return
        win = tk.Toplevel(root)
        win.title("Report Wrong Verdict")
        win.geometry("300x120")
        tk.Label(win, text=f"Model said: {last_result['verdict']}\nWhat was it actually?", font=("Segoe UI", 11)).pack(pady=10)
        btn_row = ttk.Frame(win); btn_row.pack()
        def save(user_v):
            import csv, datetime
            os.makedirs("data", exist_ok=True)
            with open("data/user_feedback.csv", "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if os.path.getsize("data/user_feedback.csv") == 0:
                    w.writerow(["timestamp","text","model_verdict","user_verdict"])
                w.writerow([datetime.datetime.now().isoformat(), last_result["text"], last_result["verdict"], user_v])
            win.destroy()
            messagebox.showinfo("ScamSense", "Saved to data/user_feedback.csv. Thank you!")
        ttk.Button(btn_row, text="It was SCAM", command=lambda: save("SCAM")).pack(side="left", padx=4)
        ttk.Button(btn_row, text="It was SAFE", command=lambda: save("LOOKS SAFE")).pack(side="left", padx=4)

    def check_many():
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("Text/CSV", "*.txt *.csv"), ("All", "*.*")])
        if not path: return
        import csv as csv_mod
        messages = []
        if path.endswith(".csv"):
            with open(path, encoding="utf-8") as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    col = "text" if "text" in row else list(row.keys())[0]
                    messages.append(row[col])
        else:
            with open(path, encoding="utf-8") as f:
                messages = [l.strip() for l in f if l.strip()]
        if not messages:
            messagebox.showinfo("ScamSense", "No messages found in file.")
            return
        results = []
        counts = {"SCAM": 0, "SUSPICIOUS": 0, "LOOKS SAFE": 0}
        for m in messages:
            p, v, r = report(m)
            results.append({"text": m, "verdict": v, "score_pct": f"{p:.0%}", "reasons": " | ".join(r)})
            counts[v] = counts.get(v, 0) + 1
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/batch_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv_mod.DictWriter(f, fieldnames=["text","verdict","score_pct","reasons"]); w.writeheader(); w.writerows(results)
        messagebox.showinfo("ScamSense", f"Checked {len(messages)} messages:\n{counts.get('SCAM',0)} SCAM, {counts.get('SUSPICIOUS',0)} SUSPICIOUS, {counts.get('LOOKS SAFE',0)} LOOKS SAFE\n\nResults saved to outputs/batch_results.csv")

    bar = ttk.Frame(root); bar.pack(pady=4)
    ttk.Button(bar, text="Check message", command=check).pack(side="left", padx=6)
    ttk.Button(bar, text="Clear", command=clear).pack(side="left", padx=6)
    ttk.Button(bar, text="Report Wrong", command=report_wrong).pack(side="left", padx=6)
    ttk.Button(bar, text="Check Many...", command=check_many).pack(side="left", padx=6)
    tk.Label(root, text="Runs fully offline. Nothing you paste leaves this computer.\nA helper, not a guarantee: never share OTP/PIN, and verify with your bank's official app.",
             font=("Segoe UI", 9), fg="#555").pack(side="bottom", pady=10)
    root.mainloop()

if __name__ == "__main__":
    if "--console" in sys.argv: run_console()
    else:
        try: run_window()
        except ImportError: run_console()
