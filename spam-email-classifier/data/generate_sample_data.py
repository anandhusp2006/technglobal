"""
generate_sample_data.py
------------------------
Generates a small, labeled spam/ham dataset in the SAME schema as the
UCI SMS Spam Collection Dataset (https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset):

    label,text
    ham,"..."
    spam,"..."

WHY THIS EXISTS
This environment has no internet access to kaggle.com, so a synthetic dataset
is generated here purely so the pipeline is runnable end-to-end out of the box.

TO USE THE REAL DATASET (recommended for an actual submission):
  1. Download from either:
       https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
       https://www.kaggle.com/datasets/jayashree02/spam-ham-email-dataset
  2. Save it as data/spam_dataset.csv with columns: label,text
  3. Delete/skip this generator -- train.py will use spam_dataset.csv automatically
     if it already exists.
"""

import csv
import random

random.seed(42)

HAM_TEMPLATES = [
    "Hey, are we still meeting for lunch tomorrow?",
    "Can you send me the report before end of day?",
    "Happy birthday! Hope you have a great day.",
    "Don't forget to pick up milk on your way home.",
    "The meeting has been moved to 3pm, see you then.",
    "Thanks for your help earlier, really appreciate it.",
    "I'll call you once I land at the airport.",
    "Let's catch up this weekend, it's been a while.",
    "Please review the attached document and let me know your thoughts.",
    "Running a bit late, be there in 10 minutes.",
    "Can you pick up the kids from school today?",
    "Great job on the presentation, the client loved it.",
    "I left my charger at your place, can I grab it tomorrow?",
    "The project deadline is next Friday, let's plan accordingly.",
    "Mom says dinner is at 7, don't be late.",
    "I booked the tickets for the concert next month.",
    "Sorry I missed your call, what's up?",
    "Congrats on the new job, well deserved!",
    "Can you review my code before I push it?",
    "Let me know when you're free to talk.",
    "The doctor's appointment is confirmed for Monday 10am.",
    "I'll send over the invoice tomorrow morning.",
    "Are you coming to the team dinner tonight?",
    "Just checking in, how are you feeling today?",
    "The wifi password is on the fridge note.",
    "See you at the gym at 6?",
    "I updated the spreadsheet with the latest numbers.",
    "Can we reschedule our call to Thursday?",
    "Thanks for the birthday wishes everyone!",
    "The package should arrive by Wednesday.",
]

SPAM_TEMPLATES = [
    "WINNER!! You have been selected to receive a $1000 Walmart gift card. Click here to claim now!",
    "URGENT: Your account has been suspended. Verify your details immediately to avoid closure.",
    "Congratulations! You've won a free iPhone 15. Claim your prize before it expires!",
    "Get rich quick! Earn $5000 a week working from home, no experience needed. Sign up now!",
    "FREE entry into our $10,000 cash prize draw. Text WIN to 80088 now!",
    "Your loan of $50000 has been approved! Click the link to receive funds instantly.",
    "Hot singles in your area are waiting to chat with you! Click here now!",
    "Limited time offer: 90% off all products! Buy now before the deal ends!",
    "You have 1 new voicemail from your bank regarding suspicious activity. Call now.",
    "Claim your free vacation to the Bahamas now, only pay for taxes and fees!",
    "Act now! Your subscription will be cancelled unless you update payment info immediately.",
    "You've been pre-approved for a credit card with a $10,000 limit. Apply today!",
    "Lose 20 pounds in 2 weeks with this one weird trick! Order now!",
    "Your PayPal account has been locked. Click here to unlock it immediately.",
    "Congratulations, you are our lucky visitor today! Claim your free prize now!",
    "Make money fast with this exclusive investment opportunity, guaranteed returns!",
    "Final notice: Your car warranty is about to expire. Call now to renew.",
    "You have won a lottery of $1,000,000! Reply with your bank details to claim.",
    "Cheap meds online, no prescription needed! Order now and save 80%!",
    "URGENT: verify your Amazon account now or it will be permanently suspended.",
    "Text STOP to unsubscribe, or click here to claim your free gift card worth $500.",
    "You have been chosen for a special cash reward, click the link to claim it now.",
    "Increase your credit score instantly! Click here for a free consultation.",
    "Your package delivery failed. Click here to reschedule and pay a small fee.",
    "Exclusive deal just for you: refinance your mortgage and save thousands!",
    "Free trial! No credit card required, click now to start earning today!",
    "This is not a scam, you really won! Claim your reward before midnight.",
    "Your number has been selected for a free cruise vacation, call now!",
    "Double your bitcoin investment in 24 hours, guaranteed! Click to start.",
    "Act fast, only 3 spots left for our free webinar on making $10k/month!",
]


def augment(text, n_variants=6):
    """Create light variations so the dataset isn't just literal repeats."""
    variants = [text]
    suffixes = ["", " Reply now.", " Don't miss out.", " Limited time.", " Thanks.", " Regards."]
    prefixes = ["", "Hi, ", "Hello, ", "Note: ", "FYI - ", "Dear customer, "]
    while len(variants) < n_variants:
        v = random.choice(prefixes) + text + random.choice(suffixes)
        if v not in variants:
            variants.append(v)
    return variants


def build_dataset():
    rows = []
    for t in HAM_TEMPLATES:
        for v in augment(t, n_variants=5):
            rows.append(("ham", v))
    for t in SPAM_TEMPLATES:
        for v in augment(t, n_variants=5):
            rows.append(("spam", v))
    random.shuffle(rows)
    return rows


def main():
    rows = build_dataset()
    out_path = "data/spam_dataset_synthetic.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"  ham:  {sum(1 for r in rows if r[0] == 'ham')}")
    print(f"  spam: {sum(1 for r in rows if r[0] == 'spam')}")


if __name__ == "__main__":
    main()
