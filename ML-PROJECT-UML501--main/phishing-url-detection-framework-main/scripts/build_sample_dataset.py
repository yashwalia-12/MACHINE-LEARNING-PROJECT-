"""Build a high-quality sample dataset for quick testing and demonstration.

Generates `data/sample_dataset.csv` containing diverse legitimate and
phishing URLs with all 74 extracted features.
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.features.extractor import extract_all_features

SAMPLE_URLS = [
    # Legitimate (label = 0)
    ("https://www.google.com/search?q=machine+learning+classification", 0),
    ("https://github.com/torvalds/linux/commit/master", 0),
    ("https://en.wikipedia.org/wiki/Machine_learning", 0),
    ("https://www.amazon.com/dp/B08N5WRWNW/ref=dp_cerb_1", 0),
    ("https://stackoverflow.com/questions/12345/python-data-preprocessing", 0),
    ("https://www.microsoft.com/en-us/windows/get-windows-11", 0),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", 0),
    ("https://docs.python.org/3/library/urllib.parse.html", 0),
    ("https://news.ycombinator.com/item?id=30002100", 0),
    ("https://www.apple.com/iphone-16-pro/specs/", 0),
    ("https://medium.com/@data-science/phishing-detection-ml", 0),
    ("https://kaggle.com/datasets/sample-ml-project", 0),
    ("https://scikit-learn.org/stable/modules/preprocessing.html", 0),
    ("https://www.nytimes.com/section/technology", 0),
    ("https://www.bbc.com/news/technology-56789", 0),
    ("https://developer.mozilla.org/en-US/docs/Web/HTTP", 0),
    ("https://arxiv.org/abs/2103.00020", 0),
    ("https://www.netflix.com/browse/genre/83", 0),
    ("https://www.spotify.com/us/premium/", 0),
    ("https://aws.amazon.com/machine-learning/what-is-ml/", 0),
    ("https://cloud.google.com/vertex-ai/docs", 0),
    ("https://www.linkedin.com/in/computer-scientist/", 0),
    ("https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html", 0),
    ("https://www.reddit.com/r/MachineLearning/comments/xyz123/", 0),
    ("https://numpy.org/doc/stable/reference/arrays.html", 0),

    # Phishing (label = 1)
    ("http://paypal-security-account-update.xyz/login.php?cmd=verify", 1),
    ("http://192.168.1.105/bank-of-america/secure-login.html", 1),
    ("http://appleid-support-icloud-security.top/auth/verify?token=ab823", 1),
    ("http://wellsfargo-online-banking-security.cf/signon/update", 1),
    ("http://microsoft-outlook365-verify.casa/webmail/auth.php", 1),
    ("http://chase-bank-verify-account.buzz/customer/login.html", 1),
    ("http://amazon-prime-account-suspended.tk/order-update?id=9928", 1),
    ("http://netflix-billing-update-required.ml/signin/payment-check", 1),
    ("http://secure-login-facebook.account-verification.ga/login.php", 1),
    ("http://docusign-document-review-secured.surf/sign-now?doc=401", 1),
    ("http://google-drive-shared-document.vip/download?auth=912093", 1),
    ("http://bank-account-security-alert-chase.top/online/login", 1),
    ("http://metamask-wallet-seed-phrase-verify.click/extension", 1),
    ("http://binance-security-account-risk.work/login.html", 1),
    ("http://steam-community-free-nitro-gift.icu/trade/offer", 1),
    ("http://discord-gift-nitro-claim.xyz/verify-account", 1),
    ("http://usps-package-delivery-redelivery.rest/tracking-update", 1),
    ("http://dhl-express-shipment-exception-fee.fit/invoice/pay", 1),
    ("http://coinbase-auth-resolve-suspension.club/signin", 1),
    ("http://instagram-copyright-infringement-appeal.cam/verify", 1),
    ("http://adobe-pdf-shared-secured-portal.tk/view-document", 1),
    ("http://irs-tax-refund-status-claim.cf/portal/refund", 1),
    ("http://att-yahoo-mail-synchronization.ga/mail/login.php", 1),
    ("http://fidelity-investments-alert-verify.ml/login/auth", 1),
    ("http://citibank-cardholder-protection.top/security/update", 1),
]


def generate_dataset(output_path: str = "data/sample_dataset.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []

    print(f"[*] Extracting 74 features for {len(SAMPLE_URLS)} sample URLs...")
    for url, label in SAMPLE_URLS:
        feats = extract_all_features(url, enable_network=False)
        row = {"url": url, "label": label, **feats}
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[✓] Saved sample dataset to {output_path} with shape {df.shape}")
    return df


if __name__ == "__main__":
    generate_dataset()
