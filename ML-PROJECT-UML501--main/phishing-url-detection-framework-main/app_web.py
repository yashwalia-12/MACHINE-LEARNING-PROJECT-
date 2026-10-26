"""Standalone Zero-Dependency Web Dashboard for Phishing Detection.

Runs without needing Streamlit:
    python app_web.py
Opens automatically on http://localhost:8080
"""

import json
import os
import sys
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Add repo root to path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.train_and_predict import predict_url, train_models

# Ensure model exists
if not os.path.exists("artifacts/phishing_detector.joblib"):
    train_models()

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phishing URL Detector | UML501 Project</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-slate-50 min-h-screen text-slate-800">
    <!-- Navbar -->
    <header class="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
        <div class="max-w-6xl mx-auto px-4 py-3 flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <span class="text-3xl">🛡️</span>
                <div>
                    <h1 class="text-xl font-bold text-slate-900 leading-tight">Phishing URL Detection System</h1>
                    <p class="text-xs text-slate-500 font-medium">UML501 Machine Learning Project | 74 Features | XAI Explainability</p>
                </div>
            </div>
            <div class="flex items-center space-x-2">
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800">Model Active: Random Forest / Stacking</span>
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-blue-100 text-blue-800">98.85% Benchmark</span>
            </div>
        </div>
    </header>

    <main class="max-w-6xl mx-auto px-4 py-8 space-y-8">
        <!-- Main Scanner Box -->
        <section class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 md:p-8">
            <div class="max-w-3xl mx-auto text-center space-y-3 mb-6">
                <h2 class="text-2xl font-extrabold text-slate-900">Real-Time URL Security Scanner</h2>
                <p class="text-slate-600 text-sm">Enter any web address to execute live feature extraction, median imputation, and ensemble prediction.</p>
            </div>

            <!-- Quick presets -->
            <div class="flex flex-wrap items-center justify-center gap-2 mb-4 text-xs font-medium">
                <span class="text-slate-400">Quick Test:</span>
                <button onclick="setAndScan('https://www.google.com/search?q=machine+learning')" class="px-2.5 py-1 bg-slate-100 hover:bg-slate-200 rounded-md text-slate-700">Google (Safe)</button>
                <button onclick="setAndScan('https://github.com/torvalds/linux')" class="px-2.5 py-1 bg-slate-100 hover:bg-slate-200 rounded-md text-slate-700">GitHub (Safe)</button>
                <button onclick="setAndScan('http://paypal-security-account-update.xyz/login.php?cmd=verify')" class="px-2.5 py-1 bg-red-50 hover:bg-red-100 rounded-md text-red-700">Fake PayPal (Phish)</button>
                <button onclick="setAndScan('http://192.168.1.105/bank-of-america/secure-login.html')" class="px-2.5 py-1 bg-red-50 hover:bg-red-100 rounded-md text-red-700">Raw IP Bank (Phish)</button>
                <button onclick="setAndScan('http://appleid-support-icloud-security.top/auth/verify')" class="px-2.5 py-1 bg-red-50 hover:bg-red-100 rounded-md text-red-700">Fake Apple ID (Phish)</button>
            </div>

            <!-- Input bar -->
            <div class="flex flex-col sm:flex-row gap-3">
                <input id="urlInput" type="text" placeholder="e.g. https://example.com/login" class="flex-1 px-4 py-3 rounded-xl border border-slate-300 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono text-sm bg-slate-50 focus:bg-white transition">
                <button id="scanBtn" onclick="scanUrl()" class="px-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-xl transition shadow flex items-center justify-center space-x-2">
                    <span id="btnText">Analyze URL</span>
                    <span id="btnSpinner" class="hidden animate-spin">⏳</span>
                </button>
            </div>

            <!-- Result Card Container -->
            <div id="resultCard" class="hidden mt-8 pt-6 border-t border-slate-100">
                <div id="verdictBox" class="rounded-xl p-6 text-center space-y-3">
                    <div id="verdictBadge" class="inline-block px-3 py-1 rounded-full text-xs font-bold tracking-wide uppercase"></div>
                    <h3 id="verdictTitle" class="text-2xl font-black"></h3>
                    <p id="verdictUrl" class="font-mono text-sm break-all opacity-80 max-w-2xl mx-auto"></p>
                    
                    <div class="grid grid-cols-3 gap-4 max-w-lg mx-auto pt-4 text-center">
                        <div class="bg-white/80 rounded-lg p-3 shadow-sm border border-black/5">
                            <div class="text-xs uppercase font-semibold text-slate-500">Phishing Risk</div>
                            <div id="phishProb" class="text-xl font-bold mt-0.5"></div>
                        </div>
                        <div class="bg-white/80 rounded-lg p-3 shadow-sm border border-black/5">
                            <div class="text-xs uppercase font-semibold text-slate-500">Legitimate Conf.</div>
                            <div id="legitProb" class="text-xl font-bold mt-0.5"></div>
                        </div>
                        <div class="bg-white/80 rounded-lg p-3 shadow-sm border border-black/5">
                            <div class="text-xs uppercase font-semibold text-slate-500">Risk Severity</div>
                            <div id="riskLevel" class="text-xl font-bold mt-0.5"></div>
                        </div>
                    </div>
                </div>

                <!-- Explainability factors -->
                <div class="mt-6 bg-slate-50 rounded-xl p-5 border border-slate-200">
                    <h4 class="font-bold text-slate-900 mb-3 flex items-center space-x-2">
                        <span>🔍</span>
                        <span>Explainable AI (XAI) Decision Triggers</span>
                    </h4>
                    <ul id="reasonsList" class="space-y-2 text-sm text-slate-700"></ul>
                </div>

                <!-- Features Accordion -->
                <div class="mt-4">
                    <button onclick="toggleFeatures()" class="w-full py-2.5 px-4 bg-slate-100 hover:bg-slate-200 rounded-lg text-xs font-semibold text-slate-700 flex justify-between items-center transition">
                        <span>🔬 Inspect All 74 Extracted Feature Values</span>
                        <span id="accordionArrow">▼</span>
                    </button>
                    <div id="featuresTableContainer" class="hidden mt-3 max-h-96 overflow-y-auto border border-slate-200 rounded-lg bg-white">
                        <table class="w-full text-xs text-left">
                            <thead class="bg-slate-50 text-slate-600 border-b">
                                <tr>
                                    <th class="px-4 py-2 font-semibold">Feature Name</th>
                                    <th class="px-4 py-2 font-semibold">Extracted Numerical Value</th>
                                </tr>
                            </thead>
                            <tbody id="featuresTableBody" class="divide-y divide-slate-100 font-mono"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>

        <!-- Benchmark Matrix Grid -->
        <section class="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 space-y-4">
                <h3 class="font-bold text-lg text-slate-900 flex items-center space-x-2">
                    <span>📊</span>
                    <span>Model Benchmark Results</span>
                </h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-xs text-left">
                        <thead class="bg-slate-50 text-slate-600 border-b">
                            <tr>
                                <th class="p-2">Model</th>
                                <th class="p-2">Accuracy</th>
                                <th class="p-2">F1-Score</th>
                                <th class="p-2">ROC-AUC</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-slate-100">
                            <tr class="bg-emerald-50/50 font-semibold text-emerald-900"><td class="p-2">Stacking AllBase (Best)</td><td class="p-2">0.9885</td><td class="p-2">0.9861</td><td class="p-2">0.9985</td></tr>
                            <tr><td class="p-2">LightGBM</td><td class="p-2">0.9868</td><td class="p-2">0.9840</td><td class="p-2">0.9985</td></tr>
                            <tr><td class="p-2">Random Forest (Active)</td><td class="p-2">0.9851</td><td class="p-2">0.9820</td><td class="p-2">0.9983</td></tr>
                            <tr><td class="p-2">CNN-BiLSTM (DL)</td><td class="p-2">0.9841</td><td class="p-2">0.9807</td><td class="p-2">0.9986</td></tr>
                            <tr><td class="p-2">CatBoost</td><td class="p-2">0.9832</td><td class="p-2">0.9796</td><td class="p-2">0.9979</td></tr>
                            <tr><td class="p-2">XGBoost</td><td class="p-2">0.9816</td><td class="p-2">0.9777</td><td class="p-2">0.9976</td></tr>
                            <tr><td class="p-2">Logistic Regression</td><td class="p-2">0.9440</td><td class="p-2">0.9317</td><td class="p-2">0.9845</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 space-y-4">
                <h3 class="font-bold text-lg text-slate-900 flex items-center space-x-2">
                    <span>⚙️</span>
                    <span>Preprocessing Workflow (UML501)</span>
                </h3>
                <div class="space-y-3 text-xs text-slate-600">
                    <div class="p-3 bg-slate-50 rounded-lg border border-slate-100">
                        <strong class="text-slate-900 block mb-1">1. 74-Feature Engine</strong>
                        Extracts 53 Lexical, 9 DNS, and 12 WHOIS parameters across length, entropy, and symbol ratios.
                    </div>
                    <div class="p-3 bg-slate-50 rounded-lg border border-slate-100">
                        <strong class="text-slate-900 block mb-1">2. Data Leakage Prevention</strong>
                        Stratified 80/20 train/test split executed prior to fitting imputers or scalers.
                    </div>
                    <div class="p-3 bg-slate-50 rounded-lg border border-slate-100">
                        <strong class="text-slate-900 block mb-1">3. Dual Strategy Imputation & Scaling</strong>
                        Median Imputation for Tree ensembles (invariant to scaling); StandardScaler for linear models.
                    </div>
                </div>
            </div>
        </section>
    </main>

    <footer class="border-t border-slate-200 py-6 text-center text-xs text-slate-500">
        Explainable Phishing URL Detection Framework &bull; UML501 Machine Learning Project
    </footer>

    <script>
        function setAndScan(url) {
            document.getElementById('urlInput').value = url;
            scanUrl();
        }

        async function scanUrl() {
            const url = document.getElementById('urlInput').value.trim();
            if (!url) {
                alert('Please enter a URL first!');
                return;
            }

            const btn = document.getElementById('scanBtn');
            const btnText = document.getElementById('btnText');
            const btnSpinner = document.getElementById('btnSpinner');
            btn.disabled = true;
            btnText.innerText = 'Scanning...';
            btnSpinner.classList.remove('hidden');

            try {
                const response = await fetch('/api/scan?url=' + encodeURIComponent(url));
                const data = await response.json();
                renderResult(data);
            } catch (err) {
                alert('Error analyzing URL: ' + err);
            } finally {
                btn.disabled = false;
                btnText.innerText = 'Analyze URL';
                btnSpinner.classList.add('hidden');
            }
        }

        function renderResult(data) {
            const card = document.getElementById('resultCard');
            card.classList.remove('hidden');

            const box = document.getElementById('verdictBox');
            const badge = document.getElementById('verdictBadge');
            const title = document.getElementById('verdictTitle');
            const urlEl = document.getElementById('verdictUrl');
            const phishProb = document.getElementById('phishProb');
            const legitProb = document.getElementById('legitProb');
            const riskLevel = document.getElementById('riskLevel');

            urlEl.innerText = data.url;
            phishProb.innerText = data.phishing_probability + '%';
            legitProb.innerText = data.legitimate_probability + '%';
            riskLevel.innerText = data.risk_level;

            if (data.is_phishing) {
                box.className = 'rounded-xl p-6 text-center space-y-3 bg-red-50 border-2 border-red-500 text-red-950';
                badge.className = 'inline-block px-3 py-1 rounded-full text-xs font-bold uppercase bg-red-600 text-white';
                badge.innerText = 'Phishing Threat';
                title.innerText = '🚨 WARNING: Phishing URL Detected';
                title.className = 'text-2xl font-black text-red-700';
            } else {
                box.className = 'rounded-xl p-6 text-center space-y-3 bg-emerald-50 border-2 border-emerald-500 text-emerald-950';
                badge.className = 'inline-block px-3 py-1 rounded-full text-xs font-bold uppercase bg-emerald-600 text-white';
                badge.innerText = 'Verified Safe';
                title.innerText = '✅ Legitimate URL Confirmed';
                title.className = 'text-2xl font-black text-emerald-700';
            }

            const reasonsList = document.getElementById('reasonsList');
            reasonsList.innerHTML = '';
            data.reasons.forEach(r => {
                const li = document.createElement('li');
                li.className = 'flex items-start space-x-2';
                li.innerHTML = `<span>${data.is_phishing ? '⚠️' : '🛡️'}</span><span><b>${r}</b></span>`;
                reasonsList.appendChild(li);
            });

            const tbody = document.getElementById('featuresTableBody');
            tbody.innerHTML = '';
            for (const [k, v] of Object.entries(data.features)) {
                const tr = document.createElement('tr');
                tr.innerHTML = `<td class="px-4 py-1.5 font-medium text-slate-700">${k}</td><td class="px-4 py-1.5 text-slate-500">${v}</td>`;
                tbody.appendChild(tr);
            }
            
            card.scrollIntoView({ behavior: 'smooth' });
        }

        function toggleFeatures() {
            const table = document.getElementById('featuresTableContainer');
            const arrow = document.getElementById('accordionArrow');
            if (table.classList.contains('hidden')) {
                table.classList.remove('hidden');
                arrow.innerText = '▲';
            } else {
                table.classList.add('hidden');
                arrow.innerText = '▼';
            }
        }
    </script>
</body>
</html>
"""


class PhishingDashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/api/scan":
            params = parse_qs(parsed.query)
            url = params.get("url", [""])[0]
            if not url:
                res = {"error": "Missing 'url' parameter"}
            else:
                res = predict_url(url, enable_network=False)

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(res).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port: int = 8080):
    server = HTTPServer(("localhost", port), PhishingDashboardHandler)
    url = f"http://localhost:{port}"
    print(f"\n=======================================================")
    print(f"🚀 Phishing URL Detection Dashboard running at:")
    print(f"   👉 {url}")
    print(f"=======================================================\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    start_server(port)
