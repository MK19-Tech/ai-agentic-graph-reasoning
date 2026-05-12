"""
check_env.py — Verify that all required packages and env vars are present.
Run with:  python check_env.py
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ── Package checks ────────────────────────────────────────────────────────────
PACKAGES = [
    "langchain",
    "langchain_community",
    "langchain_groq",
    "langgraph",
    "groq",
    "ddgs",
    "dotenv",
    "pydantic",
]

print("── Package checks ───────────────────────────────────")
all_ok = True
for pkg in PACKAGES:
    try:
        __import__(pkg)
        print(f"  ✅  {pkg}")
    except ImportError as exc:
        print(f"  ❌  {pkg}  →  {exc}")
        all_ok = False

# ── Env var checks ────────────────────────────────────────────────────────────
REQUIRED_ENV_VARS = ["GROQ_API_KEY"]

print("\n── Environment variable checks ──────────────────────")
for var in REQUIRED_ENV_VARS:
    value = os.getenv(var)
    if value:
        masked = value[:6] + "..." + value[-4:]
        print(f"  ✅  {var} = {masked}")
    else:
        print(f"  ❌  {var} is NOT set")
        all_ok = False

print()
if all_ok:
    print("All checks passed — you're good to go! 🚀")
else:
    print("Some checks failed. Fix the issues above, then re-run.")
    sys.exit(1)
