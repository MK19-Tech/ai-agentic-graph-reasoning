packages = [
    "langchain",
    "langchain_community",
    "langchain_groq",
    "langgraph",
    "groq",
    "ddgs",
    "dotenv"
]

for pkg in packages:

    try:
        __import__(pkg)
        print(f"✅ {pkg} OK")

    except Exception as e:
        print(f"❌ {pkg} FAILED -> {e}")
