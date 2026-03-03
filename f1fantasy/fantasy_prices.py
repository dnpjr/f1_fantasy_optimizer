from __future__ import annotations
import requests

BASE = "https://fantasy.formula1.com"

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

def _sess() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-GB,en;q=0.9",
        "Referer": BASE + "/",
        "Origin": BASE,
        "Connection": "keep-alive",
    })
    return s

def try_get_json(url: str) -> dict:
    s = _sess()

    # First hit homepage to obtain any cookies / bot checks
    s.get(BASE + "/", timeout=20)

    r = s.get(url, timeout=20)
    print("URL:", r.url)
    print("STATUS:", r.status_code)
    print("CTYPE:", r.headers.get("content-type"))
    print("HEAD:", r.text[:120].replace("\n", " "))

    r.raise_for_status()
    return r.json()

def main():
    candidates = [
        BASE + "/static-assets/mixapi.json",
        BASE + "/static-assets/build/mixapi.json",
        BASE + "/static-assets/build/static/mixapi.json",
        BASE + "/mixapi.json",
    ]

    last_err = None
    for u in candidates:
        try:
            data = try_get_json(u)
            print("\n✅ Got JSON from:", u)
            print("Top-level keys:", list(data.keys())[:30])
            return
        except Exception as e:
            print("❌ Failed:", u, "->", repr(e))
            last_err = e

    raise RuntimeError("Could not fetch mixapi JSON from any candidate URL") from last_err

if __name__ == "__main__":
    main()