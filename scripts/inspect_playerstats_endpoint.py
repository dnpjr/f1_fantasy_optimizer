from __future__ import annotations

import json
from pathlib import Path
import pprint

import requests


URL = "https://fantasy.formula1.com/feeds/popup/playerstats_124.json"
OUT = Path("tests/fixtures/playerstats_124_redacted.json")
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def _trim_payload(payload: dict) -> dict:
    value = payload.get("Value", {})
    trimmed = {
        "Value": {
            "PlayerId": value.get("PlayerId"),
            "PlayerSkill": value.get("PlayerSkill"),
            "GamedayWiseStats": (value.get("GamedayWiseStats") or [])[:3],
            "TourWiseStats": (value.get("TourWiseStats") or [])[:1],
            "FixtureWiseStats": (value.get("FixtureWiseStats") or [])[:2],
            "MatchWiseStats": (value.get("MatchWiseStats") or [])[:3],
        },
        "FeedTime": payload.get("FeedTime", {}),
    }
    return trimmed


def main() -> None:
    response = requests.get(URL, timeout=20, headers={"User-Agent": UA, "Accept": "application/json,text/plain,*/*"})
    response.raise_for_status()
    payload = response.json()
    value = payload.get("Value", {})

    print("URL:", URL)
    print("Top-level keys:", list(payload.keys()))
    print("Value keys:", list(value.keys()))
    print("PlayerId:", value.get("PlayerId"))
    print("PlayerSkill:", value.get("PlayerSkill"))

    for key in ["GamedayWiseStats", "TourWiseStats", "FixtureWiseStats", "MatchWiseStats"]:
        items = value.get(key) or []
        print(f"\n{key}: {type(items).__name__}, rows={len(items)}")
        if items:
            print("First row keys:", list(items[0].keys()))
            pprint.pp(items[0])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(_trim_payload(payload), indent=2), encoding="utf-8")
    print(f"\nSaved redacted sample: {OUT}")


if __name__ == "__main__":
    main()
