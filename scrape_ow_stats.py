import time
import requests
import pandas as pd

# Base URL for JSON data
BASE_URL = "https://overwatch.blizzard.com/en-us/rates/data/"

# Configure map list (fill in more slugs once you collect them)
MAP_INFO = [
     # CONTROL
    {"name": "Busan",                 "map_type": "Control",     "slug": "busan"},
    {"name": "Ilios",                 "map_type": "Control",     "slug": "ilios"},
    {"name": "Lijiang Tower",         "map_type": "Control",     "slug": "lijiang-tower"},
    {"name": "Nepal",                 "map_type": "Control",     "slug": "nepal"},
    {"name": "Oasis",                 "map_type": "Control",     "slug": "oasis"},
    {"name": "Samoa",                 "map_type": "Control",     "slug": "samoa"},
    {"name": "Antarctic Peninsula",   "map_type": "Control",     "slug": "antarctic-peninsula"},

    # ESCORT
    {"name": "Circuit Royal",         "map_type": "Escort",      "slug": "circuit-royal"},
    {"name": "Dorado",                "map_type": "Escort",      "slug": "dorado"},
    {"name": "Havana",                "map_type": "Escort",      "slug": "havana"},
    {"name": "Junkertown",            "map_type": "Escort",      "slug": "junkertown"},
    {"name": "Rialto",                "map_type": "Escort",      "slug": "rialto"},
    {"name": "Route 66",              "map_type": "Escort",      "slug": "route-66"},
    {"name": "Shambali Monastery",    "map_type": "Escort",      "slug": "shambali-monastery"},
    {"name": "Watchpoint: Gibraltar", "map_type": "Escort",      "slug": "watchpoint-gibraltar"},

    # HYBRID
    {"name": "Blizzard World",        "map_type": "Hybrid",      "slug": "blizzard-world"},
    {"name": "Eichenwalde",           "map_type": "Hybrid",      "slug": "eichenwalde"},
    {"name": "Hollywood",             "map_type": "Hybrid",      "slug": "hollywood"},
    {"name": "King's Row",            "map_type": "Hybrid",      "slug": "kings-row"},
    {"name": "Midtown",               "map_type": "Hybrid",      "slug": "midtown"},
    {"name": "Numbani",               "map_type": "Hybrid",      "slug": "numbani"},
    {"name": "Paraíso",               "map_type": "Hybrid",      "slug": "paraiso"},

    # PUSH
    {"name": "Colosseo",              "map_type": "Push",        "slug": "colosseo"},
    {"name": "Esperança",             "map_type": "Push",        "slug": "esperanca"},
    {"name": "New Queen Street",      "map_type": "Push",        "slug": "new-queen-street"},
    {"name": "Runasapi",              "map_type": "Push",        "slug": "runasapi"},

    # FLASHPOINT
    {"name": "New Junk City",         "map_type": "Flashpoint",  "slug": "new-junk-city"},
    {"name": "Suravasa",              "map_type": "Flashpoint",  "slug": "suravasa"},
]


# Filters
BASE_PARAMS = {
    "input": "PC",
    "region": "Americas",
    "role": "All",
    "rq": 1,       # Competitive – Role Queue
    "tier": "All",
    # "map" will be added per request
}

def fetch_map_data(map_slug: str) -> dict:
    params = BASE_PARAMS.copy()
    params["map"] = map_slug
    headers = {"User-Agent": "APPM3310-OW-Scraper/0.1"}
    resp = requests.get(BASE_URL, params=params, headers=headers)
    resp.raise_for_status()
    return resp.json()

def extract_hero_rows(json_data: dict, map_name: str, map_type: str, mode:str="Competitive - Role Queue") -> list[dict]:
    rows = []
    for rec in json_data.get("rates", []):
        hero = rec.get("hero", {})
        cells = rec.get("cells", {})
        name = hero.get("name")
        win = cells.get("winrate")
        pick = cells.get("pickrate")
        if name is None or win is None:
            continue
        rows.append({
            "hero": name,
            "map": map_name,
            "map_type": map_type,
            "mode": mode,
            "pickrate": pick,
            "winrate": win / 100.0   # convert from percent to decimal
        })
    return rows

def main():
    all_rows = []
    for info in MAP_INFO:
        slug = info["slug"]
        name = info["name"]
        mtype = info["map_type"]
        print(f"Fetching data for map: {name} (slug={slug}) ...")
        try:
            j = fetch_map_data(slug)
        except Exception as e:
            print("  Error fetching:", e)
            continue

        rows = extract_hero_rows(j, name, mtype)
        print("  → got", len(rows), "heroes")
        all_rows.extend(rows)

        time.sleep(1.0)  # polite pause

    df = pd.DataFrame(all_rows)
    out = "overwatch_winrates_by_map.csv"
    df.to_csv(out, index=False)
    print("Saved", len(df), "rows to", out)

if __name__ == "__main__":
    main()