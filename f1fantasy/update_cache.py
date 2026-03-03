from datetime import datetime
from f1fantasy.ergast import fetch_all_supporting

def main():
    y = datetime.utcnow().year
    # refresh current & last season supporting data
    for yr in (y, y-1):
        fetch_all_supporting(yr, force_refresh=True)
    print("Cache updated (results, qualifying, sprint, schedule).")

if __name__ == "__main__":
    main()
