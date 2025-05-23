# watchlist_utils.py
import os, json, pathlib

BASE_DIR = pathlib.Path(__file__).resolve().parent
WL_FILES = {
    "long":  BASE_DIR / "watchlist_long.json",
    "short": BASE_DIR / "watchlist_short.json",
}

def _ensure_file(side: str):
    if side not in WL_FILES:
        raise ValueError("side must be 'long' or 'short'")
    if not WL_FILES[side].exists():
        WL_FILES[side].write_text("[]")

# -------------------------------------------------------------------
def load_watchlist(side: str = "long") -> list[str]:
    """Return the watch-list for *side* ('long' | 'short')."""
    _ensure_file(side)
    return json.loads(WL_FILES[side].read_text())

def save_watchlist(side: str, tickers: list[str]) -> None:
    """Overwrite *side* list with upper-cased, de-duped symbols."""
    uniq = sorted(set(s.upper() for s in tickers if s.strip()))
    WL_FILES[side].write_text(json.dumps(uniq, indent=2))

# -------------------------------------------------------------------
def manage_watchlist() -> None:
    """
    Minimal CLI editor – lets the user pick which list to edit,
    then add/remove symbols.
    """
    side = input("\nEdit which list?  (L)ong  /  (S)hort  : ").lower()
    side = "long" if side.startswith("l") else "short"
    lst  = load_watchlist(side)

    while True:
        print(f"\nCurrent {side.capitalize()} Watch-list  →  {', '.join(lst) or '(empty)'}")
        cmd = input("(A)dd  (R)emove  (C)lear  (Q)uit : ").lower()

        if cmd.startswith("q"):
            save_watchlist(side, lst)
            break
        if cmd.startswith("c"):
            lst.clear()
        elif cmd.startswith("a"):
            sym = input("Add symbol: ").strip().upper()
            if sym and sym not in lst:
                lst.append(sym)
        elif cmd.startswith("r"):
            sym = input("Remove symbol: ").strip().upper()
            if sym in lst:
                lst.remove(sym)
