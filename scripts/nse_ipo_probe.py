#!/usr/bin/env python3
"""Probe NSE's (unofficial) IPO endpoints for the category-wise subscription split.

Why this exists
---------------
The QIB / NII / Retail subscription split — the deciding feature for the IPO
advisor — originates at the exchanges. NSE publishes it live during the bidding
window; every dashboard (Chittorgarh, IPO Alerts, etc.) just re-serves NSE/BSE.
Going direct puts you upstream of all of them, for free. This script:

  1. Opens an NSE browser-like session (NSE blocks requests without cookies +
     realistic headers, so we bootstrap a session first).
  2. Lists current IPOs, then for a symbol pulls the category-wise demand and
     the IPO detail record.
  3. Grades the result against the advisor checklist and — the important part —
     detects QIB / NII / sNII / bNII / Retail / Anchor by scanning field VALUES,
     because NSE encodes the category in a value like
     "category": "Qualified Institutional Buyers(QIBs)" rather than a `qib` key.

Endpoint paths are undocumented and occasionally change, so several candidates
are tried and the script reports which responded. It talks to the live site, so
run it where NSE is reachable (an Indian IP / your phone via Termux) — this repo's
cloud sessions have nseindia.com blocked by network policy. Stdlib only.

Usage
-----
    python scripts/nse_ipo_probe.py                 # list current IPOs, auto-probe the first
    python scripts/nse_ipo_probe.py --symbol SWIGGY # probe a specific symbol
    python scripts/nse_ipo_probe.py --list-only     # just enumerate current IPOs
    python scripts/nse_ipo_probe.py --symbol X --raw # dump full raw JSON
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

HOME = "https://www.nseindia.com/"
IPO_PAGE = "https://www.nseindia.com/market-data/all-upcoming-issues-ipo"

# Candidate list endpoints (current / upcoming issues). Tried in order.
LIST_ENDPOINTS = [
    ("all-upcoming-issues", "https://www.nseindia.com/api/all-upcoming-issues?category=ipo"),
    ("ipo-current-issue", "https://www.nseindia.com/api/ipo-current-issue"),
]
# Candidate category-subscription endpoints ({sym} filled in). The first that
# returns rows is used. This is the endpoint that carries the QIB/NII/RII split.
CATEGORY_ENDPOINTS = [
    "https://www.nseindia.com/api/ipo-active-category?symbol={sym}",
    "https://www.nseindia.com/api/ipo-active-category?symbol={sym}&series=EQ",
]
DETAIL_ENDPOINTS = [
    "https://www.nseindia.com/api/ipo-detail?symbol={sym}&series=EQ",
    "https://www.nseindia.com/api/ipo-detail?symbol={sym}",
]

BROWSER_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "identity",  # avoid gzip so urllib returns plain text
}

# --- Structural checklist (key-name based). Subscription categories are handled
# --- separately by value-scanning, since NSE puts them in values, not keys. -----
CHECKLIST = [
    ("must", "Price band (low/high)", ["price_band", "min_price", "max_price", "issueprice", "priceband", "band"]),
    ("must", "Lot size / min bid qty", ["lotsize", "lot_size", "marketlot", "market_lot", "bidlot",
                                        "minbid", "min_bid", "minimumbid", "minqty", "min_qty"]),
    ("must", "Issue size", ["issue_size", "issuesize", "totalissue", "offer_size", "issueamount"]),
    ("must", "Issue type / OFS split", ["issue_type", "issuetype", "ofs", "fresh", "offerforsale"]),
    ("must", "Series / SME vs mainboard hint", ["series", "sme", "board", "platform"]),
    ("must", "Open date", ["open", "start", "biddingstart", "issuestart"]),
    ("must", "Close date", ["close", "end", "biddingend", "issueend"]),
    ("must", "Listing date", ["listing", "listingdate"]),
    ("must", "Symbol / stable id", ["symbol", "isin", "series"]),
    ("must", "ISIN", ["isin"]),
    ("must", "Listing / issue price", ["issueprice", "finalprice", "cutoff", "cut_off", "listingprice"]),
    ("useful", "Anchor investor data", ["anchor"]),
    ("useful", "Issue P/E", ["pe", "p_e", "priceearning"]),
    ("bonus", "Company / name", ["company", "name"]),
    ("bonus", "Status", ["status", "state"]),
]
TIER_LABEL = {"must": "MUST-HAVE ", "useful": "USEFUL    ", "bonus": "BONUS     "}


class NSE:
    """A minimal NSE session that carries cookies and browser headers."""

    def __init__(self):
        self.cj = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cj))

    def _open(self, url: str, accept: str, referer: str | None = None) -> tuple[int, str]:
        req = urllib.request.Request(url)
        for k, v in BROWSER_HEADERS.items():
            req.add_header(k, v)
        req.add_header("Accept", accept)
        if referer:
            req.add_header("Referer", referer)
        try:
            with self.opener.open(req, timeout=30) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            return 0, f"network error: {exc.reason}"

    def bootstrap(self) -> None:
        """Prime cookies by visiting the homepage and the IPO page as a browser."""
        self._open(HOME, "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        self._open(IPO_PAGE, "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", referer=HOME)

    def api(self, url: str) -> tuple[int, object]:
        status, body = self._open(url, "application/json, text/plain, */*", referer=IPO_PAGE)
        if status in (401, 403):
            # Session likely stale — re-prime once and retry.
            self.bootstrap()
            status, body = self._open(url, "application/json, text/plain, */*", referer=IPO_PAGE)
        try:
            return status, json.loads(body)
        except json.JSONDecodeError:
            return status, body


def _rows(payload: object) -> list:
    """Best-effort extraction of a list of records from varied NSE shapes."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "dataList", "activeIssues", "upcomingIssues", "records"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
        # single-object detail response
        return [payload]
    return []


def _flatten_keys(obj, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            out.update(_flatten_keys(val, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        for item in obj:
            out.update(_flatten_keys(item, f"{prefix}[]"))
    else:
        out[prefix] = obj
    return out


def _match(fragments: list[str], flat: dict[str, object]) -> list[str]:
    hits = []
    for path in flat:
        segments = re.split(r"\.|\[\]", path.lower())
        for frag in (f.lower() for f in fragments):
            strict = "_" not in frag and len(frag) < 4
            for seg in segments:
                if (frag in seg.split("_")) if strict else (frag in seg):
                    hits.append(path)
                    break
            else:
                continue
            break
    return sorted(set(hits))


def detect_categories(record: object) -> dict[str, bool]:
    """Scan ALL text (keys + values) for category markers — the deciding check."""
    text = json.dumps(record, ensure_ascii=False).lower()
    return {
        "QIB": any(m in text for m in ("qib", "qualified institutional")),
        "NII (incl. sNII/bNII)": any(m in text for m in ("nii", "non institutional", "non-institutional",
                                                          "hni", "shni", "bhni")),
        "Retail": any(m in text for m in ("retail", "individual investor", "rii", "\"rii\"")),
        "Anchor": "anchor" in text,
        "Subscription multiple/times": any(m in text for m in ("subscription", "subscribed", "times",
                                                               "noofsharesbid", "sharesbid")),
    }


# NSE's ipo-active-category rows are keyed by srNo; this maps the ones we score.
SRNO_LABELS = {
    "1": "QIB (Qualified Institutional Buyers)",
    "2": "NII (Non-Institutional, combined)",
    "2.1": "  bNII  (bid > Rs 10 lakh)",
    "2.2": "  sNII  (bid Rs 2-10 lakh)",
    "3": "Retail (RIIs)",
    "4": "Employees",
}


def _find_datalists(obj) -> list:
    """Recursively collect every `dataList` array in the payload."""
    out = []
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key == "dataList" and isinstance(val, list):
                out.append(val)
            else:
                out += _find_datalists(val)
    elif isinstance(obj, list):
        for item in obj:
            out += _find_datalists(item)
    return out


def subscription_summary(record: object) -> bool:
    """Print the category-wise subscription multiples from an ipo-active-category shape.

    Returns True if a category table was found and printed.
    """
    lists = _find_datalists(record)
    if not lists:
        return False
    rows = {str(r.get("srNo")): r for r in lists[0] if isinstance(r, dict)}
    print("\n" + "=" * 72)
    print("SUBSCRIPTION MULTIPLES (noOfTotalMeant = times subscribed)")
    print("=" * 72)
    printed = False
    for srno, label in SRNO_LABELS.items():
        row = rows.get(srno)
        if not row:
            continue
        printed = True
        times = row.get("noOfTotalMeant") or ""
        offered = row.get("noOfShareOffered") or ""
        bid = row.get("noOfSharesBid") or ""
        try:
            times = f"{float(times):.3f}x"
        except (TypeError, ValueError):
            times = str(times) or "(n/a)"
        print(f"  {label:<38} {times:>10}   [offered {offered} / bid {bid}]")
    # The Total row has srNo null; find it by category name.
    total = next((r for r in lists[0] if isinstance(r, dict) and str(r.get("category")).lower() == "total"), None)
    if total:
        try:
            tv = f"{float(total.get('noOfTotalMeant')):.3f}x"
        except (TypeError, ValueError):
            tv = str(total.get("noOfTotalMeant"))
        print(f"  {'TOTAL':<38} {tv:>10}")
    return printed


def grade(record: object) -> None:
    flat = _flatten_keys(record)
    print("\n" + "=" * 72)
    print("SUBSCRIPTION CATEGORY DETECTION (value-aware — the deciding check)")
    print("=" * 72)
    cats = detect_categories(record)
    for label, present in cats.items():
        print(f"[{'OK  ' if present else 'MISS'}] {label}")

    subscription_summary(record)

    print("\n" + "=" * 72)
    print("STRUCTURAL FIELDS (key-name based)")
    print("=" * 72)
    for tier, concept, fragments in CHECKLIST:
        hits = _match(fragments, flat)
        mark = "OK  " if hits else "MISS"
        shown = ", ".join(hits[:4]) + (" ..." if len(hits) > 4 else "") if hits else "(no matching field)"
        print(f"[{mark}] {TIER_LABEL[tier]} {concept:<34} -> {shown}")

    print("-" * 72)
    if cats["QIB"] and cats["Retail"]:
        print("DECIDING FIELD: category split PRESENT -> NSE can feed the model's core feature.")
    else:
        print("DECIDING FIELD: category split NOT detected in this record. Try an IPO that is")
        print("  currently OPEN (subscription only populates during/after the bidding window),")
        print("  and confirm the category endpoint actually returned rows (see raw dump).")
    print("=" * 72)


def _ident(rec: dict) -> tuple[str | None, str | None]:
    if not isinstance(rec, dict):
        return None, None
    sym = rec.get("symbol") or rec.get("Symbol") or rec.get("series") and rec.get("symbol")
    name = rec.get("companyName") or rec.get("company") or rec.get("name") or rec.get("issuerName")
    return sym, name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", help="IPO symbol to probe (e.g. SWIGGY). Omit to auto-pick from the list.")
    parser.add_argument("--list-only", action="store_true", help="Only enumerate current IPOs.")
    parser.add_argument("--raw", action="store_true", help="Dump full raw JSON of the probed records.")
    args = parser.parse_args()

    nse = NSE()
    print("Priming NSE session (cookies + headers) ...")
    nse.bootstrap()

    symbol = args.symbol
    if not symbol or args.list_only:
        listed = None
        for name, url in LIST_ENDPOINTS:
            status, payload = nse.api(url)
            rows = _rows(payload)
            print(f"  list endpoint '{name}': HTTP {status}, {len(rows)} record(s)")
            if rows:
                listed = rows
                break
        if not listed:
            print("\nNo current IPOs returned by any list endpoint. Either none are open right now,")
            print("or the endpoint path changed. Re-run with --symbol <SYM> to probe directly.")
            return 0
        print(f"\nCurrent IPOs ({len(listed)}):")
        for i, rec in enumerate(listed[:25]):
            sym, name = _ident(rec)
            print(f"  [{i}] symbol={sym!r:>14}  {name!r}")
        if args.list_only:
            if args.raw:
                print("\n----- RAW LIST -----")
                print(json.dumps(listed, indent=2, ensure_ascii=False))
            return 0
        symbol = _ident(listed[0])[0]
        if not symbol:
            print("\nCouldn't derive a symbol from the first record; pass --symbol explicitly.")
            return 0
        print(f"\nAuto-probing symbol: {symbol!r}")

    # Pull category-wise subscription (the key endpoint) + detail, merge for grading.
    merged: dict = {}
    category_payload = None
    for tmpl in CATEGORY_ENDPOINTS:
        status, payload = nse.api(tmpl.format(sym=urllib.parse.quote(symbol)))
        rows = _rows(payload)
        print(f"  category endpoint: HTTP {status}, {len(rows)} row(s)  [{tmpl.split('?')[0].rsplit('/',1)[-1]}]")
        if rows and status == 200:
            category_payload = payload
            merged["categories"] = payload
            break
    for tmpl in DETAIL_ENDPOINTS:
        status, payload = nse.api(tmpl.format(sym=urllib.parse.quote(symbol)))
        if status == 200 and isinstance(payload, (dict, list)):
            print(f"  detail endpoint:   HTTP {status}  [{tmpl.split('?')[0].rsplit('/',1)[-1]}]")
            merged["detail"] = payload
            break

    if not merged:
        print("\nNo data returned for this symbol. If it's an SME or BSE-only issue it won't be on NSE;")
        print("subscription also only populates while an IPO is OPEN. Try an NSE mainboard IPO that's live.")
        return 0

    grade(merged)
    if args.raw:
        print("\n----- RAW MERGED RECORD -----")
        print(json.dumps(merged, indent=2, ensure_ascii=False))
    else:
        print("\n(Re-run with --raw to see the full JSON, incl. the exact category rows.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
