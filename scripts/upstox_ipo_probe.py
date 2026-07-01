#!/usr/bin/env python3
"""Probe the Upstox IPO API and grade it against the advisor's data requirements.

Why this exists
---------------
Before wiring the IPO advisor to Upstox as a data source, we need a field-by-field
answer to "can it feed the model?" — not just "does an IPO endpoint exist?". The
deciding field is the QIB / NII / Retail subscription split; everything else is
either commonly available or fillable elsewhere. This script calls the live IPO
endpoints, dumps exactly what comes back, and prints a
"fully replaces / partially replaces / doesn't fit" verdict.

It talks to the real API, so it must run on your machine (this repo's cloud
sessions have upstox.com blocked by network policy). Stdlib only — no pip install.

Usage
-----
    export UPSTOX_ACCESS_TOKEN="<paste your token>"

    # List IPOs, then auto-probe the first one's detail record:
    python scripts/upstox_ipo_probe.py

    # Filter the list (params are passed through as-is; see --help):
    python scripts/upstox_ipo_probe.py --status open --issue-type MAINBOARD

    # Probe a specific IPO by id (from the list output) and dump raw JSON:
    python scripts/upstox_ipo_probe.py --ipo-id <ID> --raw

The token can be the long-lived **Analytics Access Token** or a standard OAuth
access token — both are sent as `Authorization: Bearer <token>`. If the Analytics
token is rejected on the IPO endpoint (403 / scope error), the script says so
clearly; fall back to a full App + OAuth login token in that case.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://api.upstox.com"
LIST_PATH = "/v2/ipos"
DETAIL_PATH = "/v2/ipos/{ipo_id}"

# ---------------------------------------------------------------------------
# The requirements checklist, encoded so the probe can self-grade.
#
# Upstox's exact field names are not fully documented, so each concept lists
# candidate key-name fragments (matched case-insensitively against the flattened
# JSON key paths). This is deliberately generous: a hit means "a field that looks
# like this is present"; always eyeball the --raw dump to confirm semantics.
#
# tier: "must"  -> model can't run without it
#       "useful"-> feeds a feature you already use
#       "bonus" -> nice-to-have / v2 candidate
# ---------------------------------------------------------------------------
CHECKLIST = [
    # -- MUST-HAVE --------------------------------------------------------
    ("must", "QIB subscription multiple", ["qib"]),
    ("must", "NII subscription multiple (incl. sNII/bNII)", ["nii", "hni", "non_institution", "noninstitution"]),
    ("must", "Retail subscription multiple", ["retail", "rii"]),
    ("must", "Total/overall subscription", ["subscription", "subscribed", "times"]),
    ("must", "Price band (low/high)", ["price_band", "min_price", "max_price", "lower_price", "upper_price", "band"]),
    ("must", "Lot size", ["lot_size", "lotsize", "market_lot", "bid_lot"]),
    ("must", "Min bid quantity", ["min_bid", "minimum_bid", "min_quantity", "min_qty"]),
    ("must", "Issue size", ["issue_size", "total_issue", "offer_size", "issue_amount"]),
    ("must", "Fresh vs OFS split", ["fresh", "ofs", "offer_for_sale", "fresh_issue"]),
    ("must", "Issue type (fresh/OFS/mixed)", ["issue_type", "offer_type"]),
    ("must", "Mainboard vs SME flag", ["issue_type", "segment", "category", "sme", "mainboard", "board"]),
    ("must", "Open date", ["open_date", "start_date", "bidding_start", "issue_start"]),
    ("must", "Close date", ["close_date", "end_date", "bidding_end", "issue_end"]),
    ("must", "Listing date", ["listing_date", "list_date"]),
    ("must", "Stable IPO id", ["ipo_id", "symbol", "instrument", "id", "isin"]),
    ("must", "ISIN", ["isin"]),
    ("must", "Listing-day open price (label)", ["listing_open", "list_open", "listing_price"]),
    ("must", "Listing-day close price", ["listing_close", "list_close"]),
    ("must", "Final issue / cut-off price", ["cut_off", "cutoff", "issue_price", "final_price"]),
    # -- STRONGLY USEFUL --------------------------------------------------
    ("useful", "Anchor investor data", ["anchor"]),
    ("useful", "Issue P/E", ["pe", "p_e", "price_earning"]),
    ("useful", "Peer P/E / peer comparison", ["peer"]),
    # -- BONUS ------------------------------------------------------------
    ("bonus", "Day-by-day / intraday subscription", ["day1", "day_1", "daywise", "day_wise", "history", "timeline"]),
    ("bonus", "Registrar", ["registrar", "rta"]),
    ("bonus", "IPO status", ["status", "state"]),
]

TIER_LABEL = {"must": "MUST-HAVE ", "useful": "USEFUL    ", "bonus": "BONUS     "}


def _request(path: str, token: str, params: dict | None = None) -> tuple[int, dict | str]:
    """GET a path; return (http_status, parsed_json_or_error_text)."""
    url = BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, _try_json(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, _try_json(body)
    except urllib.error.URLError as exc:
        return 0, f"network error: {exc.reason}"


def _try_json(body: str) -> dict | str:
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def _flatten_keys(obj, prefix: str = "") -> dict[str, object]:
    """Flatten nested dict/list into {dotted.key.path: leaf_value}."""
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, val in obj.items():
            out.update(_flatten_keys(val, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        # Collapse list indices to [] so a[0].qib and a[1].qib share one path.
        for item in obj:
            out.update(_flatten_keys(item, f"{prefix}[]"))
    else:
        out[prefix] = obj
    return out


def _match(fragments: list[str], flat: dict[str, object]) -> list[str]:
    """Return flattened key paths matching any candidate fragment.

    Short fragments without an underscore (qib, nii, ofs, pe, id, sme, rta, rii)
    must match a whole underscore-delimited token, so 'pe' does not match 'type'
    and 'id' does not match 'bid'. Longer or underscored fragments match as a
    substring of a path segment.
    """
    hits = []
    for path in flat:
        segments = re.split(r"\.|\[\]", path.lower())
        for frag in (f.lower() for f in fragments):
            strict = "_" not in frag and len(frag) < 4
            for seg in segments:
                if strict:
                    if frag in seg.split("_"):
                        hits.append(path)
                        break
                elif frag in seg:
                    hits.append(path)
                    break
            else:
                continue
            break
    return sorted(set(hits))


def _grade(record: dict) -> None:
    """Print the field-by-field grade for one IPO record (list item or detail)."""
    flat = _flatten_keys(record)
    print("\n" + "=" * 72)
    print("FIELD-BY-FIELD GRADE")
    print("=" * 72)

    tallies = {"must": [0, 0], "useful": [0, 0], "bonus": [0, 0]}  # [found, total]
    for tier, concept, fragments in CHECKLIST:
        hits = _match(fragments, flat)
        tallies[tier][1] += 1
        if hits:
            tallies[tier][0] += 1
            shown = ", ".join(hits[:4]) + (" ..." if len(hits) > 4 else "")
            mark = "OK  "
        else:
            shown = "(no matching field)"
            mark = "MISS"
        print(f"[{mark}] {TIER_LABEL[tier]} {concept:<44} -> {shown}")

    print("-" * 72)
    must_found, must_total = tallies["must"]
    print(f"Must-have:      {must_found}/{must_total}")
    print(f"Strongly-useful:{tallies['useful'][0]}/{tallies['useful'][1]}")
    print(f"Bonus:          {tallies['bonus'][0]}/{tallies['bonus'][1]}")

    # The deciding field gets its own verdict line.
    qib_present = bool(_match(["qib"], flat))
    print("-" * 72)
    if qib_present:
        print("DECIDING FIELD (QIB split): PRESENT ->  Upstox can feed the model.")
    else:
        print("DECIDING FIELD (QIB split): ABSENT  ->  category subscription split not")
        print("  detected. If only a total-subscription number is returned, Upstox")
        print("  CANNOT fully feed the model regardless of the other fields.")

    print("-" * 72)
    if qib_present and must_found >= must_total - 2:  # -2 tolerance: listing prices often fillable
        print("VERDICT: FULLY REPLACES (or nearly) — sound official source for the advisor.")
    elif qib_present:
        print("VERDICT: PARTIALLY REPLACES — QIB is there; some structural fields missing.")
    else:
        print("VERDICT: DOES NOT FIT — missing the QIB/NII/retail split the model is built on.")
    print("=" * 72)
    print("NOTE: matches are heuristic on field names. Confirm semantics in the raw dump")
    print("      below / via --raw before trusting the grade.")


def _unwrap(payload) -> list:
    """Pull the list/record out of Upstox's {status, data} envelope."""
    if isinstance(payload, dict) and "data" in payload:
        data = payload["data"]
        return data if isinstance(data, list) else [data]
    if isinstance(payload, list):
        return payload
    return [payload] if payload else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", default=os.environ.get("UPSTOX_ACCESS_TOKEN"),
                        help="Access token (default: $UPSTOX_ACCESS_TOKEN).")
    parser.add_argument("--status", help="Optional list filter, e.g. 'open', 'upcoming', 'closed', 'listed'.")
    parser.add_argument("--issue-type", dest="issue_type",
                        help="Optional list filter, e.g. 'MAINBOARD' / 'SME' (exact value per Upstox docs).")
    parser.add_argument("--ipo-id", dest="ipo_id",
                        help="Probe this IPO's detail record directly (skip auto-pick from the list).")
    parser.add_argument("--raw", action="store_true", help="Also print the full raw JSON of the probed record.")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: no token. Set UPSTOX_ACCESS_TOKEN or pass --token.", file=sys.stderr)
        return 2

    record = None

    if args.ipo_id:
        print(f"GET {DETAIL_PATH.format(ipo_id=args.ipo_id)}")
        status, payload = _request(DETAIL_PATH.format(ipo_id=args.ipo_id), args.token)
        if not _ok(status, payload):
            return 1
        records = _unwrap(payload)
        record = records[0] if records else None
    else:
        params = {"status": args.status, "issue_type": args.issue_type}
        print(f"GET {LIST_PATH}  params={ {k: v for k, v in params.items() if v} }")
        status, payload = _request(LIST_PATH, args.token, params)
        if not _ok(status, payload):
            return 1
        records = _unwrap(payload)
        print(f"\nReturned {len(records)} IPO(s).")
        for i, rec in enumerate(records[:15]):
            ident = rec.get("ipo_id") or rec.get("symbol") or rec.get("isin") or rec.get("id") if isinstance(rec, dict) else None
            name = rec.get("name") or rec.get("company") or rec.get("company_name") if isinstance(rec, dict) else None
            print(f"  [{i}] id={ident!r:>16}  name={name!r}")
        if not records:
            print("List was empty — try a different --status, or an open IPO window.")
            return 0
        # Auto-pick the first record and try to fetch its richer detail view.
        record = records[0]
        ident = record.get("ipo_id") or record.get("symbol") or record.get("isin") or record.get("id")
        if ident:
            print(f"\nAuto-probing detail for id={ident!r} ...")
            d_status, d_payload = _request(DETAIL_PATH.format(ipo_id=ident), args.token)
            if _ok(d_status, d_payload, soft=True):
                detail = _unwrap(d_payload)
                if detail:
                    record = detail[0]

    if not record:
        print("No IPO record to grade.")
        return 0

    _grade(record)
    if args.raw:
        print("\n----- RAW RECORD -----")
        print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        print("\n(Re-run with --raw to dump the full JSON record.)")
    return 0


def _ok(status: int, payload, soft: bool = False) -> bool:
    """Report HTTP outcome; return True on 2xx."""
    if 200 <= status < 300:
        return True
    where = "detail" if soft else "request"
    print(f"\n{where.upper()} FAILED — HTTP {status}", file=sys.stderr)
    if status in (401, 403):
        print("  -> Token rejected or lacks scope for the IPO endpoint.", file=sys.stderr)
        print("     If this is the Analytics token, create a full App (+ App) and use an", file=sys.stderr)
        print("     OAuth login access token instead, then re-run.", file=sys.stderr)
    if isinstance(payload, (dict, list)):
        print("  Response:", json.dumps(payload, indent=2)[:1500], file=sys.stderr)
    else:
        print("  Response:", str(payload)[:1000], file=sys.stderr)
    return False


if __name__ == "__main__":
    raise SystemExit(main())
