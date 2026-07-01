#!/usr/bin/env python3
"""Get an Upstox OAuth access token from an App's api key + secret.

Upstox's login flow is three steps:

  1. Open an authorization URL in a browser and log in.
  2. Upstox redirects to your App's redirect URI with `?code=...` appended.
  3. Exchange that code (+ api key + secret) for an access_token.

This helper does steps 1 (prints the URL) and 3 (the exchange). It must run on
your own machine — it needs a browser and network access to api.upstox.com, both
of which are unavailable in this repo's cloud sessions.

The secret is read from $UPSTOX_API_SECRET so it never lands in your shell
history or the argument list. The api key is not secret (it's like a username).

Typical run
-----------
    export UPSTOX_API_SECRET="<your app secret>"

    # Step 1 — print the login URL, open it in a browser, log in:
    python scripts/upstox_get_token.py --api-key <KEY> --redirect-uri https://127.0.0.1

    # After login your browser lands on https://127.0.0.1/?code=XXXXX
    # (the page won't load — that's fine, just copy the code from the address bar)

    # Step 3 — exchange the code for a token:
    python scripts/upstox_get_token.py --api-key <KEY> --redirect-uri https://127.0.0.1 --code XXXXX

    # Then feed the printed token to the probe:
    export UPSTOX_ACCESS_TOKEN="<printed access_token>"
    python scripts/upstox_ipo_probe.py --raw

The `--redirect-uri` MUST exactly match what you set when creating the App.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

AUTH_DIALOG = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"


def build_login_url(api_key: str, redirect_uri: str) -> str:
    query = urllib.parse.urlencode(
        {"response_type": "code", "client_id": api_key, "redirect_uri": redirect_uri}
    )
    return f"{AUTH_DIALOG}?{query}"


def exchange(api_key: str, secret: str, redirect_uri: str, code: str) -> tuple[int, dict | str]:
    form = urllib.parse.urlencode(
        {
            "code": code,
            "client_id": api_key,
            "client_secret": secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
    ).encode()
    req = urllib.request.Request(TOKEN_URL, data=form, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, _json(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _json(exc.read())
    except urllib.error.URLError as exc:
        return 0, f"network error: {exc.reason}"


def _json(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return raw.decode("utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-key", required=True, help="App API key (client_id). Not secret.")
    parser.add_argument("--redirect-uri", required=True, help="Must exactly match the App's configured redirect URI.")
    parser.add_argument("--code", help="Authorization code from the redirect URL. Omit to just print the login URL.")
    parser.add_argument("--secret", default=os.environ.get("UPSTOX_API_SECRET"),
                        help="App secret (default: $UPSTOX_API_SECRET). Prefer the env var over this flag.")
    args = parser.parse_args()

    if not args.code:
        print("Step 1 — open this URL in a browser, log in, then copy the ?code=... from the redirect:\n")
        print("  " + build_login_url(args.api_key, args.redirect_uri))
        print("\nThen re-run this command adding:  --code <the_code_value>")
        return 0

    if not args.secret:
        print("ERROR: no secret. Set UPSTOX_API_SECRET (preferred) or pass --secret.", file=sys.stderr)
        return 2

    status, payload = exchange(args.api_key, args.secret, args.redirect_uri, args.code)
    if 200 <= status < 300 and isinstance(payload, dict):
        token = payload.get("access_token")
        if token:
            print("SUCCESS. access_token:\n")
            print(token)
            print("\nNext:")
            print('  export UPSTOX_ACCESS_TOKEN="' + "<the token above>" + '"')
            print("  python scripts/upstox_ipo_probe.py --raw")
            return 0

    print(f"EXCHANGE FAILED — HTTP {status}", file=sys.stderr)
    print(json.dumps(payload, indent=2)[:1500] if isinstance(payload, (dict, list)) else str(payload)[:1000],
          file=sys.stderr)
    if status == 400:
        print("\nCommon causes: code already used (each code is one-shot — get a fresh one),", file=sys.stderr)
        print("redirect_uri not matching the App exactly, or wrong/expired secret.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
