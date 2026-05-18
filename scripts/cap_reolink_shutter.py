"""One-shot: cap the Reolink max-shutter so moving vehicles don't motion-blur.

Runs from any machine on the camera's LAN (dev box, Nano, etc.):

    python cap_reolink_shutter.py                    # default: cap at 10 ms (1/100 s)
    python cap_reolink_shutter.py --max 16           # 1/60 s  (more light, more blur risk)
    python cap_reolink_shutter.py --max 5            # 1/200 s (less blur, much more noise)
    python cap_reolink_shutter.py --max 125          # revert to factory default
    python cap_reolink_shutter.py --ip 192.168.1.72  # override the camera IP
    python cap_reolink_shutter.py --get              # just print current ISP, change nothing

Camera IP comes from the first source available:
  1. --ip CLI flag
  2. $REOLINK_IP env var
  3. ./camera_config.json (the local example/config)
  4. /home/claude/NanoTracker/camera_config.json (Nano default)

Admin password comes from the first source available:
  1. $REOLINK_ADMIN_PW env var (use this for non-interactive / scripted runs)
  2. getpass prompt (silent, no terminal echo, no shell history)
"""

from __future__ import print_function

import argparse
import getpass
import json
import os
import sys
import urllib.parse
import urllib.request


def resolve_ip(cli_ip):
    if cli_ip:
        return cli_ip
    if os.environ.get("REOLINK_IP"):
        return os.environ["REOLINK_IP"]
    for path in ("./camera_config.json", "/home/claude/NanoTracker/camera_config.json"):
        if os.path.exists(path):
            try:
                return json.load(open(path))["camera"]["ip"]
            except Exception:
                pass
    sys.exit("Could not resolve camera IP: pass --ip, set $REOLINK_IP, or place camera_config.json")


def resolve_password():
    if os.environ.get("REOLINK_ADMIN_PW"):
        return os.environ["REOLINK_ADMIN_PW"]
    return getpass.getpass("Reolink ADMIN password: ")


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--max", type=int, default=10, help="New shutter.max value in ms (default 10 = 1/100 s)")
ap.add_argument("--ip", default=None, help="Camera LAN IP (overrides config / env)")
ap.add_argument("--get", action="store_true", help="Just print current ISP, do not change anything")
args = ap.parse_args()

NEW_MAX = args.max
ip = resolve_ip(args.ip)
user = "admin"
pw = resolve_password()
print("(diag) admin password length received: %d chars" % len(pw))

user_q = urllib.parse.quote(user, safe="")
pw_q = urllib.parse.quote(pw, safe="")

base = "http://%s/cgi-bin/api.cgi" % ip


def login_for_token():
    """Reolink token-based auth: send password in JSON POST body (no URL encoding),
    receive a short-lived token, use the token in subsequent calls.  Much more
    robust to passwords containing &, =, +, #, %, spaces, etc."""
    url = "%s?cmd=Login" % base
    payload = [{
        "cmd": "Login",
        "action": 0,
        "param": {"User": {"Version": "0", "userName": user, "password": pw}},
    }]
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=10).read())
    if resp[0].get("code") != 0:
        err = resp[0].get("error", {})
        sys.exit(
            "Login failed: detail=%s rspCode=%s   (auth_warning_info=%s)" % (
                err.get("detail"), err.get("rspCode"),
                err.get("auth_warning_info"),
            )
        )
    return resp[0]["value"]["Token"]["name"]


token = login_for_token()
print("(diag) login OK, token=%s..." % token[:6])

user_q = urllib.parse.quote(user, safe="")
pw_q = urllib.parse.quote(pw, safe="")

base = "http://%s/cgi-bin/api.cgi" % ip


def call(cmd, payload=None):
    """Token-authed call.  Always POST with JSON body — the read endpoints
    require a `param.channel` field that doesn't survive a bare GET with token.
    `payload` should be the full list-of-one dict the Reolink API expects;
    if None we build a default-shaped GET (action=1 returns all subfields)."""
    url = "%s?cmd=%s&token=%s" % (base, cmd, token)
    if payload is None:
        payload = [{"cmd": cmd, "action": 1, "param": {"channel": 0}}]
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=10).read())


got = call("GetIsp")
if got[0].get("code") != 0:
    sys.exit("GetIsp failed: %r" % got)
isp = got[0]["value"]["Isp"]
print("BEFORE: shutter=%s  gain=%s  exposure=%s" % (
    isp["shutter"], isp["gain"], isp["exposure"],
))

if args.get:
    sys.exit(0)

# Minimal-payload variant.  Some firmware silently ignores SetIsp when the full
# Isp dict is sent back (it re-validates every field, and any read-only one
# silently aborts the set).  Send only what we want to change.
minimal_isp = {
    "channel": 0,
    "shutter": {"max": NEW_MAX, "min": isp["shutter"]["min"]},
}
set_resp = call("SetIsp", [{"cmd": "SetIsp", "action": 0, "param": {"Isp": minimal_isp}}])
print("SetIsp raw response: %s" % json.dumps(set_resp))
print("SetIsp -> code=%s %s" % (
    set_resp[0].get("code"),
    set_resp[0].get("error") or "OK",
))

got2 = call("GetIsp")
isp2 = got2[0]["value"]["Isp"]
print("AFTER:  shutter=%s  gain=%s  exposure=%s" % (
    isp2["shutter"], isp2["gain"], isp2["exposure"],
))

if isp2["shutter"]["max"] == NEW_MAX:
    print("OK: max shutter now 1/%s s" % (1000 // NEW_MAX if NEW_MAX else "?"))
else:
    print("WARN: setting did not take effect; check 'error' fields above")
