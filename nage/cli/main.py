"""
nage CLI — command-line interface for Nage Platform.

Usage:
  nage login
  nage varve list
  nage varve create --name "my-domain" --layer CHI --file data.pdf
  nage varve push <varve_id>
  nage varve test <varve_id> --query "test question"
  nage varve health <varve_id>
  nage agent list
  nage agent run <agent_id>
  nage key create --name "production"
  nage key list
  nage think "What is SEDIM?"
  nage health
  nage status
"""

import argparse
import json
import sys
import os

API_URL = os.environ.get("NAGE_API_URL", "https://api.models.nage.ai")
PLATFORM_URL = os.environ.get("NAGE_PLATFORM_URL", "https://platform.nage.ai")
CONFIG_PATH = os.path.expanduser("~/.nage/config.json")


def _load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def _save_config(config: dict):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def _get_key() -> str:
    config = _load_config()
    key = os.environ.get("NAGE_API_KEY") or config.get("api_key", "")
    if not key:
        print("Error: No API key. Run 'nage login' or set NAGE_API_KEY env var.")
        sys.exit(1)
    return key


def _request(method: str, url: str, data: dict = None, files: dict = None) -> dict:
    import urllib.request
    import urllib.error

    key = _get_key()
    headers = {"X-Nage-Key": key, "User-Agent": "nage-cli/0.1.0"}

    if files:
        # Multipart upload
        import io
        boundary = "----NageCLIBoundary"
        body = io.BytesIO()
        for name, (filename, content, content_type) in files.items():
            body.write(f"--{boundary}\r\n".encode())
            body.write(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode())
            body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
            body.write(content)
            body.write(b"\r\n")
        body.write(f"--{boundary}--\r\n".encode())
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        req = urllib.request.Request(url, data=body.getvalue(), headers=headers, method=method)
    elif data:
        headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=headers, method=method)
    else:
        req = urllib.request.Request(url, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            err = json.loads(body)
            print(f"Error ({e.code}): {err.get('message', err.get('error', body))}")
        except Exception:
            print(f"Error ({e.code}): {body[:200]}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}")
        sys.exit(1)


# ── Commands ────────────────────────────────────────────────

def cmd_login(args):
    key = input("API Key (nk_live_... or nk_test_...): ").strip()
    if not key.startswith(("nk_live_", "nk_test_")):
        print("Invalid key format. Must start with nk_live_ or nk_test_")
        return
    _save_config({"api_key": key, "api_url": API_URL})
    print(f"Saved to {CONFIG_PATH}")
    # Test connection
    os.environ["NAGE_API_KEY"] = key
    try:
        result = _request("GET", f"{API_URL}/health")
        print(f"Connected: {result.get('app', 'Nage')} v{result.get('version', '?')}")
    except Exception:
        print("Warning: Could not connect to API. Key saved anyway.")


def cmd_health(args):
    result = _request("GET", f"{API_URL}/health")
    print(json.dumps(result, indent=2))


def cmd_status(args):
    result = _request("GET", f"{API_URL}/status")
    status = result.get("status", "unknown")
    icon = "OK" if status == "ok" else "DEGRADED" if status == "degraded" else "DOWN"
    print(f"Status: {icon}")
    for svc, state in result.get("services", result.get("checks", {})).items():
        s = "ok" if state == "ok" else "ERR"
        print(f"  {svc}: {s}")


def cmd_think(args):
    result = _request("POST", f"{API_URL}/think", {
        "query": args.query,
        "platform": args.platform,
        "max_tokens": args.max_tokens,
    })
    print(result.get("response", ""))
    if args.stemma:
        stemma = result.get("stemma", {})
        weights = stemma.get("weights", {})
        if weights:
            print(f"\nSTEMMA: {json.dumps(weights, indent=2)}")
            print(f"Dominant: {stemma.get('dominant_varve', '?')}")


def cmd_varve_list(args):
    result = _request("GET", f"{PLATFORM_URL}/varves/")
    if not result:
        print("No VARVEs found.")
        return
    print(f"{'Name':<25} {'Layer':<8} {'Type':<12} {'Status':<10} {'Rank'}")
    print("-" * 70)
    for v in result:
        print(f"{v['name']:<25} {v['layer']:<8} {v['varve_type']:<12} {v['status']:<10} {v['rank']}")


def cmd_varve_create(args):
    if args.file:
        # Upload file and create Ephemeral VARVE
        with open(args.file, "rb") as f:
            content = f.read()
        filename = os.path.basename(args.file)
        result = _request("POST", f"{PLATFORM_URL}/ingest/upload-and-create-varve",
                          files={"file": (filename, content, "application/octet-stream")})
        print(f"VARVE created: {result.get('varve_name', '?')}")
        print(f"  ID: {result.get('varve_id', '?')}")
        print(f"  Chunks: {result.get('chunks', 0)}")
        print(f"  Status: {result.get('status', '?')}")
    else:
        result = _request("POST", f"{PLATFORM_URL}/varves/", {
            "name": args.name,
            "layer": args.layer,
            "rank": args.rank,
            "varve_type": args.type,
            "description": args.description or "",
        })
        v = result.get("varve", result)
        print(f"VARVE created: {v.get('name', '?')}")
        print(f"  ID: {v.get('id', '?')}")
        print(f"  Layer: {v.get('layer', '?')} | Rank: {v.get('rank', '?')}")


def cmd_varve_test(args):
    result = _request("POST", f"{API_URL}/think", {
        "query": args.query,
        "varve_hint": args.varve_id,
    })
    print(f"Response: {result.get('response', '')[:200]}")
    stemma = result.get("stemma", {})
    print(f"STEMMA: {json.dumps(stemma.get('weights', {}))}")


def cmd_varve_health(args):
    result = _request("GET", f"{PLATFORM_URL}/varves/{args.varve_id}")
    print(f"VARVE: {result.get('name', '?')}")
    print(f"  Status: {result.get('status', '?')}")
    print(f"  Distance (rho): {result.get('varve_distance', '?')}")
    print(f"  CKA: {result.get('cka_score', '?')}")
    print(f"  VARVEiq: {result.get('varveiq_score', '?')}")


def cmd_agent_list(args):
    result = _request("GET", f"{PLATFORM_URL}/agents/")
    if not result:
        print("No agents found.")
        return
    print(f"{'Name':<25} {'Actions':<8} {'Runs':<6} {'Strategy':<12} {'Active'}")
    print("-" * 65)
    for a in result:
        print(f"{a['name']:<25} {len(a.get('actions',[])):<8} {a.get('total_runs',0):<6} {a['strategy']:<12} {a['is_active']}")


def cmd_agent_run(args):
    result = _request("POST", f"{PLATFORM_URL}/agents/{args.agent_id}/run", {
        "input": args.input,
    })
    print(f"Run started: {result.get('run_id', '?')}")
    print(f"  Steps: {result.get('total_steps', 0)}")
    print(f"  Status: {result.get('status', '?')}")
    if result.get("plan"):
        for step in result["plan"]:
            print(f"    {step.get('step', '?')}. {step.get('action', '?')} → {step.get('varve', '?')}")


def cmd_key_create(args):
    result = _request("POST", f"{PLATFORM_URL}/keys/", {
        "name": args.name,
        "tier": args.tier,
    })
    print(f"API Key created:")
    print(f"  Key: {result.get('key', '?')}")
    print(f"  Tier: {result.get('tier', '?')}")
    print(f"  Note: This key is shown ONLY ONCE. Save it now.")


def cmd_key_list(args):
    result = _request("GET", f"{PLATFORM_URL}/keys/")
    if not result:
        print("No API keys found.")
        return
    print(f"{'Name':<20} {'Prefix':<20} {'Tier':<10} {'Active'}")
    print("-" * 55)
    for k in result:
        print(f"{k['name']:<20} {k['prefix']:<20} {k['tier']:<10} {k['active']}")


# ── Main ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="nage", description="Nage Platform CLI")
    sub = parser.add_subparsers(dest="command")

    # login
    sub.add_parser("login", help="Authenticate with API key")

    # health / status
    sub.add_parser("health", help="API health check")
    sub.add_parser("status", help="Platform status")

    # think
    p = sub.add_parser("think", help="Send a query")
    p.add_argument("query", help="Query text")
    p.add_argument("--platform", default="nage-8b")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--stemma", action="store_true", help="Show STEMMA attribution")

    # varve
    varve = sub.add_parser("varve", help="VARVE management")
    vsub = varve.add_subparsers(dest="varve_cmd")

    vsub.add_parser("list", help="List VARVEs")

    vc = vsub.add_parser("create", help="Create VARVE")
    vc.add_argument("--name", default="my-varve")
    vc.add_argument("--layer", default="CHI", choices=["FEHM", "MING", "CHI", "CORTEX"])
    vc.add_argument("--rank", type=int, default=16)
    vc.add_argument("--type", default="flash", choices=["ephemeral", "flash", "market"])
    vc.add_argument("--file", help="Upload file to create Ephemeral VARVE")
    vc.add_argument("--description", help="VARVE description")

    vt = vsub.add_parser("test", help="Test VARVE with query")
    vt.add_argument("varve_id")
    vt.add_argument("--query", default="Hello, test query")

    vh = vsub.add_parser("health", help="VARVE health metrics")
    vh.add_argument("varve_id")

    # agent
    agent = sub.add_parser("agent", help="Agent management")
    asub = agent.add_subparsers(dest="agent_cmd")
    asub.add_parser("list", help="List agents")
    ar = asub.add_parser("run", help="Run agent")
    ar.add_argument("agent_id")
    ar.add_argument("--input", default="")

    # key
    key = sub.add_parser("key", help="API key management")
    ksub = key.add_subparsers(dest="key_cmd")
    ksub.add_parser("list", help="List keys")
    kc = ksub.add_parser("create", help="Create key")
    kc.add_argument("--name", default="default")
    kc.add_argument("--tier", default="SURFACE")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    dispatch = {
        "login": cmd_login,
        "health": cmd_health,
        "status": cmd_status,
        "think": cmd_think,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    elif args.command == "varve":
        {"list": cmd_varve_list, "create": cmd_varve_create,
         "test": cmd_varve_test, "health": cmd_varve_health}.get(args.varve_cmd, lambda a: print("Usage: nage varve <list|create|test|health>"))(args)
    elif args.command == "agent":
        {"list": cmd_agent_list, "run": cmd_agent_run}.get(args.agent_cmd, lambda a: print("Usage: nage agent <list|run>"))(args)
    elif args.command == "key":
        {"list": cmd_key_list, "create": cmd_key_create}.get(args.key_cmd, lambda a: print("Usage: nage key <list|create>"))(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
