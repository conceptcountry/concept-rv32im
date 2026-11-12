#!/usr/bin/env python3
"""Mirror the Qwen Concept 27B artifacts from spark3 into R2.

Why this exists: models.concept.country answers
"conceptcountry/qwen-concept-27b not found" for the exact URL the research
page links. The Worker is correct and its R2 binding is correct — the bucket
is simply empty. Nothing was ever uploaded.

Transport: the Cloudflare REST API, not wrangler and not the S3 API. The
available token authenticates against /accounts/{id}/r2/... but not against
/memberships, which wrangler calls on startup, so wrangler fails with
"Authentication failed (status: 400) [code: 9106]" while plain REST PUTs
succeed. No new credential is needed, and no S3 access key has to be minted.

Scope: the SMALL files only — config, tokenizer, index, card. The weight
shards are 143 GB and live on HuggingFace, which is a CDN built for them.
The registry's job is to answer metadata queries and serve the files that
make a repo usable; it does not need to duplicate the shards to stop
returning 404. Shards can be added later without changing anything here.

Key layout: {owner}/{name}/{path}, because the Worker derives a model id
from exactly two prefix levels (modelIds() lists with delimiter '/' twice).
A key with fewer segments is invisible to /api/models — verified by probe.

Idempotent: an object whose size already matches is skipped, so a re-run
after a partial upload costs listings rather than transfers.
"""
import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

ACCOUNT = "5965ba54fc630c91269af0d4b4181889"
BUCKET = "concept-models"
API = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/r2/buckets/{BUCKET}"

SPARK = "somewheresystems@10.88.0.23"
SSH_KEY = str(pathlib.Path.home() / ".ssh/s2_dgx_ed25519")
SSH_OPTS = ["-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=no"]
REMOTE_DIST_DEFAULT = "/home/somewheresystems/wsp/dist/qwen-concept-27b"

# Everything a client needs to inspect or load the repo, minus the shards.
SMALL_SUFFIXES = (".json", ".txt", ".md", ".model", "SHA256SUMS")
MAX_SMALL = 64 * 1024 * 1024


def token() -> str:
    out = subprocess.run(
        ["security", "find-generic-password", "-s", "cf-zero-trust-token", "-w"],
        capture_output=True, text=True, check=True)
    return out.stdout.strip()


def ssh(cmd: str) -> str:
    r = subprocess.run(["ssh", "-i", SSH_KEY, *SSH_OPTS, SPARK, cmd],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"ssh failed: {r.stderr[:300]}")
    return r.stdout


def api(method: str, path: str, tok: str, data: bytes | None = None,
        ctype: str | None = None) -> dict:
    req = urllib.request.Request(f"{API}/{path}", data=data, method=method)
    req.add_header("Authorization", f"Bearer {tok}")
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"success": False, "errors": [{"message": e.read()[:200].decode(errors="replace")}]}


def existing(tok: str) -> dict[str, int]:
    """Current bucket contents as key -> size, so a re-run skips matches."""
    out: dict[str, int] = {}
    cursor = ""
    while True:
        q = "objects?per_page=1000" + (f"&cursor={urllib.parse.quote(cursor)}" if cursor else "")
        d = api("GET", q, tok)
        if not d.get("success"):
            break
        for o in d.get("result") or []:
            out[o["key"]] = o["size"]
        cursor = (d.get("result_info") or {}).get("cursor") or ""
        if not cursor:
            break
    return out


def content_type(name: str) -> str:
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".md"):
        return "text/markdown; charset=utf-8"
    return "text/plain; charset=utf-8"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", default="conceptcountry")
    ap.add_argument("--name", default="qwen-concept-27b")
    ap.add_argument("--remote-dist", default=REMOTE_DIST_DEFAULT,
                    help="the dist directory on spark3 whose small files are mirrored")
    ap.add_argument("--card", help="JSON file whose contents become {repo}/card.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    repo = f"{a.owner}/{a.name}"
    REMOTE_DIST = a.remote_dist
    tok = token()

    print(f"  listing {REMOTE_DIST} on spark3", flush=True)
    listing = ssh(f"find {REMOTE_DIST} -type f -printf '%s\\t%P\\n'")
    files = []
    for line in listing.splitlines():
        if not line.strip():
            continue
        size_s, rel = line.split("\t", 1)
        size = int(size_s)
        base = rel.rsplit("/", 1)[-1]
        if size <= MAX_SMALL and base.endswith(SMALL_SUFFIXES):
            files.append((rel, size))
    files.sort()
    print(f"  {len(files)} small files to mirror "
          f"({sum(s for _, s in files)/1e6:.1f} MB)", flush=True)

    have = existing(tok)
    print(f"  bucket currently holds {len(have)} objects", flush=True)

    # The Worker merges {repo}/card.json into its metadata response. Without
    # it the registry answers with files and no description of what they are.
    if a.card:
        card = json.loads(pathlib.Path(a.card).read_text())
    else:
        card = {
            "id": repo,
            "modelId": repo,
            "author": "Concept Country",
            "license": "apache-2.0",
            "pipeline_tag": "text-generation",
            "base_model": "Qwen/Qwen3.8-27B",
            "tags": ["weight-editing", "weight-space-programming", "qwen3.8", "concept-country"],
            "description": (
                "Qwen3.8-27B edited in weight space to reduce overthinking. "
                "Every figure in the paper was measured on these weights served at Q8_0. "
                "Formats: bf16, fp8, nvfp4a16, and GGUF Q8_0/Q4_K_M."
            ),
            "paper": "https://concept.country/research/weight-space-programming",
            "weights": "https://huggingface.co/ConceptCountry/qwen-concept-27b",
            "note": (
                "This registry serves repository metadata and the small files. "
                "The weight shards are on HuggingFace; quantizations are deployment "
                "conveniences and were not separately evaluated."
            ),
        }
    if not a.dry_run:
        d = api("PUT", f"objects/{urllib.parse.quote(repo + '/card.json', safe='')}",
                tok, data=json.dumps(card, indent=2).encode(), ctype="application/json")
        print(f"  card.json: {'ok' if d.get('success') else d.get('errors')}", flush=True)

    sent = skipped = failed = 0
    for rel, size in files:
        key = f"{repo}/{rel}"
        if have.get(key) == size:
            skipped += 1
            continue
        if a.dry_run:
            print(f"  would send {key} ({size} B)", flush=True)
            sent += 1
            continue
        body = subprocess.run(
            ["ssh", "-i", SSH_KEY, *SSH_OPTS, SPARK, f"cat {REMOTE_DIST}/{rel}"],
            capture_output=True, timeout=300).stdout
        if len(body) != size:
            print(f"  SKIP {key}: read {len(body)} B, expected {size}", flush=True)
            failed += 1
            continue
        d = api("PUT", f"objects/{urllib.parse.quote(key, safe='')}", tok,
                data=body, ctype=content_type(rel))
        if d.get("success"):
            sent += 1
            print(f"  sent {key} ({size} B, sha256 "
                  f"{hashlib.sha256(body).hexdigest()[:12]})", flush=True)
        else:
            failed += 1
            print(f"  FAIL {key}: {d.get('errors')}", flush=True)

    print(f"\n  sent {sent}, skipped {skipped}, failed {failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
