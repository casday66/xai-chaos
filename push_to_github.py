"""
Create GitHub repo and push project.

Usage (Terminal):
    export GH_TOKEN=your_personal_access_token
    cd xai-chaos
    python push_to_github.py
"""
import os
import json
import subprocess
import urllib.request
import urllib.error

REPO_NAME = "xai-chaos"
REPO_DESC = "Hybrid Physics-AI for Chaotic Systems with Explainable AI"

TOKEN = os.environ.get("GH_TOKEN")
if not TOKEN:
    raise SystemExit("ERROR: GH_TOKEN environment variable not set.\n"
                     "Run:  export GH_TOKEN=<your_token>")


def api(method, endpoint, payload=None):
    data = json.dumps(payload).encode() if payload else None
    req  = urllib.request.Request(
        f"https://api.github.com{endpoint}",
        data=data, method=method,
        headers={
            "Authorization": f"token {TOKEN}",
            "Accept":        "application/vnd.github.v3+json",
            "User-Agent":    "xai-chaos-push",
            "Content-Type":  "application/json",
        }
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode())


def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    status = "✓" if r.returncode == 0 else "✗"
    print(f"  {status}  {cmd}")
    if r.returncode != 0 and r.stderr:
        print(f"      {r.stderr.strip()}")
    return r.returncode == 0


# 1. Authenticate
user     = api("GET", "/user")
username = user["login"]
print(f"Authenticated as: {username}\n")

# 2. Create repo (skip if exists)
try:
    api("POST", "/user/repos", {
        "name": REPO_NAME, "description": REPO_DESC,
        "private": False, "auto_init": False
    })
    print(f"Created: github.com/{username}/{REPO_NAME}\n")
except urllib.error.HTTPError as e:
    if e.code == 422:
        print(f"Repo already exists — pushing to existing.\n")
    else:
        raise

# 3. Git init + push
remote = f"https://{TOKEN}@github.com/{username}/{REPO_NAME}.git"
steps  = [
    "git init",
    "git checkout -b main",
    'git config user.email "bot@xai-chaos.local"',
    'git config user.name "xai-chaos bot"',
    "git add .",
    'git commit -m "feat: initial research version — hybrid physics-AI Lorenz + XAI"',
    f'git remote add origin "{remote}" 2>/dev/null || git remote set-url origin "{remote}"',
    "git push -u origin main --force",
]
for step in steps:
    run(step)

print(f"\n✓ Done → https://github.com/{username}/{REPO_NAME}")
