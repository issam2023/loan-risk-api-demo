import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 8765
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def files_exist():
    required = [
        ".dockerignore", ".env.example", "Dockerfile", "README.md", "RUN-COMMANDS.md", "requirements.txt", "benchmark.py",
        "app/__init__.py", "app/main.py", "app/schemas.py", "app/service.py",
        "data/loan_applications.csv",
        "models/loan_default_model_v1.joblib", "models/loan_default_model_v1.json",
        "notebooks/01_data_audit.ipynb", "notebooks/02_train_evaluate.ipynb", "notebooks/03_explainability.ipynb",
        "tests/conftest.py", "tests/test_units.py", "tests/test_api.py",
        "postman/loan_risk_api.postman_collection.json",
    ]
    missing = [f for f in required if not (ROOT / f).exists()]
    check("required files present", not missing, "missing: " + ", ".join(missing) if missing else "")
    reports = list((ROOT / "report").glob("*.docx")) if (ROOT / "report").exists() else []
    check("report/ holds the report as .docx", bool(reports), ", ".join(r.name for r in reports))
    check("folder name ends with -FP", ROOT.name.endswith("-FP"), ROOT.name)


def nothing_forbidden():
    forbidden = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        if path.name == ".env" or rel.startswith((".venv/", "venv/", ".pytest_cache/")):
            forbidden.append(rel)
    check("no .env, virtual environment, or pytest cache inside the folder", not forbidden, ", ".join(forbidden[:5]))
    size_mb = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file()) / 1e6
    check("folder under 50 MB", size_mb < 50, f"{size_mb:.1f} MB")


def no_todo_left():
    leftovers = []
    for rel in ["Dockerfile", ".dockerignore", "README.md", "RUN-COMMANDS.md", "requirements.txt", "app/main.py", "app/schemas.py", "app/service.py", "tests/test_units.py", "tests/test_api.py"]:
        path = ROOT / rel
        if path.exists() and re.search(r"\bTODO\b", path.read_text(encoding="utf-8", errors="ignore")):
            leftovers.append(rel)
    check("no TODO marker left in code, Dockerfile, README, or runbook", not leftovers, ", ".join(leftovers))


def requirements_pinned():
    lines = [l.strip() for l in (ROOT / "requirements.txt").read_text().splitlines() if l.strip() and not l.strip().startswith("#")]
    unpinned = [l for l in lines if not re.match(r"^[A-Za-z0-9_.\-\[\]]+==\S+$", l)]
    check("every requirement pinned with ==", not unpinned, ", ".join(unpinned[:5]))
    check("scikit-learn pinned to 1.7.2", any(l.lower().startswith("scikit-learn==1.7.2") for l in lines))


def docker_files():
    text = (ROOT / "Dockerfile").read_text()
    check("Dockerfile uses python:3.10-slim", "python:3.10-slim" in text)
    check("Dockerfile exposes 8000 and starts uvicorn app.main:app", "8000" in text and "app.main:app" in text)
    ignore = (ROOT / ".dockerignore").read_text()
    wanted = [".env", "notebooks", "tests", "data"]
    missing = [w for w in wanted if w not in ignore]
    check(".dockerignore keeps .env, notebooks, tests, data out of the image", not missing, "missing: " + ", ".join(missing) if missing else "")


def metadata_files():
    for v in (1,):
        path = ROOT / f"models/loan_default_model_v{v}.json"
        if not path.exists():
            check(f"metadata v{v} readable", False, "file missing")
            continue
        try:
            meta = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            check(f"metadata v{v} readable", False, str(exc))
            continue
        required = ["version", "trained_on", "sklearn_version", "features", "operating_threshold", "threshold_policy"]
        missing = [k for k in required if k not in meta]
        check(f"metadata v{v} has the required keys", not missing, "missing: " + ", ".join(missing) if missing else "")
        check(f"metadata v{v} declares version {v}", meta.get("version") == v, f"found {meta.get('version')}")
        t = meta.get("operating_threshold")
        check(f"metadata v{v} threshold strictly between 0 and 1", isinstance(t, (int, float)) and 0 < t < 1, str(t))


def notebooks_have_outputs():
    empty = []
    for name in ["01_data_audit", "02_train_evaluate", "03_explainability"]:
        path = ROOT / f"notebooks/{name}.ipynb"
        if not path.exists():
            empty.append(name)
            continue
        nb = json.loads(path.read_text(encoding="utf-8"))
        code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        if not code or not any(c.get("outputs") for c in code):
            empty.append(name)
    check("all three notebooks saved with outputs", not empty, ", ".join(empty))


def pytest_passes():
    proc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True)
    summary = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip()[-200:]
    passed = re.search(r"(\d+) passed", summary)
    count = int(passed.group(1)) if passed else 0
    failed = "failed" in summary or "error" in summary.lower()
    check("pytest: 10 or more tests, all passing", count >= 10 and not failed and proc.returncode == 0, summary)


def service_answers():
    env = dict(os.environ, MODEL_PATH="models/loan_default_model_v1.joblib", METADATA_PATH="models/loan_default_model_v1.json", MAX_BATCH="500")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{PORT}"

    def get(path):
        try:
            with urllib.request.urlopen(base + path, timeout=3) as r:
                return r.status, json.loads(r.read().decode()) if "json" in r.headers.get("content-type", "") else None
        except urllib.error.HTTPError as exc:
            return exc.code, None
        except (urllib.error.URLError, ConnectionError):
            return None, None

    try:
        status = None
        for _ in range(40):
            time.sleep(0.5)
            status, health = get("/health")
            if status is not None:
                break
        check("service starts with uvicorn app.main:app", status is not None)
        check("/health answers 200 with model_version 1", status == 200 and isinstance(health, dict) and health.get("model_version") == 1, f"status {status}")
        status, _ = get("/docs")
        check("/docs is served", status == 200)
        status, schema = get("/openapi.json")
        example = None
        if schema:
            body = schema.get("components", {}).get("schemas", {}).get("LoanApplication", {})
            examples = body.get("examples") or ([body["example"]] if "example" in body else [])
            example = examples[0] if examples else None
        check("LoanApplication publishes a worked example in /docs", example is not None)
        if example:
            req = urllib.request.Request(base + "/predict", data=json.dumps(example).encode(), headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=5) as r:
                    body = json.loads(r.read().decode())
                ok = r.status == 200 and all(k in body for k in ("request_id", "default_prediction", "default_probability", "model_version", "threshold_used"))
                check("POST /predict with the worked example answers with the five fields", ok, ", ".join(sorted(body)) if isinstance(body, dict) else "")
            except urllib.error.HTTPError as exc:
                check("POST /predict with the worked example answers with the five fields", False, f"status {exc.code}")
            req = urllib.request.Request(base + "/predict", data=json.dumps({**example, "collections_referral": "no"}).encode(), headers={"Content-Type": "application/json"}, method="POST")
            try:
                urllib.request.urlopen(req, timeout=5)
                check("an extra field is rejected with 422", False, "accepted")
            except urllib.error.HTTPError as exc:
                check("an extra field is rejected with 422", exc.code == 422, f"status {exc.code}")
    finally:
        proc.terminate()
        proc.wait()


def postman_collection():
    path = ROOT / "postman/loan_risk_api.postman_collection.json"
    if not path.exists():
        check("Postman collection is a v2.1 export with 4+ tested requests", False, "file missing")
        return
    try:
        col = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        check("Postman collection is a v2.1 export with 4+ tested requests", False, str(exc))
        return
    items = col.get("item", [])
    tested = [it for it in items if any(e.get("listen") == "test" and e.get("script", {}).get("exec") for e in it.get("event", []))]
    v21 = "v2.1" in col.get("info", {}).get("schema", "")
    check("Postman collection is a v2.1 export with 4+ tested requests", v21 and len(items) >= 4 and len(tested) == len(items), f"{len(items)} requests, {len(tested)} with tests, v2.1={v21}")


if __name__ == "__main__":
    print(f"Pre-flight check for {ROOT}\n")
    for step in (files_exist, nothing_forbidden, no_todo_left, requirements_pinned, docker_files, metadata_files, notebooks_have_outputs, pytest_passes, service_answers, postman_collection):
        try:
            step()
        except Exception as exc:
            check(step.__name__, False, f"check crashed: {exc}")
    import shutil
    for cache in list(ROOT.rglob("__pycache__")):
        shutil.rmtree(cache, ignore_errors=True)
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)} of {len(results)} checks passed.")
    print("Docker is not checked here: build the image, run it, and run the Postman collection against the container yourself.")
    sys.exit(1 if failed else 0)
