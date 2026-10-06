"""Build an anonymous supplementary.zip from the git-tracked files of the current branch.

    .venv/Scripts/python.exe scripts/make_anonymous_supplement.py --out supplementary.zip
    .venv/Scripts/python.exe scripts/make_anonymous_supplement.py --out /tmp/trial.zip --keep-staging

The working-tree copies of the tracked files (`git ls-files`) that belong in the supplement are staged
under a temporary directory (tempfile); every anonymising replacement below is applied to the staged
copy only (the repository is never modified); the five `pod_results/*.tgz` archives are re-packed in a
streaming pass (same member order and headers, GNU format, gzip mtime 0) with the Runpod pod ids and
the HuggingFace upload block removed; then the zip is written to --out. Afterwards every zipped file
and every tgz member is re-scanned (case-insensitively) for identifying strings and the script exits
non-zero if any hit is not on the allowlist of known third-party lines.

Every replacement is a content match that must occur exactly the expected number of times; a missing
match aborts with the rule name, so a line that drifted cannot silently skip a rule. This script and
`scripts/upload_checkpoints_hf.py` are excluded from the archive they build.
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import zipfile
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SELF = "scripts/make_anonymous_supplement.py"

# ---------------------------------------------------------------------------------------------- inclusion

INCLUDE_DIRS = ("src/", "configs/", "tests/", "scripts/", "results/final/", "notebooks/")
INCLUDE_FILES = {
    "REPORT.md", "EXPERIMENT_LOG.md", "LICENSE", "README.md", "pyproject.toml",
    "requirements.txt", "requirements-lock.txt", "Makefile", "Dockerfile",
}
EXCLUDE_FILES = {"scripts/upload_checkpoints_hf.py", SELF}
EXCLUDE_SUFFIXES = (".zip", ".pt", ".pyc")
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
N_ARCHIVES = 5  # the pod_results tarballs (four GPU pulls + laptop_runs.tgz); a different number means the inclusion rules need a review


def included(path: str) -> bool:
    """Inclusion rule for one repo-relative path (forward slashes, as `git ls-files` prints them)."""
    name = path.rsplit("/", 1)[-1]
    if path in EXCLUDE_FILES or name.startswith(".env") or path.endswith(EXCLUDE_SUFFIXES):
        return False
    if EXCLUDE_PARTS & set(path.split("/")):
        return False
    if path in INCLUDE_FILES or path.startswith(INCLUDE_DIRS):
        return True
    if path.startswith("pod_results/"):  # the archives and the top-level summaries, not the extracted trees
        return path.endswith(".tgz") or (path.count("/") == 1 and path.endswith(".md"))
    return False


def tracked_files() -> list[tuple[str, int]]:
    """(path, mode) of every regular file in the index of the current branch, in `git ls-files` order."""
    out = subprocess.run(["git", "ls-files", "-s", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    files = []
    for rec in out.decode("utf-8").split("\0"):
        if not rec:
            continue
        meta, path = rec.split("\t", 1)
        mode = int(meta.split()[0], 8)
        if mode & 0o170000 == 0o100000:  # regular file (skips symlinks and submodules)
            files.append((path, mode))
    return files


# ---------------------------------------------------------------------------------------------- text rules


@dataclass(frozen=True)
class Rule:
    """One content replacement: `pattern` (a regex; `lit()` for literal text) must match `count` times in `file`."""

    name: str
    file: str
    pattern: str
    repl: str
    count: int = 1
    flags: int = re.MULTILINE
    verify: Callable[[str], str | None] | None = None  # inspects each matched text; returns an error or None


def lit(text: str) -> str:
    return re.escape(text)


def _verify_sections_13_14(block: str) -> str | None:
    headings = re.findall(r"^## .*$", block, re.MULTILINE)
    want = ["## 13. Recommended paper framing", "## 14. Next steps"]
    if headings != want:
        return f"expected to remove exactly the headings {want}, found {headings}"
    return None


README_RESULTS_LINE = (
    "**Results, protocol and every table:** [REPORT.md](REPORT.md). **Chronological record:** "
    "[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). **Checkpoints:** 275 fold checkpoints (18 GB) will be released "
    "after the review period."
)

TEXT_RULES: list[Rule] = [
    Rule("license-copyright", "LICENSE", lit("Copyright (c) 2026 Sharoon Sharif"), "Copyright (c) 2026 Anonymous Authors"),
    Rule("pyproject-authors", "pyproject.toml", lit('authors = [{name = "Sharoon Sharif"}]'), 'authors = [{name = "Anonymous Authors"}]'),
    Rule("pyproject-version", "pyproject.toml", r'^version = "[^"\n]+"$', 'version = "0.1.0"'),
    Rule("pyproject-urls-table", "pyproject.toml", r"^\[project\.urls\]\n(?:[^\n]+\n)+\n", ""),
    Rule("readme-ci-badge", "README.md", r"^\[!\[CI\]\([^\n]*\n", ""),
    Rule("readme-doi-badge", "README.md", r"^\[!\[DOI\]\([^\n]*\n", ""),
    Rule("readme-results-line", "README.md", r"^\*\*Results, protocol and every table:\*\*[^\n]*$", README_RESULTS_LINE),
    Rule("readme-git-clone", "README.md", lit("git clone https://github.com/SharoonSharif/PathQFormer && cd PathQFormer"),
         "unzip supplementary.zip -d PathQFormer && cd PathQFormer"),
    Rule("readme-licence-citation", "README.md",
         lit("Code is released under the [MIT licence](LICENSE); see [CITATION.cff]") + r"[^\n]*?" + lit("(v0.2.1)."),
         "Code is released under the [MIT licence](LICENSE)."),
    Rule("readme-upload-row", "README.md", r"^\|[^\n]*upload_checkpoints_hf\.py[^\n]*\n", ""),
    Rule("readme-hf-token-comment", "README.md", lit("or: cp .env.example .env and fill it in"),
         "your own HuggingFace read token (the feature dataset is gated)"),
    Rule("readme-epoch-curves-path", "README.md", lit("paper/figures/epoch_curves_pooled.png"), "figures/epoch_curves_pooled.png"),
    Rule("readme-figures-parenthetical", "README.md",
         lit("(the committed ones in `paper/figures/` came from the Hub checkpoints on a pod)"),
         "(the paper figures were produced this way from the trained checkpoints)"),
    Rule("readme-ci-sentence", "README.md", lit(" CI runs the fast suite on every push and the slow suite on request."), ""),
    Rule("report-header-code", "REPORT.md", lit("**Code:** this repository, v0.3.0"), "**Code:** this supplementary archive"),
    Rule("report-header-date", "REPORT.md", r"^\*\*Date:\*\* 2026-10-05\b", "**Date:** October 2026"),
    Rule("report-volume-id", "REPORT.md", r" ?" + lit("(fjb5dlrrfp, EU-RO-1)"), ""),
    Rule("report-recomputes-ref", "REPORT.md", lit('section "Reviewer-requested recomputes"'), 'section "Additional recomputes"'),
    Rule("report-incident-4", "REPORT.md",
         lit("(4) The account balance reached zero once and Runpod deleted every pod, including one unrelated to this project;"),
         "(4) A billing interruption stopped all pods once;"),
    Rule("report-drop-sections-13-14", "REPORT.md",
         r"^## 13\. Recommended paper framing\n.*?^## 14\. Next steps\n.*?(?=^## |\Z)", "",
         flags=re.MULTILINE | re.DOTALL, verify=_verify_sections_13_14),
    Rule("log-drop-public-release-paragraph", "EXPERIMENT_LOG.md",
         r"^\*\*Repository made public; version 0\.3\.0\.\*\*[^\n]*\n(?:[^\n]+\n?)*", "", flags=re.MULTILINE),
    Rule("log-commit-0c6e0fc", "EXPERIMENT_LOG.md", lit("from commit 0c6e0fc the"), "from that point the"),
    Rule("log-commit-8298d0f", "EXPERIMENT_LOG.md", lit("fixed in 8298d0f:"), "fixed:"),
    Rule("log-volume-id", "EXPERIMENT_LOG.md", lit("network volume fjb5dlrrfp"), "the network volume"),
    Rule("log-recomputes-heading", "EXPERIMENT_LOG.md", lit("## 2026-09-22 - Reviewer-requested recomputes"),
         "## 2026-09-22 - Additional recomputes"),
    Rule("log-balance-stretch", "EXPERIMENT_LOG.md", lit("to stretch the account balance"), "to save compute"),
    Rule("log-balance-protect", "EXPERIMENT_LOG.md", lit("to protect the account balance"), "to save compute"),
    Rule("log-balance-zero", "EXPERIMENT_LOG.md", lit("when the Runpod balance hit $0 late on 2026-09-17"),
         "when the pods were stopped on 2026-09-17"),
    Rule("log-watcher-i-added", "EXPERIMENT_LOG.md", lit("the self-removal watcher I added"), "the self-removal watcher added"),
    Rule("log-balance-forced-stop", "EXPERIMENT_LOG.md", lit("before the account balance forced a stop"), "before it was stopped"),
    Rule("log-i-removed-it", "EXPERIMENT_LOG.md", lit("before I removed it"), "before it was removed"),
    Rule("log-cost-14", "EXPERIMENT_LOG.md", lit(" ($14)"), ""),
    Rule("log-cost-15.5", "EXPERIMENT_LOG.md", lit(" (~$15.5)"), ""),
    Rule("batch2-commit-ref", "scripts/pod/batch2.sh", lit("fixed in fd1d7ba)"), "fixed in a later commit)"),
    Rule("batch9-reviewer-note", "scripts/pod/batch9.sh", lit(" (2026-09-22, reviewer requests)"), ": additional evaluations"),
]


def apply_text_rules(stage: Path, rules: list[Rule]) -> None:
    """Apply every rule to the staged copy of its file; abort with the rule name on an unexpected match count."""
    by_file: dict[str, list[Rule]] = defaultdict(list)
    for r in rules:
        by_file[r.file].append(r)
    for file, file_rules in by_file.items():
        path = stage / file
        if not path.is_file():
            sys.exit(f"[anonymise] rule {file_rules[0].name}: {file} is not staged")
        text = path.read_bytes().decode("utf-8")
        for r in file_rules:
            pat = re.compile(r.pattern, r.flags)
            matches = list(pat.finditer(text))
            if len(matches) != r.count:
                sys.exit(f"[anonymise] rule {r.name}: expected {r.count} match(es) in {file}, found {len(matches)}")
            if r.verify is not None:
                for m in matches:
                    err = r.verify(m.group(0))
                    if err:
                        sys.exit(f"[anonymise] rule {r.name}: {err}")
            text = pat.sub(lambda _m, s=r.repl: s, text)
        path.write_bytes(text.encode("utf-8"))
        print(f"[anonymise] {file}: {len(file_rules)} rule(s) applied")


# ---------------------------------------------------------------------------------------------- tgz archives

POD_IDS = ["sxswkyts2qg3dc", "leiaxingd43t2i", "wmgqgyo9cuph9v", "gbhdwpvsqn9ry5",
           "v89ang0cd7pgrk", "cs84cvq0duoo8i", "lx4xgbs19pcvij", "zif2p7042jpf7h"]
POD_ID_MAP = {pid.encode(): f"pod-{i}".encode() for i, pid in enumerate(POD_IDS, 1)}
TEXT_MEMBER_SUFFIXES = (".out", ".sh", ".md")  # csv/json/yaml members are only touched if they contain an id

# logs/job_g.out of results_0923.tgz: "model card uploaded", "uploading 9 runs ...", "done", "uploading 2 runs ...", "done"
HF_UPLOAD_BLOCK = re.compile(rb"^model card uploaded\n^uploading 9 runs [^\n]*\n^done\n^uploading 2 runs [^\n]*\n^done\n", re.MULTILINE)


def _drop_hf_upload_block(data: bytes) -> bytes:
    n = len(HF_UPLOAD_BLOCK.findall(data))
    if n != 1:
        sys.exit(f"[anonymise] rule job_g-hf-upload-block: expected 1 HuggingFace upload block, found {n}")
    return HF_UPLOAD_BLOCK.sub(b"", data)


@dataclass(frozen=True)
class MemberRule:
    """A content rule for one member of one archive; `apply` must abort if the expected content is missing."""

    name: str
    archive: str
    member: str
    apply: Callable[[bytes], bytes]


MEMBER_RULES: list[MemberRule] = [
    MemberRule("job_g-hf-upload-block", "pod_results/pull_0923/results_0923.tgz", "logs/job_g.out", _drop_hf_upload_block),
]


class _GzipStreamWriter:
    """Streaming gzip writer with a fixed header (mtime 0, no file name, OS = Unix, like the original archives);
    gzip.GzipFile would stamp OS 255 and, by default, the current time."""

    def __init__(self, fileobj, level: int = 6) -> None:
        self.fileobj = fileobj
        self.comp = zlib.compressobj(level, zlib.DEFLATED, -zlib.MAX_WBITS)
        self.crc = 0
        self.size = 0
        fileobj.write(b"\x1f\x8b\x08\x00" + b"\x00\x00\x00\x00" + b"\x00" + b"\x03")

    def write(self, data: bytes) -> int:
        self.crc = zlib.crc32(data, self.crc)
        self.size += len(data)
        self.fileobj.write(self.comp.compress(data))
        return len(data)

    def close(self) -> None:
        self.fileobj.write(self.comp.flush())
        self.fileobj.write(struct.pack("<II", self.crc & 0xFFFFFFFF, self.size & 0xFFFFFFFF))


def repack_tgz(src: Path, dst: Path, archive: str, ids_seen: Counter) -> tuple[int, list[str]]:
    """Stream `src` into `dst` member by member (same order, same TarInfo headers, GNU format), replacing the pod
    ids in every member that contains one and applying the MEMBER_RULES of this archive. Returns (members, notes)."""
    rules = {r.member: r for r in MEMBER_RULES if r.archive == archive}
    fired: set[str] = set()
    notes: list[str] = []
    n = 0
    with open(dst, "wb") as raw:
        gz = _GzipStreamWriter(raw)
        with tarfile.open(src, "r|gz") as tin, tarfile.open(fileobj=gz, mode="w|", format=tarfile.GNU_FORMAT) as tout:
            for m in tin:
                n += 1
                if not m.isfile():
                    tout.addfile(m)
                    continue
                data = tin.extractfile(m).read()
                new = data
                if m.name in rules:
                    new = rules[m.name].apply(new)
                    fired.add(m.name)
                    notes.append(f"{m.name}: {rules[m.name].name} ({len(data) - len(new)} bytes removed)")
                found = [pid for pid in POD_ID_MAP if pid in new]
                if found:
                    for pid in found:
                        ids_seen[pid.decode()] += new.count(pid)
                        new = new.replace(pid, POD_ID_MAP[pid])
                    kind = "" if m.name.endswith(TEXT_MEMBER_SUFFIXES) else " (non-text member)"
                    notes.append(f"{m.name}: {len(found)} pod id(s) replaced{kind}")
                if new is not data:
                    m.size = len(new)
                tout.addfile(m, io.BytesIO(new))
        gz.close()
    missing = sorted(set(rules) - fired)
    if missing:
        sys.exit(f"[anonymise] {archive}: member(s) {missing} not found, rule(s) {[rules[k].name for k in missing]} not applied")
    return n, notes


# ---------------------------------------------------------------------------------------------- scan

SCAN_TERMS = [
    "sharif", "sharoon", "aamu", "alabama", "bulldogs", "faith comes", "sharoonsharif1", "PathQFormer-checkpoints",
    "zenodo", "2290012", "orcid", "github.com/SharoonSharif", "fjb5dlrrfp", "0c6e0fc", "8298d0f", "fd1d7ba", "v0.2.",
    "reviewer-requested", *POD_IDS, "C:\\Users", "C:/Users", "/home/", "onedrive", "hosanna",
]
SCAN_PATTERNS = [
    r"\bhf_[A-Za-z0-9]{16,}",  # HuggingFace tokens (HF_TOKEN / hf_hub_download / hf_... placeholders do not match)
    r"\brpa_[A-Za-z0-9]{16,}",  # Runpod API keys
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # e-mail addresses
]
SCAN = re.compile("|".join([lit(t) for t in SCAN_TERMS] + SCAN_PATTERNS), re.IGNORECASE)
ALLOWLIST = ("mahmoodlab",)  # third-party lines: github.com/mahmoodlab/SurvPath, huggingface.co/MahmoodLab/UNI2-h


def scan_text(label: str, data: bytes, hits: list[str], allowed: list[str]) -> None:
    """Record every line of `data` (and `label` itself) that matches SCAN as a hit or an allowlisted hit."""
    for m in SCAN.finditer(label):
        hits.append(f"{label} (name): {m.group(0)!r}")
    text = data.decode("utf-8", "replace")
    seen: dict[int, list[str]] = defaultdict(list)
    for m in SCAN.finditer(text):
        seen[text.count("\n", 0, m.start()) + 1].append(m.group(0))
    if not seen:
        return
    lines = text.split("\n")
    for ln, terms in seen.items():
        line = lines[ln - 1].strip()
        entry = f"{label}:{ln}  {sorted(set(terms))}  | {line[:160]}"
        (allowed if any(a in line.lower() for a in ALLOWLIST) else hits).append(entry)


def scan_zip(out: Path) -> tuple[list[str], list[str], int]:
    """Re-scan every file in the zip and every member of every tgz inside it. Returns (hits, allowed, n_scanned)."""
    hits: list[str] = []
    allowed: list[str] = []
    n = 0
    with zipfile.ZipFile(out) as zf:
        for name in zf.namelist():
            if name.endswith(".tgz"):
                with zf.open(name) as fobj, tarfile.open(fileobj=fobj, mode="r|gz") as tf:
                    for m in tf:
                        if m.isfile():
                            scan_text(f"{name}::{m.name}", tf.extractfile(m).read(), hits, allowed)
                            n += 1
            else:
                scan_text(name, zf.read(name), hits, allowed)
                n += 1
    return hits, allowed, n


# ---------------------------------------------------------------------------------------------- build


def write_zip(stage: Path, files: list[tuple[str, int]], out: Path) -> None:
    """Deterministic zip: git order, fixed timestamps, Unix mode bits from the git index, archives stored as-is."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as zf:
        for path, mode in files:
            zi = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            zi.create_system = 3
            zi.external_attr = (mode & 0xFFFF) << 16
            zi.compress_type = zipfile.ZIP_STORED if path.endswith(".tgz") else zipfile.ZIP_DEFLATED
            with open(stage / path, "rb") as src, zf.open(zi, "w") as dst:
                shutil.copyfileobj(src, dst)


def top_level_listing(out: Path) -> list[str]:
    counts: Counter = Counter()
    with zipfile.ZipFile(out) as zf:
        for name in zf.namelist():
            counts[name.split("/", 1)[0] + ("/" if "/" in name else "")] += 1
    return [f"{k} ({v})" if k.endswith("/") else k for k, v in sorted(counts.items())]


def build(out: Path, keep_staging: bool) -> int:
    files = [(p, mode) for p, mode in tracked_files() if included(p)]
    archives = [p for p, _ in files if p.endswith(".tgz")]
    if len(archives) != N_ARCHIVES:
        sys.exit(f"[anonymise] expected {N_ARCHIVES} pod_results archives, found {len(archives)}: {archives}")
    tmp = Path(tempfile.mkdtemp(prefix="anon_supplement_"))
    stage = tmp / "stage"
    try:
        ids_seen: Counter = Counter()
        for path, _ in files:
            src, dst = ROOT / path, stage / path
            if not src.is_file():
                sys.exit(f"[anonymise] tracked file missing from the working tree: {path}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if path.endswith(".tgz"):
                n, notes = repack_tgz(src, dst, path, ids_seen)
                print(f"[anonymise] {path}: {n} members re-packed; {len(notes)} member(s) changed")
                for note in notes:
                    print(f"    {note}")
            else:
                shutil.copyfile(src, dst)
        never = [pid for pid in POD_IDS if ids_seen[pid] == 0]
        if never:
            sys.exit(f"[anonymise] rule pod-ids: id(s) {never} occur in no archive; the id list is stale")
        apply_text_rules(stage, TEXT_RULES)
        write_zip(stage, files, out)
    finally:
        if keep_staging:
            print(f"[anonymise] staging kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"[anonymise] {len(files)} files staged, {len(TEXT_RULES)} text rules, {len(MEMBER_RULES)} member rule(s), "
          f"pod ids replaced: {dict(ids_seen)}")

    hits, allowed, n_scanned = scan_zip(out)
    print(f"[scan] {n_scanned} files/members scanned for {len(SCAN_TERMS)} terms + {len(SCAN_PATTERNS)} patterns")
    for a in allowed:
        print(f"[scan] allowlisted: {a}")
    for h in hits:
        print(f"[scan] HIT: {h}")
    size = out.stat().st_size
    print(f"[zip] {out} ({size:,} bytes = {size / 1e6:.1f} MB)")
    print("[zip] top level: " + "  ".join(top_level_listing(out)))
    if hits:
        print(f"[scan] FAILED: {len(hits)} hit(s) outside the allowlist", file=sys.stderr)
        return 1
    print(f"[scan] clean: 0 hits outside the allowlist ({len(allowed)} allowlisted line(s))")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path, help="path of the zip to write (e.g. supplementary.zip)")
    ap.add_argument("--keep-staging", action="store_true", help="keep the temporary staging directory for inspection")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sys.exit(build(args.out.resolve(), args.keep_staging))


if __name__ == "__main__":
    main()
