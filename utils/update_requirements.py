#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

OPS = ("==", ">=", "<=", "~=", "!=", ">", "<")
TARGET_FILENAMES = {"requirements.txt"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replace a requirement spec (name/op/version) in all requirements*.txt under --dir."
    )
    p.add_argument("--dir", required=True, help="root directory to scan")
    p.add_argument("--source", required=True, help='e.g. "transformers==4.53.1"')
    p.add_argument("--target", required=True, help='e.g. "transformers==4.53.2"')
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _dbg(v: bool, *msg):
    if v:
        print("[DBG]", *msg)


def _trunc(s: str, n: int = 160) -> str:
    return s if len(s) <= n else s[:n] + f" …(+{len(s) - n} chars)"


def find_op(s: str) -> tuple[int, str] | None:
    for i in range(len(s)):
        for op in OPS:
            if s.startswith(op, i):
                return i, op
    return None


def parse_spec(spec: str) -> tuple[str, str, str, str]:
    spec = spec.strip()
    pos = find_op(spec)
    if not pos:
        raise ValueError(f"invalid spec: {spec!r}")
    i, op = pos
    left_raw = spec[:i].strip()
    ver = spec[i + len(op) :].strip()
    if not left_raw or not ver:
        raise ValueError(f"invalid spec: {spec!r}")
    base_lower = left_raw.split("[", 1)[0].strip().lower()
    return base_lower, left_raw, op, ver


def first_req(req: str) -> tuple[str, str, str, str, str]:
    req = req.strip()
    pos = find_op(req)
    if not req or not pos:
        return req, "", "", "", ""
    i, op = pos
    left_raw = req[:i].strip()
    right = req[i + len(op) :].strip()
    if "," in right:
        ver, rest = right.split(",", 1)
        remainder = "," + rest.strip()
    else:
        ver, remainder = right, ""
    base_lower = left_raw.split("[", 1)[0].strip().lower()
    return left_raw, base_lower, op, ver.strip(), remainder


def extract_extras(left_raw: str) -> str:
    return (
        left_raw[left_raw.index("[") :]
        if "[" in left_raw and left_raw.endswith("]")
        else ""
    )


def rebuild(
    left_raw: str,
    tgt_name_raw: str,
    tgt_op: str,
    tgt_ver: str,
    remainder: str,
    env_marker: str,
) -> str:
    extras = extract_extras(left_raw)
    left2 = f"{tgt_name_raw}{extras}"
    env = f" {env_marker.strip()}" if env_marker else ""
    return f"{left2}{tgt_op}{tgt_ver}{remainder}{env}".strip()


def iter_requirement_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.casefold() in TARGET_FILENAMES:
                yield Path(dirpath) / fn


def process_file(
    path: Path,
    src: tuple[str, str, str, str],
    tgt: tuple[str, str, str, str],
    dry: bool,
    verbose: bool,
) -> tuple[bool, int]:
    src_base, _src_raw, src_op, src_ver = src
    _tgt_base, tgt_raw, tgt_op, tgt_ver = tgt

    encodings = ("utf-8", "utf-8-sig", "latin-1")
    last_err = None
    encoding = "utf-8"
    for enc in encodings:
        try:
            text = path.read_text(encoding=enc)
            encoding = enc
            break
        except Exception as e:
            last_err = e
    else:
        raise OSError(f"cannot read {path}: {last_err}")

    lines = text.splitlines()
    _dbg(verbose, f"{path}: opened encoding={encoding}, lines={len(lines)}")

    changed = False
    hits = 0
    out_lines: list[str] = []

    file_has_name = False
    file_has_exact = False

    for ln, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")

        hash_pos = line.find("#")
        comment = ""
        if hash_pos != -1:
            comment = line[hash_pos:]
            core = line[:hash_pos].rstrip()
        else:
            core = line

        if not core:
            out_lines.append(line)
            continue

        semi_pos = core.find(";")
        req_part = core if semi_pos == -1 else core[:semi_pos].rstrip()
        env_marker = "" if semi_pos == -1 else core[semi_pos:].lstrip()

        left_raw, base, op, ver, rem = first_req(req_part)
        name_match = bool(base) and (base == src_base)
        exact_match = name_match and (op == src_op and ver == src_ver)

        if name_match:
            file_has_name = True
        if exact_match:
            file_has_exact = True

        if verbose and (name_match or env_marker):
            extras = extract_extras(left_raw) if left_raw else ""
            _dbg(
                verbose,
                f"{path}:{ln}: base={base!r} op={op!r} ver={ver!r} rem={_trunc(rem)!r} "
                f"extras={extras!r} env={_trunc(env_marker)!r} comment={_trunc(comment)!r}",
            )

        if exact_match:
            new_req = rebuild(left_raw, tgt_raw, tgt_op, tgt_ver, rem, env_marker)
            new_line = new_req + (
                f" {comment}" if comment and not comment.startswith("#") else comment
            )
            if new_line != line:
                hits += 1
                changed = True
                _dbg(
                    verbose,
                    f"{path}:{ln}: REPLACE '{_trunc(line)}' -> '{_trunc(new_line)}'",
                )
            else:
                _dbg(verbose, f"{path}:{ln}: matched but no textual change")
            out_lines.append(new_line)
        else:
            if verbose and name_match and not exact_match:
                _dbg(
                    verbose,
                    f"{path}:{ln}: NAME MATCH ONLY (found '{base}{op}{ver}'), "
                    f"expected '{src_base}{src_op}{src_ver}' -> no change",
                )
            out_lines.append(line)

    if not changed:
        if not file_has_name:
            _dbg(verbose, f"{path}: SKIP (package '{src_base}' not found)")
        elif not file_has_exact:
            _dbg(
                verbose,
                f"{path}: SKIP (found '{src_base}' but no exact spec '{src_op}{src_ver}')",
            )
        print(f"[SKIP] {path} (line_hits=0)")
        return False, 0

    if not dry:
        fd, tmp_path = tempfile.mkstemp(prefix=".reqtmp.", dir=path.parent)
        _dbg(verbose, f"{path}: writing tmp={tmp_path} encoding={encoding}")
        try:
            with os.fdopen(fd, "w", encoding=encoding, newline="\n") as wf:
                wf.write("\n".join(out_lines))
                wf.write("\n")
            os.replace(tmp_path, path)
            _dbg(verbose, f"{path}: replaced atomically")
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    print(f"[{'DRY-RUN' if dry else 'UPDATED'}] {path} (line_hits={hits})")
    return True, hits


def main():
    a = parse_args()
    root = Path(a.dir).resolve()
    if not root.exists():
        raise SystemExit(f"[ERROR] Directory not found: {root}")

    src = parse_spec(a.source)
    tgt = parse_spec(a.target)

    print(f"[START] root={root}")
    _dbg(a.verbose, f"src={src} tgt={tgt}")

    scanned = 0
    updated = 0
    total_hits = 0
    for p in iter_requirement_files(root):
        scanned += 1
        ch, hits = process_file(p, src, tgt, a.dry_run, a.verbose)
        updated += bool(ch)
        total_hits += hits

    print(
        f"[END] {'DRY-RUN' if a.dry_run else 'DONE'}. scanned_files={scanned}, "
        f"updated_files={updated}, total_line_hits={total_hits}"
    )


if __name__ == "__main__":
    main()
