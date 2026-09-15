#!/usr/bin/env python3
"""Archive a large directory tree to HuggingFace as verifiable shards.

Uploading a corpus file-by-file does not work past a few tens of thousands of
files. This packs a tree into fixed-size tar archives, records a SHA-256 for
each, uploads them, and can verify the uploads by re-downloading and comparing
hashes before you delete anything local.

Shards are written to --work-dir, which should sit on the same large volume as
the source. Sharding 468 GB needs somewhere to put 468 GB.

Usage:
    # 1. Pack (resumable: existing shards with matching hashes are skipped)
    python scripts/archive_to_hf.py pack \
        --src /Volumes/16TB_Sid/DeepSafe/dataset/images/fake \
        --work-dir /Volumes/16TB_Sid/_archive/images-fake \
        --prefix images-fake

    # 2. Upload
    python scripts/archive_to_hf.py upload \
        --work-dir /Volumes/16TB_Sid/_archive/images-fake \
        --repo deepsafe/evaluation-dataset --path-in-repo raw-generators/images-fake

    # 3. Verify before deleting the source
    python scripts/archive_to_hf.py verify \
        --work-dir /Volumes/16TB_Sid/_archive/images-fake \
        --repo deepsafe/evaluation-dataset --path-in-repo raw-generators/images-fake
"""

import argparse
import hashlib
import json
import os
import pathlib
import sys
import tarfile

SHARD_BYTES = 5 * 1024**3  # 5 GiB: large enough to keep the count low, small
                           # enough that one failed upload is cheap to retry.
MANIFEST = "manifest.json"
READ_CHUNK = 1024 * 1024


def sha256(path: pathlib.Path) -> str:
    """Return the hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(READ_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def human(n: int) -> str:
    """Format a byte count."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def pack(src: pathlib.Path, work_dir: pathlib.Path, prefix: str) -> int:
    """Pack ``src`` into tar shards under ``work_dir``.

    Files are grouped in sorted order so the layout is deterministic and a
    re-run produces the same shards. Already-written shards are reused.

    Returns:
        Process exit code.
    """
    if not src.is_dir():
        print(f"error: {src} is not a directory", file=sys.stderr)
        return 1
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"scanning {src} ...", flush=True)
    entries = []
    for path in sorted(src.rglob("*")):
        if path.is_file() and not path.is_symlink():
            try:
                entries.append((path, path.stat().st_size))
            except OSError:
                continue
    total = sum(size for _, size in entries)
    print(f"  {len(entries):,} files, {human(total)}")

    shards, current, current_bytes, index = [], [], 0, 0

    def flush() -> None:
        nonlocal current, current_bytes, index
        if not current:
            return
        index += 1
        name = f"{prefix}-{index:04d}.tar"
        out = work_dir / name
        if out.exists():
            print(f"  [{index:04d}] exists, reusing {name}")
        else:
            tmp = out.with_suffix(".tar.partial")
            with tarfile.open(tmp, "w") as archive:
                for path in current:
                    archive.add(path, arcname=str(path.relative_to(src)))
            tmp.rename(out)
            print(f"  [{index:04d}] wrote {name} ({human(out.stat().st_size)})")
        shards.append({
            "name": name,
            "files": len(current),
            "bytes": out.stat().st_size,
            "sha256": sha256(out),
        })
        current, current_bytes = [], 0

    for path, size in entries:
        if current_bytes + size > SHARD_BYTES and current:
            flush()
        current.append(path)
        current_bytes += size
    flush()

    manifest = {
        "source": str(src),
        "prefix": prefix,
        "file_count": len(entries),
        "source_bytes": total,
        "shard_bytes_target": SHARD_BYTES,
        "shards": shards,
    }
    (work_dir / MANIFEST).write_text(json.dumps(manifest, indent=2))
    print(f"\npacked {len(entries):,} files into {len(shards)} shards "
          f"({human(sum(s['bytes'] for s in shards))})")
    print(f"manifest: {work_dir / MANIFEST}")
    return 0


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        sys.exit("huggingface_hub is required:  uv pip install huggingface_hub")
    return HfApi()


def upload(work_dir: pathlib.Path, repo: str, path_in_repo: str,
           repo_type: str) -> int:
    """Upload every shard plus the manifest. Skips shards already present."""
    manifest_path = work_dir / MANIFEST
    if not manifest_path.exists():
        print(f"error: no manifest at {manifest_path}; run `pack` first",
              file=sys.stderr)
        return 1
    manifest = json.loads(manifest_path.read_text())
    api = _api()

    try:
        existing = set(api.list_repo_files(repo, repo_type=repo_type))
    except Exception:
        existing = set()

    for shard in manifest["shards"]:
        remote = f"{path_in_repo}/{shard['name']}"
        if remote in existing:
            print(f"  skip (already uploaded) {shard['name']}")
            continue
        print(f"  uploading {shard['name']} ({human(shard['bytes'])}) ...",
              flush=True)
        api.upload_file(
            path_or_fileobj=str(work_dir / shard["name"]),
            path_in_repo=remote,
            repo_id=repo,
            repo_type=repo_type,
        )
    api.upload_file(
        path_or_fileobj=str(manifest_path),
        path_in_repo=f"{path_in_repo}/{MANIFEST}",
        repo_id=repo,
        repo_type=repo_type,
    )
    print("upload complete")
    return 0


def verify(work_dir: pathlib.Path, repo: str, path_in_repo: str,
           repo_type: str) -> int:
    """Re-download each shard and compare hashes against the manifest.

    Nothing local should be deleted until this passes. A shard that uploaded
    without error can still be truncated.
    """
    from huggingface_hub import hf_hub_download

    manifest = json.loads((work_dir / MANIFEST).read_text())
    failures = []
    for shard in manifest["shards"]:
        remote = f"{path_in_repo}/{shard['name']}"
        print(f"  verifying {shard['name']} ...", flush=True)
        try:
            local = hf_hub_download(repo, remote, repo_type=repo_type)
        except Exception as exc:
            failures.append((shard["name"], f"download failed: {exc}"))
            continue
        actual = sha256(pathlib.Path(local))
        if actual != shard["sha256"]:
            failures.append((shard["name"], "SHA-256 mismatch"))

    if failures:
        print("\nVERIFICATION FAILED. Do not delete the source.")
        for name, why in failures:
            print(f"  {name}: {why}")
        return 1
    print(f"\nall {len(manifest['shards'])} shards verified. "
          f"{manifest['file_count']:,} files are safely archived.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--work-dir", type=pathlib.Path, required=True)
    common.add_argument("--repo", default="deepsafe/evaluation-dataset")
    common.add_argument("--path-in-repo", required=False, default="raw-generators")
    common.add_argument("--repo-type", default="dataset",
                        choices=["dataset", "model"])

    p_pack = sub.add_parser("pack", parents=[common])
    p_pack.add_argument("--src", type=pathlib.Path, required=True)
    p_pack.add_argument("--prefix", required=True)

    sub.add_parser("upload", parents=[common])
    sub.add_parser("verify", parents=[common])

    args = parser.parse_args()
    if args.command == "pack":
        return pack(args.src, args.work_dir, args.prefix)
    if args.command == "upload":
        return upload(args.work_dir, args.repo, args.path_in_repo, args.repo_type)
    return verify(args.work_dir, args.repo, args.path_in_repo, args.repo_type)


if __name__ == "__main__":
    raise SystemExit(main())
