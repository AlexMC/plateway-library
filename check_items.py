"""Check every library item a pull request adds or changes, with the same checks Plateway runs before Alex's quick look
(`check.py`, vendored from Plateway's `cad/library/check.py`).

    python check_items.py --root PR_CHECKOUT BASE_SHA     # the items changed since BASE_SHA
    python check_items.py --root . pieces/community_x/1   # these item folders

An item is a folder `<pieces|trains|furniture>/<id>/<version>/` holding `metadata.json` and one file: `.stl` for a
piece, `.glb` for a train, `.json` for furniture. Folders a change deletes are skipped. Prints one line per item and
exits 1 when any item is refused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import check

KINDS = {'pieces': 'piece', 'trains': 'train', 'furniture': 'furniture'}
SUFFIXES = {'piece': '.stl', 'train': '.glb', 'furniture': '.json'}
METADATA = 'metadata.json'


def changed_folders(root: Path, base: str) -> list[str]:
    paths = subprocess.run(['git', '-C', str(root), 'diff', '--name-only', '--no-renames', f'{base}...HEAD'],
                           check=True, capture_output=True, text=True).stdout.splitlines()
    return sorted({'/'.join(parts[:3]) for parts in (path.split('/') for path in paths)
                   if len(parts) >= 4 and parts[0] in KINDS})


def check_item(root: Path, folder: str) -> str:
    """What was checked; raises `check.Refused` with why the item is refused."""
    parts = folder.split('/')
    if len(parts) != 3 or parts[0] not in KINDS:
        raise check.Refused('Not an item folder: <pieces|trains|furniture>/<id>/<version>.')
    top, item_id, version = parts
    kind = KINDS[top]
    if not version.isdigit() or int(version) < 1:
        raise check.Refused(f'"{version}" is not a version: item folders are <id>/<whole number>/.')
    path = root / folder
    names = sorted(entry.name for entry in path.iterdir())
    if METADATA not in names:
        raise check.Refused(f'It has no {METADATA}.')
    files = [name for name in names if name != METADATA]
    if len(files) != 1 or not files[0].endswith(SUFFIXES[kind]) or not (path / files[0]).is_file():
        raise check.Refused(f'It must hold {METADATA} and one {SUFFIXES[kind]} file, not: {", ".join(files) or "nothing"}.')
    try:
        metadata = json.loads((path / METADATA).read_text(encoding='utf-8'))
    except (UnicodeDecodeError, ValueError):
        raise check.Refused(f'{METADATA} is not JSON.') from None
    if not isinstance(metadata, dict) or metadata.get('id') != item_id:
        raise check.Refused(f'{METADATA} must give the folder\'s id, "{item_id}".')
    return json.dumps(check.check(kind, (path / files[0]).read_bytes(), metadata))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('targets', nargs='+', help='a base commit, or item folders')
    args = parser.parse_args(argv)
    folders = (changed_folders(args.root, args.targets[0])
               if len(args.targets) == 1 and '/' not in args.targets[0] else [t.strip('/') for t in args.targets])
    refused = 0
    for folder in folders:
        if not (args.root / folder).is_dir():
            print(f'{folder}: deleted, skipped')
            continue
        try:
            print(f'{folder}: passed {check_item(args.root, folder)}')
        except check.Refused as reason:
            refused += 1
            print(f'{folder}: refused: {reason}')
    if not folders:
        print('No library items changed.')
    return 1 if refused else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
