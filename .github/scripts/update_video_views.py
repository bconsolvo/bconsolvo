#!/usr/bin/env python3
"""Refresh YouTube view counts in the Videos table of README.md.

Runs in CI (see .github/workflows/update-video-views.yml). For every row in the
Videos table that links to a YouTube video, it looks up the current public view
count via the YouTube Data API v3 and rewrites the trailing "Views" cell. The
"All Videos Views" total row is recomputed from every numeric Views cell in the
table (so non-YouTube rows that already carry a number still count).

Rows whose Views cell is "-" (e.g. intel.com pages with no YouTube ID) are left
untouched. Set the YOUTUBE_API_KEY environment variable before running.

The script is intentionally conservative: if the API returns nothing for a
video (deleted / private / quota exhausted) the existing cell is preserved, so a
transient failure never blanks out the table.
"""
import os
import re
import sys
import json
import urllib.request
import urllib.parse

README = os.path.join(os.path.dirname(__file__), "..", "..", "README.md")
API = "https://www.googleapis.com/youtube/v3/videos"

# Matches the 11-char YouTube id in youtu.be/<id>, watch?v=<id>,
# shorts/<id>, and embed/<id> forms.
ID_PATTERNS = [
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/watch\?v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/embed/([A-Za-z0-9_-]{11})"),
]


def first_video_id(row: str):
    for pat in ID_PATTERNS:
        m = pat.search(row)
        if m:
            return m.group(1)
    return None


def fetch_view_counts(ids, api_key):
    """Return {video_id: int_views} for the given ids (batched 50 at a time)."""
    counts = {}
    for i in range(0, len(ids), 50):
        batch = ids[i : i + 50]
        params = urllib.parse.urlencode(
            {"part": "statistics", "id": ",".join(batch), "key": api_key}
        )
        req = urllib.request.Request(f"{API}?{params}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        for item in data.get("items", []):
            vc = item.get("statistics", {}).get("viewCount")
            if vc is not None:
                counts[item["id"]] = int(vc)
    return counts


def split_table_cells(row: str):
    # Leading/trailing pipes produce empty first/last entries; keep inner cells.
    return row.split("|")


def set_views_cell(row: str, value: str) -> str:
    """Replace the last non-empty cell (the Views column) with `value`."""
    # Normalise trailing whitespace / optional trailing pipe.
    stripped = row.rstrip()
    had_trailing_pipe = stripped.endswith("|")
    body = stripped[:-1] if had_trailing_pipe else stripped
    idx = body.rfind("|")
    if idx == -1:
        return row
    new = f"{body[:idx]}| {value} "
    return new + ("|" if had_trailing_pipe else "|")


def views_cell_value(row: str):
    stripped = row.rstrip()
    body = stripped[:-1] if stripped.endswith("|") else stripped
    idx = body.rfind("|")
    if idx == -1:
        return None
    return body[idx + 1 :].strip()


def main():
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        print("YOUTUBE_API_KEY is not set", file=sys.stderr)
        return 1

    raw = open(README, encoding="utf-8").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    lines = raw.split(nl)

    # Find the Videos table: total row anchors the start of data rows.
    start = None
    for i, line in enumerate(lines):
        if line.startswith("| All Videos Views"):
            start = i
            break
    if start is None:
        print("Could not find the 'All Videos Views' total row", file=sys.stderr)
        return 1

    j = start + 1
    data_idx = []
    while j < len(lines) and lines[j].startswith("|"):
        data_idx.append(j)
        j += 1

    # Collect ids across all data rows, then one batched API call.
    row_ids = {i: first_video_id(lines[i]) for i in data_idx}
    ids = sorted({v for v in row_ids.values() if v})
    counts = fetch_view_counts(ids, api_key) if ids else {}
    print(f"Fetched view counts for {len(counts)}/{len(ids)} YouTube videos")

    updated = 0
    for i in data_idx:
        vid = row_ids[i]
        if vid and vid in counts:
            new_val = f"{counts[vid]:,}"
            if views_cell_value(lines[i]) != new_val:
                lines[i] = set_views_cell(lines[i], new_val)
                updated += 1

    # Recompute total from every numeric Views cell in the data rows.
    total = 0
    for i in data_idx:
        val = views_cell_value(lines[i]) or ""
        digits = val.replace(",", "")
        if digits.isdigit():
            total += int(digits)
    lines[start] = set_views_cell(lines[start], f"{total:,}")
    print(f"Updated {updated} rows; new total = {total:,}")

    open(README, "w", encoding="utf-8", newline=nl).write(nl.join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
