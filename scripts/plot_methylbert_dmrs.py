#!/usr/bin/env python3
"""Create lightweight visual summaries for MethylBERT/DSS DMRs.

The script uses only the Python standard library and writes SVG/HTML outputs,
so it can run on the cluster without extra plotting packages.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


HG38_CHROM_SIZES = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415,
}

CHROM_ORDER = {chrom: i for i, chrom in enumerate(HG38_CHROM_SIZES)}
COLORS = [
    "#3366cc",
    "#dc3912",
    "#ff9900",
    "#109618",
    "#990099",
    "#0099c6",
    "#dd4477",
    "#66aa00",
    "#b82e2e",
    "#316395",
]


def chrom_key(chrom: str) -> tuple[int, str]:
    return (CHROM_ORDER.get(chrom, 10_000), chrom)


def read_bed(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open() as handle:
        for idx, line in enumerate(handle):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise SystemExit(f"BED row has fewer than 3 columns: {line[:120]!r}")
            chrom, start_raw, end_raw = parts[:3]
            start = int(float(start_raw))
            end = int(float(end_raw))
            name = parts[3] if len(parts) > 3 else str(idx)
            if end <= start:
                continue
            rows.append(
                {
                    "rank": len(rows) + 1,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "name": name,
                    "length": end - start,
                }
            )
    if not rows:
        raise SystemExit(f"no DMRs found in {path}")
    return rows


def read_tsv(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            return {}
        rows = {}
        for idx, row in enumerate(reader):
            key = row.get("dmr_id") or str(idx)
            rows[str(key)] = row
        return rows


def annotate_with_tsv(bed_rows: list[dict[str, object]], tsv_rows: dict[str, dict[str, str]]) -> None:
    for idx, row in enumerate(bed_rows):
        match = tsv_rows.get(str(row["name"])) or tsv_rows.get(str(idx))
        if not match:
            continue
        for key in ("areaStat", "abs_areaStat", "diff.Methy", "meanMethy1", "meanMethy2", "pval", "fdr"):
            if key in match and match[key] not in ("", "NA"):
                try:
                    row[key] = float(match[key])
                except ValueError:
                    row[key] = match[key]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_page(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}'
        '.title{font-size:22px;font-weight:700}.axis{stroke:#555;stroke-width:1}'
        '.grid{stroke:#ddd;stroke-width:1}.label{font-size:12px}.small{font-size:10px}'
        '</style>\n'
        f"{body}\n</svg>\n"
    )


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    path.write_text(svg_page(width, height, body))


def bar_chart(
    values: list[tuple[str, float]],
    title: str,
    y_label: str,
    path: Path,
    width: int = 1000,
    height: int = 520,
) -> None:
    margin = dict(left=80, right=30, top=60, bottom=110)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    max_v = max((v for _, v in values), default=1)
    max_v = max(max_v, 1)
    bar_w = plot_w / max(len(values), 1)
    body = [f'<text x="{width/2}" y="32" text-anchor="middle" class="title">{esc(title)}</text>']
    body.append(f'<line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" class="axis"/>')
    body.append(f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" class="axis"/>')
    body.append(f'<text x="18" y="{margin["top"] + plot_h/2}" transform="rotate(-90 18 {margin["top"] + plot_h/2})" class="label">{esc(y_label)}</text>')
    for i in range(5):
        y = margin["top"] + plot_h - plot_h * i / 4
        val = max_v * i / 4
        body.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{margin["left"] + plot_w}" y2="{y:.1f}" class="grid"/>')
        body.append(f'<text x="{margin["left"] - 8}" y="{y + 4:.1f}" text-anchor="end" class="small">{val:.0f}</text>')
    for idx, (label, value) in enumerate(values):
        x = margin["left"] + idx * bar_w + 2
        h = plot_h * value / max_v
        y = margin["top"] + plot_h - h
        color = COLORS[idx % len(COLORS)]
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bar_w - 4, 1):.1f}" height="{h:.1f}" fill="{color}"/>')
        body.append(f'<text x="{x + bar_w/2 - 2:.1f}" y="{margin["top"] + plot_h + 16}" text-anchor="end" transform="rotate(-45 {x + bar_w/2 - 2:.1f} {margin["top"] + plot_h + 16})" class="small">{esc(label)}</text>')
    write_svg(path, width, height, "\n".join(body))


def histogram(values: list[float], title: str, x_label: str, path: Path, bins: int = 25) -> None:
    if not values:
        return
    lo, hi = min(values), max(values)
    if lo == hi:
        hi = lo + 1
    counts = [0] * bins
    for value in values:
        idx = min(int((value - lo) / (hi - lo) * bins), bins - 1)
        counts[idx] += 1
    labels = [f"{lo + (hi - lo) * i / bins:.0f}" for i in range(bins)]
    bar_chart(list(zip(labels, counts)), title, "DMR count", path)


def rank_line(rows: list[dict[str, object]], value_key: str, title: str, y_label: str, path: Path) -> None:
    values = [float(row[value_key]) for row in rows if isinstance(row.get(value_key), (int, float))]
    if not values:
        return
    width, height = 1000, 520
    left, right, top, bottom = 75, 35, 60, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    min_v, max_v = min(values), max(values)
    if min_v == max_v:
        max_v = min_v + 1
    points = []
    for idx, value in enumerate(values):
        x = left + plot_w * idx / max(len(values) - 1, 1)
        y = top + plot_h - plot_h * (value - min_v) / (max_v - min_v)
        points.append((x, y, value))
    body = [f'<text x="{width/2}" y="32" text-anchor="middle" class="title">{esc(title)}</text>']
    body.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    body.append(f'<text x="{width/2}" y="{height - 20}" text-anchor="middle" class="label">DMR rank</text>')
    body.append(f'<text x="18" y="{top + plot_h/2}" transform="rotate(-90 18 {top + plot_h/2})" class="label">{esc(y_label)}</text>')
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
    body.append(f'<polyline points="{poly}" fill="none" stroke="#3366cc" stroke-width="2"/>')
    for x, y, _ in points:
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#dc3912"/>')
    body.append(f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" class="small">{max_v:.2f}</text>')
    body.append(f'<text x="{left - 8}" y="{top + plot_h + 4}" text-anchor="end" class="small">{min_v:.2f}</text>')
    write_svg(path, width, height, "\n".join(body))


def genome_lanes(rows: list[dict[str, object]], path: Path) -> None:
    chroms = sorted({str(row["chrom"]) for row in rows}, key=chrom_key)
    width = 1200
    row_h = 24
    top = 58
    left = 80
    right = 40
    height = top + row_h * len(chroms) + 65
    plot_w = width - left - right
    max_size = max(HG38_CHROM_SIZES.get(chrom, max(int(r["end"]) for r in rows if r["chrom"] == chrom)) for chrom in chroms)
    body = [f'<text x="{width/2}" y="32" text-anchor="middle" class="title">Top DMR genomic distribution</text>']
    for idx, chrom in enumerate(chroms):
        y = top + idx * row_h
        chrom_size = HG38_CHROM_SIZES.get(chrom, max(int(r["end"]) for r in rows if r["chrom"] == chrom))
        lane_w = plot_w * chrom_size / max_size
        body.append(f'<text x="{left - 10}" y="{y + 5}" text-anchor="end" class="label">{esc(chrom)}</text>')
        body.append(f'<line x1="{left}" y1="{y}" x2="{left + lane_w:.1f}" y2="{y}" stroke="#bbb" stroke-width="6" stroke-linecap="round"/>')
        chrom_rows = [row for row in rows if row["chrom"] == chrom]
        for row in chrom_rows:
            x1 = left + lane_w * int(row["start"]) / chrom_size
            x2 = left + lane_w * int(row["end"]) / chrom_size
            w = max(x2 - x1, 2)
            body.append(f'<rect x="{x1:.1f}" y="{y - 7}" width="{w:.1f}" height="14" fill="#d62728" opacity="0.8"><title>{esc(row["name"])} {esc(chrom)}:{row["start"]}-{row["end"]}</title></rect>')
    body.append(f'<text x="{left}" y="{height - 26}" class="small">Chromosomes are scaled by hg38 chromosome length; red intervals are selected DMRs.</text>')
    write_svg(path, width, height, "\n".join(body))


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    chroms = sorted({str(row["chrom"]) for row in rows}, key=chrom_key)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["chrom", "n_dmrs", "total_bp", "median_bp", "min_bp", "max_bp"])
        for chrom in chroms:
            lengths = sorted(int(row["length"]) for row in rows if row["chrom"] == chrom)
            n = len(lengths)
            median = lengths[n // 2] if n % 2 else (lengths[n // 2 - 1] + lengths[n // 2]) / 2
            writer.writerow([chrom, n, sum(lengths), median, min(lengths), max(lengths)])


def write_annotated(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "rank",
        "chrom",
        "start",
        "end",
        "name",
        "length",
        "areaStat",
        "abs_areaStat",
        "diff.Methy",
        "meanMethy1",
        "meanMethy2",
        "pval",
        "fdr",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_index(output_dir: Path, prefix: str, files: list[Path], bed: Path, tsv: Path | None) -> None:
    rel_files = [path.name for path in files]
    body = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{esc(prefix)} DMR plots</title>",
        "<style>body{font-family:Arial,Helvetica,sans-serif;margin:24px;max-width:1280px}"
        "h1{font-size:24px}figure{margin:28px 0}img{max-width:100%;border:1px solid #ddd}</style>",
        "</head><body>",
        f"<h1>{esc(prefix)} DMR plots</h1>",
        f"<p>BED: <code>{esc(bed)}</code></p>",
    ]
    if tsv:
        body.append(f"<p>DSS TSV: <code>{esc(tsv)}</code></p>")
    for name in rel_files:
        if name.endswith(".svg"):
            body.append(f"<figure><h2>{esc(name)}</h2><img src='{esc(name)}' alt='{esc(name)}'></figure>")
        else:
            body.append(f"<p><a href='{esc(name)}'>{esc(name)}</a></p>")
    body.append("</body></html>\n")
    (output_dir / f"{prefix}_overview.html").write_text("\n".join(body))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bed", required=True, help="Top DMR BED file")
    parser.add_argument("--dmr-tsv", help="Optional DSS DMR TSV for scores/effect sizes")
    parser.add_argument("--output-dir", required=True, help="Output directory for plots")
    parser.add_argument("--prefix", default="methylbert_dmrs")
    args = parser.parse_args()

    bed = Path(args.bed)
    tsv = Path(args.dmr_tsv) if args.dmr_tsv else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_bed(bed)
    annotate_with_tsv(rows, read_tsv(tsv))

    chroms = sorted({str(row["chrom"]) for row in rows}, key=chrom_key)
    counts = [(chrom, sum(1 for row in rows if row["chrom"] == chrom)) for chrom in chroms]
    total_bp = [(chrom, sum(int(row["length"]) for row in rows if row["chrom"] == chrom)) for chrom in chroms]
    lengths = [int(row["length"]) for row in rows]

    outputs = [
        output_dir / f"{args.prefix}_genome_distribution.svg",
        output_dir / f"{args.prefix}_counts_by_chrom.svg",
        output_dir / f"{args.prefix}_bp_by_chrom.svg",
        output_dir / f"{args.prefix}_length_histogram.svg",
        output_dir / f"{args.prefix}_length_by_rank.svg",
    ]

    genome_lanes(rows, outputs[0])
    bar_chart(counts, "Top DMR count by chromosome", "DMR count", outputs[1])
    bar_chart(total_bp, "Total selected DMR span by chromosome", "bp", outputs[2])
    histogram([math.log10(v) for v in lengths], "DMR length distribution", "log10(bp)", outputs[3])
    rank_line(rows, "length", "DMR length by rank", "bp", outputs[4])

    if any("abs_areaStat" in row for row in rows):
        score_path = output_dir / f"{args.prefix}_abs_areaStat_by_rank.svg"
        rank_line(rows, "abs_areaStat", "DSS absolute areaStat by rank", "abs(areaStat)", score_path)
        outputs.append(score_path)
    elif any("areaStat" in row for row in rows):
        score_path = output_dir / f"{args.prefix}_areaStat_by_rank.svg"
        rank_line(rows, "areaStat", "DSS areaStat by rank", "areaStat", score_path)
        outputs.append(score_path)

    if any("diff.Methy" in row for row in rows):
        diff_path = output_dir / f"{args.prefix}_diff_methy_by_rank.svg"
        rank_line(rows, "diff.Methy", "DSS methylation difference by rank", "diff.Methy", diff_path)
        outputs.append(diff_path)

    summary_path = output_dir / f"{args.prefix}_summary.tsv"
    annotated_path = output_dir / f"{args.prefix}_annotated.tsv"
    write_summary(rows, summary_path)
    write_annotated(rows, annotated_path)
    outputs.extend([summary_path, annotated_path])
    write_index(output_dir, args.prefix, outputs, bed, tsv)

    print(f"wrote {len(rows)} DMR plots and tables to {output_dir}")
    print(output_dir / f"{args.prefix}_overview.html")


if __name__ == "__main__":
    main()
