"""Side-by-side HTML report renderer.

The template is embedded as a string constant rather than a separate file so the
package stays single-import.  Paths are relative to the session directory so
the HTML stays portable (matches scripts/recolor_session.py's posture).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from jinja2 import Template

from alpr.eval.metrics import (
    SKIP_VALUES,
    cross_pipeline_agreement,
    per_pipeline_detection_rate,
    per_pipeline_read_rate,
)


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ALPR comparison — {{ session_name }}</title>
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }
  h1 { margin: 0 0 4px; font-size: 18px; }
  .meta { color: #999; font-size: 12px; margin-bottom: 12px; }
  .summary { display: flex; gap: 24px; padding: 12px; background: #1c1c1c; border-radius: 6px; margin-bottom: 16px; flex-wrap: wrap; }
  .summary .stat { font-size: 13px; }
  .summary .stat b { display: block; font-size: 20px; color: #6cf; }
  .filters { margin-bottom: 12px; font-size: 13px; }
  .filters label { margin-right: 14px; cursor: pointer; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 6px 8px; border-bottom: 1px solid #222; vertical-align: middle; }
  th { text-align: left; background: #1c1c1c; position: sticky; top: 0; }
  td.thumb img { width: 200px; max-height: 120px; object-fit: contain; background: #000; }
  td.crop img { max-width: 180px; max-height: 60px; object-fit: contain; background: #000; }
  td.plate { font-family: ui-monospace, Consolas, monospace; font-size: 16px; letter-spacing: 1px; }
  .conf { color: #888; font-size: 11px; }
  .row.disagreement td.plate { color: #fc6; }
  .row.match td.plate { color: #6f9; }
  .row.no-read td.plate { color: #666; font-style: italic; }
  .badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin-left: 4px; }
  .badge.label-ok { background: #1d3; color: #000; }
  .badge.label-bad { background: #d33; color: #fff; }
  .badge.label-pending { background: #555; color: #ccc; }
  .pipeline-header { color: #6cf; }
  a { color: #6cf; }
</style>
</head>
<body>
<h1>ALPR comparison — {{ session_name }}</h1>
<div class="meta">
  {{ total_snaps }} snaps across {{ pipelines|length }} pipeline{{ 's' if pipelines|length != 1 else '' }}
  · {{ labeled_n }} labeled · generated {{ generated_at }}
</div>

<div class="summary">
  {% for p in pipelines %}
  <div class="stat">
    <b>{{ "%.0f"|format(detection_rate[p] * 100) }}%</b>
    {{ p }} detection rate
  </div>
  <div class="stat">
    <b>{{ "%.0f"|format(read_rate[p] * 100) }}%</b>
    {{ p }} read rate
  </div>
  {% endfor %}
  {% for k, v in agreement.items() %}
    {% if k.endswith('_exact') %}
  <div class="stat">
    <b>{{ "%.0f"|format(v * 100) }}%</b>
    {{ k|replace('_exact', '') }} exact agree (n={{ agreement[k|replace('_exact', '_overlap_n')] }})
  </div>
    {% endif %}
  {% endfor %}
</div>

<div class="filters">
  Filter:
  <label><input type="radio" name="f" value="all" checked> all</label>
  <label><input type="radio" name="f" value="disagreement"> disagreement</label>
  <label><input type="radio" name="f" value="match"> all-agree</label>
  <label><input type="radio" name="f" value="no-read"> no read</label>
  <label><input type="radio" name="f" value="needs-label"> needs label</label>
</div>

<table id="rows">
<thead>
  <tr>
    <th>snap</th>
    <th>track</th>
    {% for p in pipelines %}<th class="pipeline-header">{{ p }} crop</th><th class="pipeline-header">{{ p }} read</th>{% endfor %}
    <th>label</th>
  </tr>
</thead>
<tbody>
{% for row in rows %}
  <tr class="row {{ row.row_class }}" data-status="{{ row.status }}" data-labeled="{{ '1' if row.label else '0' }}">
    <td class="thumb"><a href="{{ row.image_rel }}" target="_blank"><img src="{{ row.image_rel }}" loading="lazy"></a></td>
    <td>{{ row.track_id }} (snap {{ row.snap_index }})</td>
    {% for p in pipelines %}
      {% set rec = row.reads.get(p) %}
      <td class="crop">
        {% if rec and rec.crop_rel %}<img src="{{ rec.crop_rel }}" loading="lazy">{% endif %}
      </td>
      <td class="plate">
        {% if rec and rec.text %}
          {{ rec.text }}
          <span class="conf">{{ "%.2f"|format(rec.ocr_conf or 0) }}</span>
        {% else %}
          <span style="color:#555">—</span>
        {% endif %}
        {% if row.label and rec and rec.text %}
          {% if rec.text == row.label %}<span class="badge label-ok">✓</span>
          {% else %}<span class="badge label-bad">✗</span>{% endif %}
        {% endif %}
      </td>
    {% endfor %}
    <td class="plate">
      {% if row.label %}{{ row.label }}
      {% elif row.label_sentinel %}<span class="conf">{{ row.label_sentinel }}</span>
      {% else %}<span class="badge label-pending">unlabeled</span>{% endif %}
    </td>
  </tr>
{% endfor %}
</tbody>
</table>

<script>
document.querySelectorAll('input[name="f"]').forEach(function(el) {
  el.addEventListener('change', function() {
    const v = el.value;
    document.querySelectorAll('#rows tbody tr').forEach(function(tr) {
      const s = tr.dataset.status;
      const l = tr.dataset.labeled === '1';
      let show = true;
      if (v === 'disagreement') show = s === 'disagreement';
      else if (v === 'match') show = s === 'match';
      else if (v === 'no-read') show = s === 'no-read';
      else if (v === 'needs-label') show = !l && s !== 'no-read';
      tr.style.display = show ? '' : 'none';
    });
  });
});
</script>
</body>
</html>
"""


def render_html(
    session_dir: Path,
    records: list[dict],
    labels: dict[str, dict],
    generated_at: str,
) -> str:
    """Render the side-by-side comparison HTML.  All asset paths are relative to session_dir."""
    pipelines = sorted({r["pipeline"] for r in records})

    # Group records by image so each row in the table is one snap, multiple OCR cells.
    by_image: dict[str, dict[str, dict]] = defaultdict(dict)
    image_meta: dict[str, dict] = {}
    for r in records:
        img = r["image"]
        by_image[img][r["pipeline"]] = r
        image_meta.setdefault(img, {
            "track_id": r["track_id"],
            "snap_index": r["snap_index"],
            "class_name": r["class_name"],
        })

    rows = []
    for img, reads in sorted(by_image.items(), key=lambda kv: (kv[1].get(pipelines[0], {}).get("track_id", 0), kv[0])):
        meta = image_meta[img]
        label_info = labels.get(img) or {}
        label_str = label_info.get("plate")
        label_clean = label_str if label_str and label_str not in SKIP_VALUES else None
        label_sentinel = label_str if label_str in SKIP_VALUES else None

        # Determine row status: match / disagreement / no-read
        texts = [reads[p].get("ocr_text") for p in pipelines if p in reads]
        if not any(texts):
            status = "no-read"
        elif all(t == texts[0] for t in texts if t):
            status = "match"
        else:
            status = "disagreement"

        row = {
            "image_rel": img,  # the JPEG sits in the session dir alongside the HTML
            "track_id": meta["track_id"],
            "snap_index": meta["snap_index"],
            "label": label_clean,
            "label_sentinel": label_sentinel,
            "status": status,
            "row_class": status,
            "reads": {},
        }
        for p in pipelines:
            rec = reads.get(p)
            if rec is None:
                row["reads"][p] = None
                continue
            crop_path = rec.get("crop_path")
            crop_rel = _relativize(crop_path, session_dir) if crop_path else None
            row["reads"][p] = {
                "text": rec.get("ocr_text"),
                "ocr_conf": rec.get("ocr_conf"),
                "crop_rel": crop_rel,
            }
        rows.append(row)

    template = Template(_TEMPLATE, autoescape=True)
    return template.render(
        session_name=session_dir.name,
        total_snaps=len({r["image"] for r in records}),
        pipelines=pipelines,
        detection_rate=per_pipeline_detection_rate(records),
        read_rate=per_pipeline_read_rate(records),
        agreement=cross_pipeline_agreement(records),
        labeled_n=sum(1 for v in labels.values() if v.get("plate") not in SKIP_VALUES),
        rows=rows,
        generated_at=generated_at,
    )


def _relativize(absolute_or_rel: str, session_dir: Path) -> str:
    """Make a path relative to session_dir if possible; else return as-is."""
    p = Path(absolute_or_rel)
    try:
        if p.is_absolute():
            return str(p.relative_to(session_dir)).replace("\\", "/")
    except ValueError:
        pass
    return str(p).replace("\\", "/")
