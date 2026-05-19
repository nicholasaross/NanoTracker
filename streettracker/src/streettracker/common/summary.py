"""HTML summary dashboard generator.

Ported from NanoTracker's `generate_html()` (nano_tracker.py ~line 673).
Behavior preserved exactly so live deploys can swap implementations
without retraining users on a new dashboard:

  - Virtualized renderer (server emits JSON; browser renders only the
    visible viewport's worth of rows).
  - Polls ``vehicles.json`` every ``refresh_seconds`` seconds and
    re-renders in place; sort state + scroll position + active tab
    survive refresh.
  - Sibling ``vehicles.json`` written next to the HTML.
  - Tiny ``index.html`` redirector so ``http://<host>:<port>/`` lands on
    the latest summary without having to know the timestamped filename.

String concatenation is intentional (NOT ``str.format`` / f-strings inside
the JS body): CSS / JS braces collide with format placeholders. The few
runtime values that need interpolation are concatenated explicitly.
"""

from __future__ import annotations

import html as html_mod
import json
from pathlib import Path
from typing import Any

from streettracker.common.schema import TrackRecord


def generate_html(
    records: list[TrackRecord | dict[str, Any]],
    output_dir: Path,
    html_path: Path,
    session_label: str,
    meta: dict[str, Any],
    refresh_seconds: int = 15,
) -> None:
    """Render the session dashboard to ``html_path`` and the row payload
    to ``output_dir / "vehicles.json"``.

    Also writes / updates an ``index.html`` redirector so the dashboard
    is reachable at the directory root.
    """
    rows = [r.to_json_dict() if isinstance(r, TrackRecord) else r for r in records]

    class_counts: dict[str, int] = {}
    for v in rows:
        class_counts[v["class_name"]] = class_counts.get(v["class_name"], 0) + 1
    parts = [
        f"{c} {n}{'s' if c != 1 else ''}"
        for n, c in sorted(class_counts.items())
    ]
    summary_text = (
        f"{len(rows)} entr{'ies' if len(rows) != 1 else 'y'}: {', '.join(parts)}"
        if rows
        else "No detections yet"
    )

    meta_kv = " · ".join(f"{k}: {v}" for k, v in sorted(meta.items()))

    rows_payload = [
        {
            "track_id": v["track_id"],
            "class_name": v["class_name"],
            "color": v["color"],
            "time_start": v["time_start"],
            "time_start_unix": v["time_start_unix"],
            "duration_visible": v["duration_visible"],
            "direction": v["direction"],
            "speed_px_s": v["speed_px_s"],
            "lane": v["lane"],
            "avg_confidence": v["avg_confidence"],
            "asset_prefix": v.get("asset_prefix", "vehicle"),
            "main_snaps": list(v.get("main_snaps", [])),
        }
        for v in rows
    ]
    data_json = json.dumps(rows_payload, separators=(",", ":"))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vehicles.json").write_text(data_json, encoding="utf-8")

    poll_ms = int(max(refresh_seconds, 0)) * 1000
    page = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        + '<title>StreetTracker — ' + html_mod.escape(session_label) + '</title>'
        + '<style>'
        + '*{margin:0;padding:0;box-sizing:border-box}'
        + 'body{background:#1a1a1a;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,monospace;padding:24px}'
        + 'h1{font-size:1.4rem;margin-bottom:8px}'
        + '.summary{font-size:0.9rem;color:#aaa;margin-bottom:8px}'
        + '.meta{font-size:0.75rem;color:#777;margin-bottom:20px;font-family:monospace}'
        + '.thead,.row{display:grid;grid-template-columns:88px 56px 76px 76px 100px 80px 110px 96px 76px 80px;gap:10px;padding:8px 12px;align-items:center;font-size:0.85rem}'
        + '.thead{background:#252525;color:#ccc;font-weight:600;border-bottom:1px solid #444;position:sticky;top:0;z-index:2}'
        + '.thead span{cursor:pointer;user-select:none}'
        + '.thead span:hover{color:#fff}'
        + '.thead span.sk-asc::after{content:" \\25B2"}'
        + '.thead span.sk-desc::after{content:" \\25BC"}'
        + '.vp{height:85vh;overflow-y:auto;border:1px solid #333;position:relative}'
        + '.spacer{position:relative}'
        + '.row{position:absolute;left:0;right:0;border-bottom:1px solid #2a2a2a}'
        + '.row:hover{background:#252525}'
        + '.row img{max-width:80px;max-height:54px;border-radius:4px;display:block}'
        + '.no-img{width:80px;height:54px;background:#333;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#666;font-size:0.7rem}'
        + '.thumb-cell{display:flex;flex-direction:column;align-items:flex-start;gap:3px;width:80px}'
        + '.thumb-cell>a:first-child{display:block;line-height:0}'
        + '.snap-row{display:flex;flex-wrap:wrap;gap:3px;min-height:14px}'
        + '.snap-badge{display:inline-block;background:#0a84ff;color:#fff;font-size:0.6rem;font-weight:700;padding:1px 5px;border-radius:3px;text-decoration:none;line-height:1.3;min-width:12px;text-align:center}'
        + '.snap-badge:hover{background:#1f95ff}'
        + '.snap-badge.label{background:transparent;color:#888;padding:1px 0;font-weight:400}'
        + '.tabs{display:flex;gap:4px;margin-bottom:0;border-bottom:1px solid #333}'
        + '.tab{padding:8px 16px;background:#252525;color:#aaa;border:1px solid #333;border-bottom:none;border-radius:4px 4px 0 0;cursor:pointer;font-size:0.85rem;user-select:none;font-weight:600}'
        + '.tab:hover{color:#fff}'
        + '.tab.active{background:#1a1a1a;color:#e0e0e0;border-color:#444;position:relative;top:1px}'
        + '.tab .ct{color:#666;font-weight:400;margin-left:6px;font-size:0.8rem}'
        + '.tab.active .ct{color:#888}'
        + '</style></head><body>'
        + '<h1>StreetTracker Summary</h1>'
        + '<div class="summary">Session: ' + html_mod.escape(session_label)
        + ' &mdash; <span id="summary-text">' + html_mod.escape(summary_text) + '</span></div>'
        + '<div class="meta">' + html_mod.escape(meta_kv) + '</div>'
        + '<div class="tabs" id="tabs"></div>'
        + '<div class="vp" id="vp">'
        + '  <div class="thead">'
        + '    <span>Thumbnail</span>'
        + '    <span data-sk="track_id">ID</span>'
        + '    <span data-sk="class_name">Type</span>'
        + '    <span data-sk="color">Color</span>'
        + '    <span data-sk="time_start_unix" class="sk-desc">Time</span>'
        + '    <span data-sk="duration_visible">Duration</span>'
        + '    <span data-sk="direction">Direction</span>'
        + '    <span data-sk="speed_px_s">Speed</span>'
        + '    <span data-sk="lane">Lane</span>'
        + '    <span data-sk="avg_confidence">Conf</span>'
        + '  </div>'
        + '  <div class="spacer" id="spacer"><div id="rows"></div></div>'
        + '</div>'
        + '<script id="vehicles-data" type="application/json">' + data_json + '</script>'
        + '<script>'
        + 'let DATA=JSON.parse(document.getElementById("vehicles-data").textContent);'
        + 'const POLL_MS=' + str(poll_ms) + ';'
        + 'const ROW_H=88;'
        + 'const TABS=[{label:"Cars",cls:"car"},{label:"People",cls:"person"}];'
        + 'function readTabHash(){const m=location.hash.match(/tab=([^&]+)/);return m?decodeURIComponent(m[1]):null}'
        + 'let activeTab=readTabHash()||TABS[0].cls;'
        + 'let sortKey="time_start_unix",sortAsc=false;'
        + 'let sorted=[];'
        + 'const VP=document.getElementById("vp"),SPACER=document.getElementById("spacer"),ROWS=document.getElementById("rows"),TABBAR=document.getElementById("tabs");'
        + 'function esc(s){return String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",\'"\':"&quot;"}[c]))}'
        + 'function sortData(){sorted.sort((a,b)=>{let va=a[sortKey],vb=b[sortKey];if(typeof va==="number")return sortAsc?va-vb:vb-va;va=String(va);vb=String(vb);return sortAsc?va.localeCompare(vb):vb.localeCompare(va)})}'
        + 'function rowHtml(v,i){'
        +   'const t=(v.time_start||"").substr(11,8);'
        +   'const pfx=v.asset_prefix||"vehicle";'
        +   'const thumb=pfx+"_"+v.track_id+".jpg";'
        +   'const hq=pfx+"_"+v.track_id+"_hq.jpg";'
        +   'const snaps=v.main_snaps||[];'
        +   'const snapBadges=snaps.length>0'
        +     '?`<span class="snap-badge label">4K</span>`+snaps.map(n=>`<a class="snap-badge" href="${pfx}_${v.track_id}_main_${n}.jpg" target="_blank" title="main-stream snapshot ${n} of ${snaps.length}">${n}</a>`).join("")'
        +     ':"";'
        +   'return `<div class="row" style="top:${i*ROW_H}px">'
        +     '<span class="thumb-cell">'
        +       '<a href="${hq}" target="_blank" title="open full-quality crop"><img src="${thumb}" loading="lazy"></a>'
        +       '<span class="snap-row">${snapBadges}</span>'
        +     '</span>'
        +     '<span>#${v.track_id}</span>'
        +     '<span>${esc(v.class_name)}</span>'
        +     '<span>${esc(v.color)}</span>'
        +     '<span title="${esc(v.time_start)}">${t}</span>'
        +     '<span>${v.duration_visible}s</span>'
        +     '<span>${esc(v.direction)}</span>'
        +     '<span>${v.speed_px_s} px/s</span>'
        +     '<span>${esc(v.lane)}</span>'
        +     '<span>${v.avg_confidence.toFixed(3)}</span>'
        +   '</div>`'
        + '}'
        + 'function render(){'
        +   'SPACER.style.height=(sorted.length*ROW_H)+"px";'
        +   'const top=VP.scrollTop,visH=VP.clientHeight;'
        +   'const start=Math.max(0,Math.floor(top/ROW_H)-5);'
        +   'const end=Math.min(sorted.length,Math.ceil((top+visH)/ROW_H)+5);'
        +   'let h="";for(let i=start;i<end;i++)h+=rowHtml(sorted[i],i);'
        +   'ROWS.innerHTML=h;'
        + '}'
        + 'document.querySelectorAll(".thead span[data-sk]").forEach(s=>{'
        +   's.addEventListener("click",()=>{'
        +     'const k=s.dataset.sk;'
        +     'if(sortKey===k)sortAsc=!sortAsc;else{sortKey=k;sortAsc=true}'
        +     'document.querySelectorAll(".thead span").forEach(x=>x.classList.remove("sk-asc","sk-desc"));'
        +     's.classList.add(sortAsc?"sk-asc":"sk-desc");'
        +     'sortData();render();'
        +   '});'
        + '});'
        + 'VP.addEventListener("scroll",render);'
        + 'window.addEventListener("resize",render);'
        + 'function updateSummary(){'
        +   'const counts={};'
        +   'DATA.forEach(v=>{counts[v.class_name]=(counts[v.class_name]||0)+1});'
        +   'const parts=Object.keys(counts).sort().map(n=>{const c=counts[n];return c+" "+n+(c!==1?"s":"")});'
        +   'const n=DATA.length;'
        +   'const txt=n>0?(n+" detection"+(n!==1?"s":"")+": "+parts.join(", ")):"No detections yet";'
        +   'const el=document.getElementById("summary-text");if(el)el.textContent=txt;'
        + '}'
        + 'function renderTabs(){'
        +   'const counts={};'
        +   'DATA.forEach(v=>{counts[v.class_name]=(counts[v.class_name]||0)+1});'
        +   'TABBAR.innerHTML=TABS.map(t=>{'
        +     'const c=counts[t.cls]||0;'
        +     'const cls="tab"+(t.cls===activeTab?" active":"");'
        +     'return `<span class="${cls}" data-cls="${t.cls}">${t.label}<span class="ct">${c}</span></span>`;'
        +   '}).join("");'
        +   'TABBAR.querySelectorAll(".tab").forEach(el=>{'
        +     'el.addEventListener("click",()=>setTab(el.dataset.cls));'
        +   '});'
        + '}'
        + 'function setTab(cls){'
        +   'if(cls===activeTab)return;'
        +   'activeTab=cls;'
        +   'history.replaceState(null,"","#tab="+encodeURIComponent(cls));'
        +   'renderTabs();'
        +   'applyFilterSortRender(true);'
        + '}'
        + 'function applyFilterSortRender(resetScroll){'
        +   'sorted=DATA.filter(v=>v.class_name===activeTab);'
        +   'sortData();'
        +   'if(resetScroll)VP.scrollTop=0;'
        +   'render();'
        + '}'
        + 'renderTabs();applyFilterSortRender(false);updateSummary();'
        + 'if(POLL_MS>0){'
        +   'setInterval(()=>{'
        +     'fetch("vehicles.json?t="+Date.now()).then(r=>r.ok?r.json():null).then(d=>{'
        +       'if(!d)return;'
        +       'DATA=d;renderTabs();applyFilterSortRender(false);updateSummary();'
        +     '}).catch(()=>{});'
        +   '},POLL_MS);'
        + '}'
        + 'window.addEventListener("hashchange",()=>{'
        +   'const t=readTabHash();if(t&&t!==activeTab){activeTab=t;renderTabs();applyFilterSortRender(true);}'
        + '});'
        + '</script>'
        + '</body></html>'
    )
    html_path.write_text(page, encoding="utf-8")

    # Index redirector so http://<host>:<port>/ lands on the latest summary
    # without needing to know the timestamped filename.
    index_path = output_dir / "index.html"
    if (
        not index_path.exists()
        or index_path.stat().st_mtime < html_path.stat().st_mtime - 1
    ):
        index_path.write_text(
            f'<!DOCTYPE html><meta http-equiv="refresh" content="0; url={html_path.name}">',
            encoding="utf-8",
        )
