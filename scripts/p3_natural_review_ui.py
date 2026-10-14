#!/usr/bin/env python3
"""Build a self-contained, tracker-blind review UI and MIDI excerpts."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
from pathlib import Path

import pretty_midi

try:
    from scripts import p3_natural_boundary_review_packet as packet_builder
    from scripts import p3_natural_event_restart_candidates as event
except ImportError:
    import p3_natural_boundary_review_packet as packet_builder
    import p3_natural_event_restart_candidates as event

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "results/p3_natural_boundary_review_packet"
MEDIA = PACKET / "media"
HTML = PACKET / "review.html"
CLIP_BEFORE_SEC = 12.0
CLIP_AFTER_SEC = 12.0
OVERVIEW_POINTS = 320


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_name(value):
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def make_clip(source, destination, center):
    original = pretty_midi.PrettyMIDI(str(source))
    start, end = max(0.0, center - CLIP_BEFORE_SEC), center + CLIP_AFTER_SEC
    output = pretty_midi.PrettyMIDI(initial_tempo=120)
    for original_instrument in original.instruments:
        instrument = pretty_midi.Instrument(program=original_instrument.program,
                                             is_drum=original_instrument.is_drum,
                                             name=original_instrument.name)
        for note in original_instrument.notes:
            if note.end < start or note.start > end:
                continue
            instrument.notes.append(pretty_midi.Note(
                velocity=note.velocity, pitch=note.pitch,
                start=max(0.0, note.start-start), end=max(0.02, min(end, note.end)-start)))
        if instrument.notes:
            output.instruments.append(instrument)
    output.write(str(destination))


def overview(notes):
    duration = max(note[1] for note in notes)
    bins = [[] for _ in range(OVERVIEW_POINTS)]
    for start, _, pitch in notes:
        index = min(OVERVIEW_POINTS-1, int(start/duration*OVERVIEW_POINTS))
        bins[index].append(pitch)
    return [{"x": round(index/(OVERVIEW_POINTS-1), 5),
             "y": None if not values else round(sum(values)/len(values), 2)}
            for index, values in enumerate(bins)]


def main():
    MEDIA.mkdir(parents=True, exist_ok=True)
    records = {record["record_time"]: record for record in packet_builder.all_records()}
    candidates = read_csv(PACKET/"candidates.csv")
    files = read_csv(PACKET/"files.csv")
    units = read_csv(PACKET/"units.csv")
    full_dir, clip_dir, score_dir = MEDIA/"full", MEDIA/"clips", MEDIA/"scores"
    for folder in (full_dir, clip_dir, score_dir): folder.mkdir(exist_ok=True)
    file_payload = []
    for row in files:
        record = records[row["record_time"]]; key = safe_name(row["record_time"])
        performance = ROOT/record["performance_path"]; score = ROOT/record["score_midi_path"]
        performance_copy, score_copy = full_dir/f"{key}.mid", score_dir/f"{key}_score.mid"
        shutil.copyfile(performance, performance_copy); shutil.copyfile(score, score_copy)
        notes = event.load_notes(performance)
        file_payload.append({**row, "performance_media": performance_copy.relative_to(PACKET).as_posix(),
            "score_media": score_copy.relative_to(PACKET).as_posix(), "duration_sec": round(max(n[1] for n in notes), 3),
            "overview": overview(notes)})
    file_map = {row["record_time"]: row for row in file_payload}
    candidate_payload = []
    for row in candidates:
        key = safe_name(row["candidate_id"]); center = float(row["time_sec"])
        source = ROOT/records[row["record_time"]]["performance_path"]; destination = clip_dir/f"{key}.mid"
        make_clip(source, destination, center)
        candidate_payload.append({**row, "time_sec": center, "clip_media": destination.relative_to(PACKET).as_posix(),
                                  "overview": file_map[row["record_time"]]["overview"]})
    data = {"candidates": candidate_payload, "files": file_payload, "units": units,
            "settings": {"clip_before_sec": CLIP_BEFORE_SEC, "clip_after_sec": CLIP_AFTER_SEC}}
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Natural P3 blind boundary review</title>
<style>
:root{{--bg:#f6f7f9;--panel:#fff;--text:#18202a;--muted:#667180;--line:#d9dee5;--active:#244d8f;--mark:#c34f3f;--ok:#287a4b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:var(--panel);border-bottom:1px solid var(--line);padding:12px 18px}}
h1{{font-size:18px;margin:0 0 8px}}button,select,input,textarea{{font:inherit}}button{{padding:7px 11px;border:1px solid var(--line);background:var(--panel);border-radius:6px}}button.active{{background:var(--active);color:white;border-color:var(--active)}}
.bar{{display:flex;gap:8px;align-items:center;flex-wrap:wrap}}main{{max-width:1050px;margin:18px auto;padding:0 18px}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:18px}}.meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin:12px 0;color:var(--muted)}}
.plot{{width:100%;height:180px;border:1px solid var(--line);background:var(--panel)}}.plot polyline{{fill:none;stroke:var(--active);stroke-width:1.5}}.plot .boundary{{stroke:var(--mark);stroke-width:2}}.plot text{{fill:var(--muted);font-size:11px}}textarea{{width:100%;min-height:80px;padding:8px;border:1px solid var(--line);border-radius:6px}}.choices{{display:flex;gap:8px;margin:12px 0}}.status{{margin-left:auto;color:var(--muted)}}a{{color:var(--active)}}.hidden{{display:none}}.notice{{padding:9px 11px;background:#fff4d6;border-left:3px solid #c28600;margin:10px 0}}.done{{color:var(--ok)}}
@media(max-width:600px){{header{{position:static}}.status{{width:100%;margin:0}}}}
</style></head><body>
<header><h1>Natural P3 blind boundary review</h1><div class="bar">
<label>Reviewer <select id="reviewer"><option value="reviewer_A">A</option><option value="reviewer_B">B</option></select></label>
<button data-tab="candidates" class="active">Candidates</button><button data-tab="files">Missed boundaries</button><button data-tab="units">Provisional units</button>
<button id="previous">Previous</button><button id="next">Next</button><button id="export">Export current CSV</button><span class="status" id="status"></span></div></header>
<main><section class="panel"><div id="content"></div></section></main>
<script>const DATA={payload};
const state={{tab:'candidates',index:0,reviewer:'reviewer_A'}};const saved=JSON.parse(localStorage.getItem('p3-natural-review')||'{{}}');
function key(row){{return row.candidate_id||row.record_time||row.unit_id}}function store(){{localStorage.setItem('p3-natural-review',JSON.stringify(saved))}}
function esc(v){{return String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]))}}
function currentRows(){{return DATA[state.tab]}}function answer(row){{const k=state.reviewer+'|'+state.tab+'|'+key(row);return saved[k]||(saved[k]={{decision:'',reason:''}})}}
function spark(points,boundary){{const pts=points.filter(p=>p.y!==null);const ys=pts.map(p=>p.y),lo=Math.min(...ys),hi=Math.max(...ys);const poly=pts.map(p=>`${{p.x*1000}},${{165-(p.y-lo)/Math.max(1,hi-lo)*145}}`).join(' ');return `<svg class="plot" viewBox="0 0 1000 180" role="img" aria-label="Performance pitch overview"><polyline points="${{poly}}"/><line class="boundary" x1="${{boundary*1000}}" x2="${{boundary*1000}}" y1="0" y2="180"/><text x="8" y="174">start</text><text x="960" y="174">end</text></svg>`}}
function render(){{const rows=currentRows();state.index=Math.max(0,Math.min(rows.length-1,state.index));const row=rows[state.index],a=answer(row);let body='';
 if(state.tab==='candidates'){{const file=DATA.files.find(x=>x.record_time===row.record_time);body=`<h2>${{esc(row.candidate_id)}}</h2><div class="meta"><span>${{esc(row.composer)}}</span><span>${{esc(row.piece)}}</span><span>${{row.time_sec.toFixed(2)}} s</span><span>${{esc(row.signals)}}</span></div>${{spark(row.overview,row.time_sec/file.duration_sec)}}<p><a href="${{esc(row.clip_media)}}">Open 24-second MIDI excerpt</a> · <a href="${{esc(file.performance_media)}}">Open full performance MIDI</a> · <a href="${{esc(file.score_media)}}">Open score MIDI</a></p><p>Local top-pitch context: <code>${{esc(row.context_top_pitches)}}</code></p>`}}
 else if(state.tab==='files'){{body=`<h2>${{esc(row.record_time)}}</h2><div class="meta"><span>${{esc(row.composer)}}</span><span>${{esc(row.piece)}}</span><span>${{Number(row.duration_sec).toFixed(1)}} s</span><span>${{row.candidate_count}} candidates</span></div>${{spark(row.overview,0)}}<p><a href="${{esc(row.performance_media)}}">Open full performance MIDI</a> · <a href="${{esc(row.score_media)}}">Open score MIDI</a></p><div class="notice">Inspect the full recording for any restart or take boundary absent from the candidate list.</div>`}}
 else{{body=`<h2>${{esc(row.unit_id)}}</h2><div class="meta"><span>${{esc(row.composer)}}</span><span>${{esc(row.piece)}}</span><span>${{Number(row.start_sec).toFixed(2)}}–${{Number(row.end_sec).toFixed(2)}} s</span><span>${{row.anchor_count}} anchors</span><span>score span ${{Number(row.score_span).toFixed(3)}}</span><span>algorithm flag: ${{esc(row.algorithm_reviewable)}}</span></div><div class="notice">Context only. Final units are rebuilt after A/B boundary consensus, then reviewed in phase 2.</div>`;document.getElementById('content').innerHTML=body;status();return}}
 const options=state.tab==='files'?['yes','no','uncertain']:['accept','reject','uncertain'];body+=`<div class="choices">${{options.map(v=>`<button class="choice ${{a.decision===v?'active':''}}" data-value="${{v}}">${{v}}</button>`).join('')}}</div><label>Reason<textarea id="reason">${{esc(a.reason)}}</textarea></label>`;document.getElementById('content').innerHTML=body;
 document.querySelectorAll('.choice').forEach(b=>b.onclick=()=>{{a.decision=b.dataset.value;store();render()}});document.getElementById('reason').oninput=e=>{{a.reason=e.target.value;store();status()}};status()}}
function status(){{const rows=currentRows();if(state.tab==='units'){{document.getElementById('status').textContent=`${{state.index+1}}/${{rows.length}} · context only`;return}}const done=rows.filter(r=>answer(r).decision&&answer(r).reason.trim()).length;document.getElementById('status').textContent=`${{state.index+1}}/${{rows.length}} · complete ${{done}}/${{rows.length}}`}}
document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>{{document.querySelectorAll('[data-tab]').forEach(x=>x.classList.remove('active'));b.classList.add('active');state.tab=b.dataset.tab;state.index=0;document.getElementById('export').disabled=state.tab==='units';render()}});document.getElementById('previous').onclick=()=>{{state.index--;render()}};document.getElementById('next').onclick=()=>{{state.index++;render()}};document.getElementById('reviewer').onchange=e=>{{state.reviewer=e.target.value;render()}};
function csvCell(v){{v=String(v??'');return /[",\\n]/.test(v)?'"'+v.replace(/"/g,'""')+'"':v}}document.getElementById('export').onclick=()=>{{const rows=currentRows();let fields;if(state.tab==='candidates')fields=['candidate_id','record_time','composer','piece','time_sec','signals','context_top_pitches','decision','reviewer','reason'];else if(state.tab==='files')fields=['record_time','composer','piece','performance_path','score_path','performance_sha256','score_sha256','candidate_count','reviewable_units','missed_boundary_check','reviewer','review_reason'];else fields=['unit_id','record_time','composer','piece','start_sec','end_sec','anchor_count','score_span','monotone_share','algorithm_reviewable','decision','reviewer','reason'];const lines=[fields.join(',')];for(const row of rows){{const a=answer(row),out={{...row}};if(state.tab==='files'){{out.missed_boundary_check=a.decision;out.review_reason=a.reason}}else{{out.decision=a.decision;out.reason=a.reason}}out.reviewer=state.reviewer;lines.push(fields.map(f=>csvCell(out[f])).join(','))}}const blob=new Blob(['\ufeff'+lines.join('\\n')],{{type:'text/csv'}}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=state.tab+'.csv';link.click();URL.revokeObjectURL(url)}};render();</script></body></html>'''
    HTML.write_text(page, encoding="utf-8")
    media_files = sorted(path for path in MEDIA.rglob("*") if path.is_file())
    aggregate = hashlib.sha256()
    for path in media_files:
        aggregate.update(path.relative_to(PACKET).as_posix().encode("utf-8"))
        aggregate.update(hashlib.sha256(path.read_bytes()).digest())
    result = {"html": str(HTML.relative_to(ROOT)).replace("\\", "/"), "candidates": len(candidate_payload),
              "files": len(file_payload), "units": len(units), "midi_clips": len(list(clip_dir.glob("*.mid"))),
              "html_sha256": hashlib.sha256(HTML.read_bytes()).hexdigest(),
              "media_files": len(media_files), "media_bytes": sum(path.stat().st_size for path in media_files),
              "media_manifest_sha256": aggregate.hexdigest(), "tracker_outcomes_included": False}
    (PACKET/"ui_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
