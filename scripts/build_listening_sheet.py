"""Build the listening session: randomized sheet CSV + self-contained HTML player.

Inputs:  data/manifests/sample_200.csv, analysis/clip_qc.csv
Outputs: reports/listening_sheet.csv   (flat sheet with empty flag columns)
         reports/listening_sheet.html  (audio player UI; marks kept in
                                        localStorage; export button -> CSV)

Order: auto-QC-flagged clips first (highest suspicion first), then the rest
in seeded random order.
"""

import html
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "sample_200.csv"
QC = ROOT / "analysis" / "clip_qc.csv"
REPORTS = ROOT / "reports"
SEED = 42

CHECKLIST = [
    "clean_single_speaker",
    "ghanaian_english",
    "background_music",
    "other_voices",
    "applause_crowd",
    "clipped_audio",
    "mic_problems",
    "transcript_matches",
    "unnatural_punctuation",
    "names_places_wrong",
    "unclean_start_end",
]

# simplified marking system: one verdict + one reason, nothing else
VERDICTS = ["keep", "borderline", "reject"]
REASONS = [
    "cut_start", "cut_end", "multiple_speakers", "noise", "music",
    "clipping", "transcript_mismatch", "non_ghanaian", "unnatural_delivery",
    "other",
]

CRITERIA_HTML = """
<ol>
  <li><b>One speaker?</b> REJECT overlap, interruptions, second voices.</li>
  <li><b>Ghanaian English?</b> pronunciation / rhythm / accent genuinely Ghanaian.</li>
  <li><b>Clean audio?</b> light background noise OK; drowning noise / echo isn't.</li>
  <li><b>Clean start?</b> first second: no missing word-half, no mid-sentence entry.</li>
  <li><b>Clean ending?</b> THE BIG ONE &mdash; 15&nbsp;s hard ceiling means many cuts.</li>
  <li><b>Transcript match?</b> read the text <i>after</i> listening. Wrong words / names / numbers matter; punctuation doesn't.</li>
  <li><b>Music?</b> speech over noticeable music / jingle / bed &rarr; lean reject.</li>
  <li><b>Natural delivery?</b> shouting / chanting / singing / crowd responses &rarr; not useful for TTS.</li>
  <li><b>Standalone sentence?</b> complete utterances beat &ldquo;&hellip;and this is why we believe&hellip;&rdquo;.</li>
</ol>
<p><b>Reject priority:</b> 1&nbsp;multiple speakers &middot; 2&nbsp;cut start/end &middot; 3&nbsp;transcript mismatch &middot; 4&nbsp;heavy noise/music &middot; 5&nbsp;usable Ghanaian English &middot; 6&nbsp;everything else.
Don't reject a clip just because it isn't pristine &mdash; large useful dataset, not studio recordings.</p>
"""


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MANIFEST)
    qc = pd.read_csv(QC)
    df = m.merge(qc[["wav_file", "flags", "n_flags"]], on="wav_file", how="left")
    df["flags"] = df["flags"].fillna("")
    df["n_flags"] = df["n_flags"].fillna(0).astype(int)

    rng = random.Random(SEED)
    clean = df[df["n_flags"] == 0].sample(frac=1.0, random_state=SEED)
    flagged = df[df["n_flags"] > 0].sort_values(["n_flags", "wav_file"], ascending=False)
    ordered = pd.concat([flagged, clean]).reset_index(drop=True)
    ordered["listen_order"] = ordered.index + 1
    ordered = ordered.rename(columns={"flags": "auto_flags"})

    # ---- CSV sheet ----
    cols = ["listen_order", "wav_file", "stratum", "shard", "duration_ss",
            "mean_speech_prob", "dbfs", "auto_flags", "corrected_text",
            "verdict", "reason", "notes"]
    sheet = ordered.copy()
    for c in ["verdict", "reason", "notes"]:
        sheet[c] = ""
    sheet = sheet[cols]
    sheet.to_csv(REPORTS / "listening_sheet.csv", index=False)

    # ---- HTML player ----
    items = []
    for _, r in ordered.iterrows():
        items.append({
            "order": int(r["listen_order"]),
            "wav": f"../data/samples/{r['wav_file']}",
            "name": r["wav_file"],
            "stratum": r["stratum"],
            "dur": float(r["duration_ss"]),
            "prob": float(r["mean_speech_prob"]),
            "dbfs": float(r["dbfs"]),
            "auto": r["auto_flags"],
            "text": r["corrected_text"],
        })
    data_json = json.dumps(items, ensure_ascii=False)
    reasons_json = json.dumps(REASONS)

    html_doc = (HTML_TEMPLATE
                .replace("__DATA__", data_json)
                .replace("__REASONS__", reasons_json)
                .replace("__CRITERIA__", CRITERIA_HTML))
    (REPORTS / "listening_sheet.html").write_text(html_doc, encoding="utf-8")

    print(f"[done] {len(sheet)} rows -> reports/listening_sheet.csv")
    print(f"[done] player     -> reports/listening_sheet.html")
    print(f"       flagged first: {(df['n_flags'] > 0).sum()} clips, then {len(df) - (df['n_flags'] > 0).sum()} clean-random")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ghana TTS - Listening Session (200 clips)</title>
<style>
  :root { --bg:#111418; --card:#1b2027; --line:#2c343f; --txt:#dce3ec; --dim:#8a96a5;
          --acc:#4da3ff; --warn:#e0b341; --bad:#e25555; --ok:#3fb96f; }
  * { box-sizing:border-box; }
  body { margin:0; font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--txt); }
  header { position:sticky; top:0; z-index:9; background:#0d1013ee; backdrop-filter:blur(6px);
           border-bottom:1px solid var(--line); padding:10px 18px; display:flex; gap:14px; align-items:center; flex-wrap:wrap; }
  header b { font-size:16px; }
  #progress { color:var(--dim); }
  button { background:var(--acc); color:#04121f; border:0; border-radius:6px; padding:6px 12px;
           font-weight:600; cursor:pointer; }
  button.secondary { background:#2c343f; color:var(--txt); }
  label.jump { color:var(--dim); display:flex; gap:6px; align-items:center; }
  input[type=number] { width:70px; background:#0d1013; color:var(--txt); border:1px solid var(--line); border-radius:5px; padding:4px; }
  #cards { padding:16px; display:grid; gap:12px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
  .card.flagged { border-left:4px solid var(--warn); }
  .card.done { opacity:.55; }
  .card.done.verdict-keep { border-left:4px solid var(--ok); opacity:.8; }
  .card.done.verdict-reject { border-left:4px solid var(--bad); opacity:.8; }
  .card.done.verdict-borderline { border-left:4px solid var(--warn); opacity:.8; }
  .top { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:6px; }
  .ord { font-weight:700; color:var(--acc); min-width:44px; }
  .tag { font-size:11px; padding:2px 8px; border-radius:20px; background:#2c343f; color:var(--dim); }
  .tag.auto { background:#4a3a12; color:var(--warn); }
  .meta { color:var(--dim); font-size:12px; }
  .text { margin:8px 0; color:#eef3f9; }
  audio { width:100%; height:32px; margin:4px 0 8px; }
  .checks { display:flex; flex-wrap:wrap; gap:6px 14px; margin:6px 0; }
  .checks label { color:var(--dim); font-size:12.5px; display:flex; gap:5px; align-items:center; cursor:pointer; }
  .verdicts { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:6px; }
  .verdicts label { font-size:13px; cursor:pointer; padding:3px 10px; border:1px solid var(--line); border-radius:6px; color:var(--dim); }
  .verdicts input { display:none; }
  .verdicts input:checked + span { font-weight:700; }
  .verdicts label:has(input:checked) { border-color:var(--acc); color:var(--txt); background:#223244; }
  textarea { width:100%; background:#0d1013; color:var(--txt); border:1px solid var(--line);
             border-radius:6px; min-height:30px; resize:vertical; font:inherit; padding:5px; }
  select.reason { background:#0d1013; color:var(--txt); border:1px solid var(--line); border-radius:6px; padding:4px 6px; font:inherit; }
  #criteria { margin:10px 16px 0; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 14px; color:var(--dim); }
  #criteria summary { cursor:pointer; color:var(--txt); }
  #criteria ol { margin:8px 0; padding-left:20px; }
  #criteria li { margin:3px 0; }
</style>
</head>
<body>
<header>
  <b>Ghana TTS listening session</b>
  <span id="progress"></span>
  <label class="jump">jump <input id="jump" type="number" min="1"></label>
  <button class="secondary" onclick="jumpTo()">Go</button>
  <button onclick="exportCsv()">Export marks (CSV)</button>
  <button class="secondary" onclick="resetMarks()">Reset all marks</button>
</header>
<details id="criteria"><summary><b>What to listen for</b> &mdash; 9 checks + reject priority (click to expand)</summary>
  __CRITERIA__
</details>
<div id="cards"></div>
<script>
const ITEMS = __DATA__;
const REASONS = __REASONS__;
const KEY = "ghana-tts-listening-marks-v2";

function loadMarks() { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } }
function saveMarks(m) { localStorage.setItem(KEY, JSON.stringify(m)); }
let marks = loadMarks();

function clipMark(name) { return marks[name] || (marks[name] = { verdict: "", reason: "", notes: "" }); }

function render() {
  const root = document.getElementById("cards");
  for (const it of ITEMS) {
    const mk = clipMark(it.name);
    const card = document.createElement("div");
    card.className = "card" + (it.auto ? " flagged" : "");
    card.id = "card-" + it.order;
    const tags = `<span class="tag">${it.stratum}</span>` +
      (it.auto ? it.auto.split(";").map(f => `<span class="tag auto">${f}</span>`).join("") : "");
    card.innerHTML = `
      <div class="top">
        <span class="ord">#${it.order}</span>${tags}
        <span class="meta">${it.name} &middot; ${it.dur.toFixed(1)}s &middot; p=${it.prob.toFixed(3)} &middot; ${it.dbfs.toFixed(1)} dBFS</span>
      </div>
      <div class="text">${escapeHtml(it.text)}</div>
      <audio controls preload="none" src="${it.wav}"></audio>
      <div class="verdicts">
        ${["keep", "borderline", "reject"].map(v => `
          <label><input type="radio" name="v-${it.name}" value="${v}" ${mk.verdict === v ? "checked" : ""}><span>${v}</span></label>`).join("")}
        <select data-name="${it.name}" data-role="reason" class="reason">
          <option value="">reason&hellip;</option>
          ${REASONS.map(r => `<option value="${r}" ${mk.reason === r ? "selected" : ""}>${r}</option>`).join("")}
        </select>
      </div>
      <textarea data-name="${it.name}" placeholder="notes...">${escapeHtml(mk.notes)}</textarea>`;
    root.appendChild(card);
    if (mk.verdict) card.classList.add("done", "verdict-" + mk.verdict);
  }
  updateProgress();
}

function escapeHtml(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

document.addEventListener("change", e => {
  const t = e.target;
  if (t.dataset.role === "reason") {
    clipMark(t.dataset.name).reason = t.value;
    saveMarks(marks);
  } else if (t.type === "radio" && t.name.startsWith("v-")) {
    const name = t.name.slice(2);
    clipMark(name).verdict = t.value;
    saveMarks(marks);
    const card = t.closest(".card");
    card.classList.remove("done", "verdict-keep", "verdict-reject", "verdict-borderline");
    card.classList.add("done", "verdict-" + t.value);
    updateProgress();
    const next = card.nextElementSibling;
    if (next) next.querySelector("audio")?.play().catch(() => {});
  }
});
document.addEventListener("input", e => {
  const t = e.target;
  if (t.tagName === "TEXTAREA" && t.dataset.name) {
    clipMark(t.dataset.name).notes = t.value;
    saveMarks(marks);
  }
});

function updateProgress() {
  const done = ITEMS.filter(i => marks[i.name]?.verdict).length;
  document.getElementById("progress").textContent = `${done}/${ITEMS.length} judged`;
}

function jumpTo() {
  const n = parseInt(document.getElementById("jump").value, 10);
  const el = document.getElementById("card-" + n);
  if (el) { el.scrollIntoView({ behavior: "smooth", block: "start" }); el.querySelector("audio")?.play().catch(() => {}); }
}

function exportCsv() {
  const head = ["listen_order", "wav_file", "verdict", "reason", "notes"];
  const rows = [head.join(",")];
  for (const it of ITEMS) {
    const mk = marks[it.name] || {};
    const cells = [it.order, it.name, mk.verdict || "", mk.reason || "",
      JSON.stringify(mk.notes || "")];
    rows.push(cells.join(","));
  }
  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "listening_marks.csv";
  a.click();
}

function resetMarks() {
  if (confirm("Clear all marks?")) { marks = {}; saveMarks(marks); location.reload(); }
}

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
