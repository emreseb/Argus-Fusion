"""
Dataset Reviewer
-----------------
Run:  python reviewer.py
Open: http://localhost:5000

Keys:  ← → navigate   I L D B S flip bits   Enter commit all
JSON history file (rename_history.json) persists renames across sessions.
Previously renamed frames are highlighted green on startup.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request, send_file

# ── Config ─────────────────────────────────────────────────────────────────────
IMAGES_DIR   = "/home/emre/Desktop/NATO/DATASETv3/images(all)"
LABELS_DIR   = "/home/emre/Desktop/NATO/DATASETv3/fully_paired_annotated/labels"
HISTORY_FILE = "rename_history.json"
# ───────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

flip_registry: dict[str, str] = {}   # original_stem -> pending_new_stem
queued:        set[str]       = set()


# ── History helpers ─────────────────────────────────────────────────────────────

def load_history() -> dict:
    """Returns {original_stem: {new_stem, timestamp}} for all past commits."""
    if Path(HISTORY_FILE).exists():
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_history(history: dict):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ── Filename helpers ────────────────────────────────────────────────────────────

EXPERIMENT_CODES = ["ALL", "ERF", "E", "R", "N", "M"]


def extract_experiment_code(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    for part in stem.split("_"):
        if not part.isdigit():
            code = re.sub(r"\d+$", "", part).upper()
            return code if code else "OTHER"
    return "OTHER"


def get_images(experiment_filter: str = "ALL") -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    imgs = sorted(
        p for p in Path(IMAGES_DIR).iterdir()
        if p.suffix.lower() in exts and not p.name.startswith(".")
    )
    if experiment_filter != "ALL":
        imgs = [p for p in imgs if extract_experiment_code(p.name) == experiment_filter]
    return imgs


def flip_bit_generic(stem: str, cat: str) -> str | None:
    parts    = stem.split("_")
    num_idxs = [i for i, p in enumerate(parts) if p.isdigit()]
    if len(num_idxs) < 3:
        return None
    i_idx, ldb_idx, s_idx = num_idxs[0], num_idxs[1], num_idxs[2]
    ldb = parts[ldb_idx].zfill(3)

    if   cat == "item":
        parts[i_idx]   = "1" if parts[i_idx] == "0" else "0"
    elif cat == "light":
        parts[ldb_idx] = ("1" if ldb[0]=="0" else "0") + ldb[1] + ldb[2]
    elif cat == "distance":
        parts[ldb_idx] = ldb[0] + ("1" if ldb[1]=="0" else "0") + ldb[2]
    elif cat == "background":
        parts[ldb_idx] = ldb[0] + ldb[1] + ("1" if ldb[2]=="0" else "0")
    elif cat == "sensor":
        parts[s_idx]   = "1" if parts[s_idx] == "0" else "0"
    else:
        return None
    return "_".join(parts)


# ── Routes ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    all_imgs = get_images("ALL")
    code_counts: dict[str, int] = {}
    for p in all_imgs:
        c = extract_experiment_code(p.name)
        code_counts[c] = code_counts.get(c, 0) + 1

    names   = [p.name for p in all_imgs]
    history = load_history()
    # Set of new_stems that have already been committed (shown green)
    renamed_stems = {v["new_stem"] for v in history.values()}

    return render_template_string(
        HTML_TEMPLATE,
        image_names     = names,
        total           = len(names),
        experiment_codes= EXPERIMENT_CODES,
        code_counts     = code_counts,
        renamed_stems   = list(renamed_stems),
    )


@app.route("/images_list")
def images_list():
    exp  = request.args.get("experiment", "ALL")
    imgs = get_images(exp)
    return jsonify(names=[p.name for p in imgs])


@app.route("/image/<filename>")
def serve_image(filename):
    path = Path(IMAGES_DIR) / filename
    if not path.exists():
        return "Not found", 404
    return send_file(str(path))


@app.route("/queue_flip", methods=["POST"])
def queue_flip():
    data = request.json
    stem = data.get("stem")
    cat  = data.get("cat")
    if not stem or not cat:
        return jsonify(ok=False, error="Missing stem or cat"), 400

    base    = flip_registry.get(stem, stem)
    flipped = flip_bit_generic(base, cat)
    if flipped is None:
        return jsonify(ok=False, error="Could not parse stem"), 400

    # If flipping back to original, remove from queue entirely
    if flipped == stem:
        flip_registry.pop(stem, None)
        queued.discard(stem)
        return jsonify(ok=True, queued=len(queued), new_stem=stem, removed=True)

    flip_registry[stem] = flipped
    queued.add(stem)
    return jsonify(ok=True, queued=len(queued), new_stem=flipped, removed=False)


@app.route("/history")
def get_history():
    return jsonify(load_history())


@app.route("/commit", methods=["POST"])
def commit():
    results = {"renamed": [], "errors": [], "skipped_labels": []}
    history = load_history()

    for original_stem, new_stem in list(flip_registry.items()):

        # ── Rename image ──────────────────────────────────────────────────────
        img_renamed = False
        for ext in (".jpg", ".jpeg", ".png"):
            old_img = Path(IMAGES_DIR) / (original_stem + ext)
            if old_img.exists():
                new_img = Path(IMAGES_DIR) / (new_stem + ext)
                old_img.rename(new_img)
                results["renamed"].append(f"{old_img.name} → {new_img.name}")
                img_renamed = True
                break

        if not img_renamed:
            results["errors"].append(f"Image not found: {original_stem}")
            continue

        # ── Rename label (graceful — skip if missing) ─────────────────────────
        old_txt = Path(LABELS_DIR) / (original_stem + ".txt")
        if old_txt.exists():
            new_txt = Path(LABELS_DIR) / (new_stem + ".txt")
            old_txt.rename(new_txt)
            results["renamed"].append(f"{old_txt.name} → {new_txt.name}")
        else:
            results["skipped_labels"].append(f"No label for: {original_stem} (image renamed only)")

        # ── Update history ────────────────────────────────────────────────────
        # If this stem was itself a prior rename, chain back to the true original
        true_original = next(
            (k for k, v in history.items() if v["new_stem"] == original_stem),
            original_stem
        )
        history[true_original] = {
            "new_stem":  new_stem,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "label_renamed": old_txt.exists(),   # always False here since we already renamed it; kept for schema clarity
        }

        queued.discard(original_stem)
        del flip_registry[original_stem]

    save_history(history)
    return jsonify(results)


# ── HTML ────────────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Dataset Reviewer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #1a1a1a; color: #eee;
         display: flex; height: 100vh; overflow: hidden; }

  #viewer { flex: 1; display: flex; flex-direction: column;
            align-items: center; justify-content: center; padding: 24px; gap: 12px; }

  #img-wrap { position: relative; max-width: 100%; max-height: calc(100vh - 160px); }
  #main-img { max-width: 100%; max-height: calc(100vh - 160px);
              border-radius: 6px; border: 3px solid transparent; transition: border-color .2s; }
  #main-img.queued  { border-color: #f59e0b; }
  #main-img.renamed { border-color: #22c55e; }

  .img-badge {
    display: none; position: absolute; top: 10px; left: 10px;
    font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 20px;
  }
  #badge-queued  { background: #f59e0b; color: #000; }
  #badge-renamed { background: #22c55e; color: #000; }
  #main-img.queued  ~ #badge-queued  { display: block; }
  #main-img.renamed ~ #badge-renamed { display: block; }

  #nav { display: flex; align-items: center; gap: 16px; }
  #nav button { background: #2d2d2d; color: #eee; border: 1px solid #444;
                padding: 8px 20px; border-radius: 6px; cursor: pointer;
                font-size: 14px; transition: background .15s; }
  #nav button:hover { background: #3d3d3d; }
  #counter { font-size: 14px; color: #aaa; min-width: 90px; text-align: center; }

  /* ── Filename display ── */
  #filename-box {
    background: #111; border: 1px solid #2d2d2d; border-radius: 8px;
    padding: 10px 18px; max-width: 780px; width: 100%; text-align: center;
  }
  #filename-current {
    font-family: monospace; font-size: 13px; color: #e2e8f0;
    word-break: break-all; line-height: 1.5;
  }
  #filename-pending {
    font-family: monospace; font-size: 12px; color: #f59e0b;
    margin-top: 4px; word-break: break-all;
    display: none;
  }
  #filename-was {
    font-family: monospace; font-size: 11px; color: #22c55e;
    margin-top: 4px; word-break: break-all;
    display: none;
  }

  #tags { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
  .tag { font-size: 12px; font-weight: 600; padding: 3px 12px;
         border-radius: 20px; background: #2d2d2d; }
  .tag span { font-weight: 400; margin-right: 4px; color: #555; }
  .tag-item       { color: #60a5fa; }
  .tag-light      { color: #fbbf24; }
  .tag-distance   { color: #f87171; }
  .tag-background { color: #a78bfa; }
  .tag-sensor     { color: #34d399; }

  /* ── Sidebar ── */
  #sidebar { width: 230px; background: #111; border-left: 1px solid #2d2d2d;
             display: flex; flex-direction: column; padding: 20px 14px;
             gap: 14px; overflow-y: auto; }
  #sidebar h2 { font-size: 12px; color: #666; text-transform: uppercase;
                letter-spacing: .1em; }

  .btn { width: 100%; padding: 9px; border-radius: 6px; border: none;
         cursor: pointer; font-size: 13px; font-weight: 600;
         transition: opacity .15s; }
  .btn:hover  { opacity: .85; }
  .btn:active { opacity: .7; }

  #btn-commit { background: #22c55e; color: #000; }
  .exp-btn { border: 1px solid #333 !important; text-align: left;
             background: #1e1e1e; color: #bbb; display: flex;
             justify-content: space-between; align-items: center; }
  .exp-btn.active { background: #2d2d2d !important; color: #fff !important;
                    border-color: #666 !important; }
  .exp-badge { font-size: 11px; color: #555; }

  #queue-count { font-size: 12px; color: #666; text-align: center; }

  #log { flex: 1; overflow-y: auto; font-size: 11px; color: #555;
         border-top: 1px solid #222; padding-top: 10px; line-height: 1.8; }
  #log .entry  { color: #f59e0b; }
  #log .ok     { color: #22c55e; }
  #log .skip   { color: #94a3b8; }
  #log .err    { color: #ef4444; }
</style>
</head>
<body>

<div id="viewer">
  <div id="img-wrap">
    <img id="main-img" src="" alt="">
    <div class="img-badge" id="badge-queued">QUEUED</div>
    <div class="img-badge" id="badge-renamed">RENAMED</div>
  </div>

  <div id="nav">
    <button id="btn-prev">&#8592; Prev</button>
    <span id="counter">— / —</span>
    <button id="btn-next">Next &#8594;</button>
  </div>

  <div id="filename-box">
    <div id="filename-current">—</div>
    <div id="filename-pending"></div>
    <div id="filename-was"></div>
  </div>

  <div id="tags"></div>
</div>

<div id="sidebar">
  <h2>Actions</h2>
  <button class="btn" id="btn-commit">&#10003; Commit All</button>
  <div id="queue-count">0 queued</div>

  <h2>Flip bit</h2>
  <div style="display:flex;flex-direction:column;gap:6px;">
    <button class="btn flip-btn" data-cat="item"       style="background:#1a2e4a;color:#60a5fa;">[I] Item</button>
    <button class="btn flip-btn" data-cat="light"      style="background:#2e2200;color:#fbbf24;">[L] Light</button>
    <button class="btn flip-btn" data-cat="distance"   style="background:#2e1515;color:#f87171;">[D] Distance</button>
    <button class="btn flip-btn" data-cat="background" style="background:#1f1535;color:#a78bfa;">[B] Background</button>
    <button class="btn flip-btn" data-cat="sensor"     style="background:#0d2318;color:#34d399;">[S] Sensor</button>
  </div>

  <h2>Experiment</h2>
  <div style="display:flex;flex-direction:column;gap:5px;">
    {% for code in experiment_codes %}
    <button class="btn exp-btn" data-code="{{ code }}">
      <span>{{ code }}</span>
      <span class="exp-badge">
        {% if code == "ALL" %}{{ code_counts.values()|sum }}
        {% else %}{{ code_counts.get(code, 0) }}{% endif %}
      </span>
    </button>
    {% endfor %}
  </div>

  <div id="log"></div>
</div>

<script>
const images       = {{ image_names | tojson }};
const renamedStems = new Set({{ renamed_stems | tojson }});
let idx    = 0;
let queued = new Set();

const imgEl        = document.getElementById("main-img");
const counter      = document.getElementById("counter");
const filenameCur  = document.getElementById("filename-current");
const filenamePend = document.getElementById("filename-pending");
const filenameWas  = document.getElementById("filename-was");
const tagsEl       = document.getElementById("tags");
const btnCommit    = document.getElementById("btn-commit");
const queueCount   = document.getElementById("queue-count");
const logEl        = document.getElementById("log");

// Per-stem pending new name (so we can show it in the filename box)
const pendingNames = {};  // stem -> new_stem string

const MAPS = {
  item:       {"0":"Bird",      "1":"Drone"},
  light:      {"0":"Bright",    "1":"Low"},
  distance:   {"0":"Close",     "1":"Far"},
  background: {"0":"Clear",     "1":"Cluttered"},
  sensor:     {"0":"EO",        "1":"IR"},
};

function stemOf(filename) { return filename.replace(/\.[^.]+$/, ""); }

function parseCategories(s) {
  const parts = s.split("_");
  const nums  = parts.filter(p => /^\d+$/.test(p));
  if (nums.length < 3) return null;
  const LDB = nums[1].padStart(3, "0");
  return {
    item:       MAPS.item[nums[0]]       || "?",
    light:      MAPS.light[LDB[0]]      || "?",
    distance:   MAPS.distance[LDB[1]]   || "?",
    background: MAPS.background[LDB[2]] || "?",
    sensor:     MAPS.sensor[nums[2]]    || "?",
  };
}

function renderTags(s) {
  const c = parseCategories(s);
  if (!c) { tagsEl.innerHTML = ""; return; }
  tagsEl.innerHTML = `
    <div class="tag tag-item">      <span>Item</span>${c.item}</div>
    <div class="tag tag-light">     <span>Light</span>${c.light}</div>
    <div class="tag tag-distance">  <span>Distance</span>${c.distance}</div>
    <div class="tag tag-background"><span>Background</span>${c.background}</div>
    <div class="tag tag-sensor">    <span>Sensor</span>${c.sensor}</div>
  `;
}

function load(i) {
  if (!images.length) {
    counter.textContent = "0 / 0";
    filenameCur.textContent = "No images";
    return;
  }
  idx = (i + images.length) % images.length;
  const name = images[idx];
  const s    = stemOf(name);

  imgEl.src = "/image/" + encodeURIComponent(name);
  counter.textContent = `${idx + 1} / ${images.length}`;

  // ── Filename box ──────────────────────────────────────────────────────────
  filenameCur.textContent = name;

  const pending = pendingNames[s];
  if (pending) {
    filenamePend.textContent = "→ " + pending;
    filenamePend.style.display = "block";
  } else {
    filenamePend.style.display = "none";
  }

  // Show original name if this stem is the result of a prior commit
  const wasEntry = [...renamedStems].includes(s) ? s : null;
  // We can't easily reverse-lookup here without the full history,
  // so we just mark it visually via the badge
  filenameWas.style.display = "none";

  // ── Border state ──────────────────────────────────────────────────────────
  const isQ = queued.has(s);
  const isR = renamedStems.has(s);
  imgEl.classList.toggle("queued",  isQ);
  imgEl.classList.toggle("renamed", isR && !isQ);

  renderTags(pending || s);
}

function addLog(msg, cls) {
  const d = document.createElement("div");
  d.className = cls || "";
  d.textContent = msg;
  logEl.prepend(d);
}

function updateQueueCount() {
  queueCount.textContent = queued.size + " queued";
}

function flipCatInStem(s, cat) {
  const parts   = s.split("_");
  const numIdxs = parts.reduce((a, p, i) => (/^\d+$/.test(p) ? [...a, i] : a), []);
  if (numIdxs.length < 3) return null;
  const [iIdx, ldbIdx, sIdx] = numIdxs;
  const ldb = parts[ldbIdx].padStart(3, "0");

  if      (cat === "item")       parts[iIdx]   = parts[iIdx]   === "0" ? "1" : "0";
  else if (cat === "light")      parts[ldbIdx] = (ldb[0]==="0"?"1":"0") + ldb[1] + ldb[2];
  else if (cat === "distance")   parts[ldbIdx] = ldb[0] + (ldb[1]==="0"?"1":"0") + ldb[2];
  else if (cat === "background") parts[ldbIdx] = ldb[0] + ldb[1] + (ldb[2]==="0"?"1":"0");
  else if (cat === "sensor")     parts[sIdx]   = parts[sIdx]   === "0" ? "1" : "0";
  else return null;
  return parts.join("_");
}

async function flipCategory(cat) {
  if (!images.length) return;
  const name = images[idx];
  const s    = stemOf(name);

  const res  = await fetch("/queue_flip", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({stem: s, cat})
  });
  const data = await res.json();

  if (data.removed) {
    queued.delete(s);
    delete pendingNames[s];
    addLog(`Unqueued: ${s}`, "");
  } else {
    queued.add(s);
    pendingNames[s] = data.new_stem;
    addLog(`[${cat}] ${s} → ${data.new_stem}`, "entry");
  }
  updateQueueCount();
  load(idx);
}

// ── Flip buttons ─────────────────────────────────────────────────────────────
document.querySelectorAll(".flip-btn").forEach(btn =>
  btn.addEventListener("click", () => flipCategory(btn.dataset.cat))
);

// ── Commit ───────────────────────────────────────────────────────────────────
btnCommit.addEventListener("click", async () => {
  if (!queued.size) { addLog("Nothing queued.", ""); return; }
  const res  = await fetch("/commit", {method: "POST"});
  const data = await res.json();
  data.renamed.forEach(r => {
    addLog("✓ " + r, "ok");
    // Move new stems into renamedStems so they highlight green
    const newStem = r.split(" → ")[1]?.replace(/\.[^.]+$/, "");
    if (newStem) renamedStems.add(newStem);
  });
  data.skipped_labels?.forEach(s => addLog("⚠ " + s, "skip"));
  data.errors?.forEach(e       => addLog("✗ " + e, "err"));
  queued.clear();
  Object.keys(pendingNames).forEach(k => delete pendingNames[k]);
  updateQueueCount();
  setTimeout(() => location.reload(), 600);
});

// ── Navigation ───────────────────────────────────────────────────────────────
document.getElementById("btn-prev").addEventListener("click", () => load(idx - 1));
document.getElementById("btn-next").addEventListener("click", () => load(idx + 1));

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  if (e.key === "ArrowLeft"  || e.key === "a") return load(idx - 1);
  if (e.key === "ArrowRight" || e.key === "d") return load(idx + 1);
  if (e.key === "Enter") return btnCommit.click();
  const keyMap = {i:"item", l:"light", d:"distance", b:"background", s:"sensor"};
  const cat = keyMap[e.key.toLowerCase()];
  if (cat) flipCategory(cat);
});

// ── Experiment filter ─────────────────────────────────────────────────────────
let activeExp = "ALL";
const expBtns = document.querySelectorAll(".exp-btn");

function setActiveExp(code) {
  activeExp = code;
  expBtns.forEach(b => b.classList.toggle("active", b.dataset.code === code));
}

expBtns.forEach(btn => {
  btn.addEventListener("click", async () => {
    const code = btn.dataset.code;
    setActiveExp(code);
    const res  = await fetch("/images_list?experiment=" + encodeURIComponent(code));
    const data = await res.json();
    images.length = 0;
    data.names.forEach(n => images.push(n));
    addLog(`Filter: ${code} — ${images.length} images`, "");
    load(0);
  });
});

setActiveExp("ALL");
load(0);
updateQueueCount();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting reviewer at http://localhost:5000")
    print(f"  Images  : {IMAGES_DIR}")
    print(f"  Labels  : {LABELS_DIR}")
    print(f"  History : {Path(HISTORY_FILE).resolve()}")
    app.run(host="0.0.0.0", debug=False, port=5000)