"""
Dataset Background-Bit Reviewer
--------------------------------
Run:  python reviewer.py
Then open:  http://localhost:5000

Controls:
  - Scroll through images with Prev / Next (or arrow keys)
  - "Change"  — queues the current image for a background bit flip, highlights it
  - "Commit"  — renames all queued .jpg and .txt files in place (B bit flipped)
  - A pending_changes.txt log is written alongside this script
"""

import os
import re
from pathlib import Path
from flask import Flask, render_template_string, jsonify, request, send_file

# ── Config ────────────────────────────────────────────────────────────────────
IMAGES_DIR = "/home/emre/Desktop/NATO/DATASETv3/fully_paired_annotated/images"
LABELS_DIR = "/home/emre/Desktop/NATO/DATASETv3/fully_paired_annotated/labels"
LOG_FILE   = "pending_changes.txt"
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# flip_registry[stem] = new_stem (accumulated flips)
flip_registry: dict[str, str] = {}
queued: set[str] = set()


@app.route("/queue_flip", methods=["POST"])
def queue_flip():
    data     = request.json
    stem     = data.get("stem")
    new_stem = data.get("new_stem")
    cat      = data.get("cat")
    if not stem or not new_stem:
        return jsonify(ok=False, error="Missing stem or new_stem"), 400

    # If already in registry, apply the flip on top of the already-pending new_stem
    base = flip_registry.get(stem, stem)
    # Re-flip the cat on the current pending stem
    flipped = flip_background_bit_generic(base, cat)
    if flipped is None:
        return jsonify(ok=False, error="Could not parse"), 400

    flip_registry[stem] = flipped
    queued.add(stem)

    with open(LOG_FILE, "a") as f:
        f.write(f"{stem} -> {flipped}\n")

    return jsonify(ok=True, queued=len(queued), new_stem=flipped)


def flip_background_bit_generic(stem: str, cat: str) -> str | None:
    """Flip any category bit in a stem."""
    parts = stem.split("_")
    num_idxs = [i for i, p in enumerate(parts) if p.isdigit()]
    if len(num_idxs) < 3:
        return None
    i_idx, ldb_idx, s_idx = num_idxs[0], num_idxs[1], num_idxs[2]
    ldb = parts[ldb_idx].zfill(3)

    if cat == "item":
        parts[i_idx] = "1" if parts[i_idx] == "0" else "0"
    elif cat == "light":
        parts[ldb_idx] = ("1" if ldb[0]=="0" else "0") + ldb[1] + ldb[2]
    elif cat == "distance":
        parts[ldb_idx] = ldb[0] + ("1" if ldb[1]=="0" else "0") + ldb[2]
    elif cat == "background":
        parts[ldb_idx] = ldb[0] + ldb[1] + ("1" if ldb[2]=="0" else "0")
    elif cat == "sensor":
        parts[s_idx] = "1" if parts[s_idx] == "0" else "0"
    else:
        return None
    return "_".join(parts)


def get_images():
    exts = {".jpg", ".jpeg", ".png"}
    imgs = sorted(
        p for p in Path(IMAGES_DIR).iterdir()
        if p.suffix.lower() in exts and not p.name.startswith(".")
    )
    return imgs


def flip_background_bit(stem: str) -> str:
    """
    Filename format: {I}_{LDB}_{S}_{Experiment}_frame{###}
    LDB is token[1]; B is LDB[2]. Flip it 0<->1.
    """
    parts = stem.split("_")
    # Find the LDB token (second purely-numeric token)
    numeric_indices = [i for i, p in enumerate(parts) if p.isdigit()]
    if len(numeric_indices) < 2:
        raise ValueError(f"Cannot parse LDB from stem: {stem}")

    ldb_idx = numeric_indices[1]
    ldb = parts[ldb_idx]

    if len(ldb) != 3:
        raise ValueError(f"LDB token '{ldb}' is not 3 digits in: {stem}")

    b = ldb[2]
    if b not in ("0", "1"):
        raise ValueError(f"Background bit '{b}' is not 0 or 1 in: {stem}")

    new_b   = "1" if b == "0" else "0"
    new_ldb = ldb[:2] + new_b
    parts[ldb_idx] = new_ldb
    return "_".join(parts)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    images = get_images()
    names  = [p.name for p in images]
    return render_template_string(HTML_TEMPLATE, image_names=names,
                                  total=len(names))


@app.route("/image/<filename>")
def serve_image(filename):
    path = Path(IMAGES_DIR) / filename
    if not path.exists():
        return "Not found", 404
    return send_file(str(path))


@app.route("/queue", methods=["POST"])
def queue_image():
    stem = request.json.get("stem")
    if not stem:
        return jsonify(ok=False, error="No stem provided"), 400
    queued.add(stem)
    # Append to log
    with open(LOG_FILE, "a") as f:
        f.write(stem + "\n")
    return jsonify(ok=True, queued=len(queued))


@app.route("/unqueue", methods=["POST"])
def unqueue_image():
    stem = request.json.get("stem")
    if stem in queued:
        queued.discard(stem)
    return jsonify(ok=True, queued=len(queued))


@app.route("/queued")
def get_queued():
    return jsonify(stems=list(queued))


@app.route("/commit", methods=["POST"])
def commit():
    results = {"renamed": [], "errors": []}

    for stem, new_stem in list(flip_registry.items()):
        # Rename image
        for ext in (".jpg", ".jpeg", ".png"):
            old_img = Path(IMAGES_DIR) / (stem + ext)
            if old_img.exists():
                new_img = Path(IMAGES_DIR) / (new_stem + ext)
                old_img.rename(new_img)
                results["renamed"].append(f"{old_img.name} → {new_img.name}")
                break

        # Rename label
        old_txt = Path(LABELS_DIR) / (stem + ".txt")
        if old_txt.exists():
            new_txt = Path(LABELS_DIR) / (new_stem + ".txt")
            old_txt.rename(new_txt)
            results["renamed"].append(f"{old_txt.name} → {new_txt.name}")
        else:
            results["errors"].append(f"Label not found for: {stem}")

        queued.discard(stem)
        del flip_registry[stem]

    open(LOG_FILE, "w").close()
    return jsonify(results)


# ── HTML ──────────────────────────────────────────────────────────────────────

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

  /* ── Left: image viewer ── */
  #viewer { flex: 1; display: flex; flex-direction: column;
            align-items: center; justify-content: center; padding: 24px; gap: 16px; }

  #img-wrap { position: relative; max-width: 100%; max-height: calc(100vh - 140px); }
  #main-img { max-width: 100%; max-height: calc(100vh - 140px);
              border-radius: 6px; border: 3px solid transparent; transition: border-color .2s; }
  #main-img.queued { border-color: #f59e0b; }

  #queued-badge {
    display: none; position: absolute; top: 10px; left: 10px;
    background: #f59e0b; color: #000; font-size: 12px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
  }
  #main-img.queued + #queued-badge { display: block; }

  #nav { display: flex; align-items: center; gap: 16px; }
  #nav button { background: #2d2d2d; color: #eee; border: 1px solid #444;
                padding: 8px 20px; border-radius: 6px; cursor: pointer;
                font-size: 14px; transition: background .15s; }
  #nav button:hover { background: #3d3d3d; }
  #counter { font-size: 14px; color: #aaa; min-width: 90px; text-align: center; }
  #filename { font-size: 12px; color: #888; max-width: 600px;
              text-align: center; word-break: break-all; }

  #tags { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
  .tag { font-size: 12px; font-weight: 600; padding: 3px 12px;
         border-radius: 20px; background: #2d2d2d; }
  .tag span { font-weight: 400; margin-right: 4px; color: #666; }
  .tag-item       { color: #60a5fa; }
  .tag-light      { color: #fbbf24; }
  .tag-distance   { color: #f87171; }
  .tag-background { color: #a78bfa; }
  .tag-sensor     { color: #34d399; }

  /* ── Right: controls ── */
  #sidebar { width: 220px; background: #111; border-left: 1px solid #2d2d2d;
             display: flex; flex-direction: column; padding: 24px 16px; gap: 16px; }
  #sidebar h2 { font-size: 14px; color: #aaa; text-transform: uppercase;
                letter-spacing: .08em; }

  .btn { width: 100%; padding: 10px; border-radius: 6px; border: none;
         cursor: pointer; font-size: 14px; font-weight: 600; transition: opacity .15s; }
  .btn:hover { opacity: .85; }
  .btn:active { opacity: .7; }

  #btn-change  { background: #f59e0b; color: #000; }
  #btn-commit  { background: #22c55e; color: #000; }
  #btn-change.unchange { background: #6b7280; color: #eee; }

  #queue-count { font-size: 13px; color: #aaa; text-align: center; }

  #log { flex: 1; overflow-y: auto; font-size: 11px; color: #666;
         border-top: 1px solid #2d2d2d; padding-top: 12px; line-height: 1.7; }
  #log .entry { color: #f59e0b; }
  #log .ok    { color: #22c55e; }
  #log .err   { color: #ef4444; }
</style>
</head>
<body>

<div id="viewer">
  <div id="img-wrap">
    <img id="main-img" src="" alt="">
    <div id="queued-badge">QUEUED</div>
  </div>
  <div id="nav">
    <button id="btn-prev">&#8592; Prev</button>
    <span id="counter">1 / {{ total }}</span>
    <button id="btn-next">Next &#8594;</button>
  </div>
  <div id="filename"></div>
  <div id="tags"></div>
</div>

<div id="sidebar">
  <h2>Actions</h2>
  <button class="btn" id="btn-change">Change (background)</button>
  <button class="btn" id="btn-commit">Commit All</button>
  <div id="queue-count">0 queued</div>

  <h2 style="margin-top:8px;">Flip bit</h2>
  <div style="display:flex;flex-direction:column;gap:8px;">
    <button class="btn flip-btn" data-cat="item"       data-key="I" style="background:#1e3a5f;color:#60a5fa;">I — Item</button>
    <button class="btn flip-btn" data-cat="light"      data-key="L" style="background:#3d2e00;color:#fbbf24;">L — Light</button>
    <button class="btn flip-btn" data-cat="distance"   data-key="D" style="background:#3d1a1a;color:#f87171;">D — Distance</button>
    <button class="btn flip-btn" data-cat="background" data-key="B" style="background:#2e1f4f;color:#a78bfa;">B — Background</button>
    <button class="btn flip-btn" data-cat="sensor"     data-key="S" style="background:#0f2e20;color:#34d399;">S — Sensor</button>
  </div>
  <div id="log"></div>
</div>

<script>
const images = {{ image_names | tojson }};
let idx = 0;
let queued = new Set();

const imgEl      = document.getElementById("main-img");
const counter    = document.getElementById("counter");
const fileLabel  = document.getElementById("filename");
const tagsEl     = document.getElementById("tags");
const btnChange  = document.getElementById("btn-change");
const btnCommit  = document.getElementById("btn-commit");
const queueCount = document.getElementById("queue-count");
const log        = document.getElementById("log");

const ITEM_MAP       = {"0": "Bird",      "1": "Drone"};
const LIGHT_MAP      = {"0": "Bright",    "1": "Low"};
const DISTANCE_MAP   = {"0": "Close",     "1": "Far"};
const BACKGROUND_MAP = {"0": "Clear",     "1": "Cluttered"};
const SENSOR_MAP     = {"0": "EO",        "1": "IR"};

function parseCategories(s) {
  const parts = s.split("_");
  const numParts = [];
  for (const p of parts) {
    if (/^\d+$/.test(p)) numParts.push(p);
    else break;
  }
  if (numParts.length < 3) return null;
  const LDB = numParts[1].padStart(3, "0");
  return {
    item:       ITEM_MAP[numParts[0]]  || "?",
    light:      LIGHT_MAP[LDB[0]]     || "?",
    distance:   DISTANCE_MAP[LDB[1]]  || "?",
    background: BACKGROUND_MAP[LDB[2]]|| "?",
    sensor:     SENSOR_MAP[numParts[2]]|| "?",
  };
}

function renderTags(s) {
  const cats = parseCategories(s);
  if (!cats) { tagsEl.innerHTML = ""; return; }
  tagsEl.innerHTML = `
    <div class="tag tag-item">      <span>Item</span>${cats.item}</div>
    <div class="tag tag-light">     <span>Light</span>${cats.light}</div>
    <div class="tag tag-distance">  <span>Distance</span>${cats.distance}</div>
    <div class="tag tag-background"><span>Background</span>${cats.background}</div>
    <div class="tag tag-sensor">    <span>Sensor</span>${cats.sensor}</div>
  `;
}

function stem(filename) {
  return filename.replace(/\.[^.]+$/, "");
}

// Flip a specific category bit in a stem string, return new stem
function flipCatInStem(s, cat) {
  const parts = s.split("_");
  const numIdxs = [];
  for (let i = 0; i < parts.length; i++) {
    if (/^\d+$/.test(parts[i])) numIdxs.push(i);
    else break;
  }
  if (numIdxs.length < 3) return null;

  const iIdx   = numIdxs[0];
  const ldbIdx = numIdxs[1];
  const sIdx   = numIdxs[2];
  const ldb    = parts[ldbIdx].padStart(3, "0");

  if (cat === "item") {
    parts[iIdx] = parts[iIdx] === "0" ? "1" : "0";
  } else if (cat === "light") {
    parts[ldbIdx] = (ldb[0] === "0" ? "1" : "0") + ldb[1] + ldb[2];
  } else if (cat === "distance") {
    parts[ldbIdx] = ldb[0] + (ldb[1] === "0" ? "1" : "0") + ldb[2];
  } else if (cat === "background") {
    parts[ldbIdx] = ldb[0] + ldb[1] + (ldb[2] === "0" ? "1" : "0");
  } else if (cat === "sensor") {
    parts[sIdx] = parts[sIdx] === "0" ? "1" : "0";
  }
  return parts.join("_");
}

function load(i) {
  idx = (i + images.length) % images.length;
  const name = images[idx];
  const s    = stem(name);
  imgEl.src  = "/image/" + encodeURIComponent(name);
  counter.textContent = (idx + 1) + " / " + images.length;
  fileLabel.textContent = name;
  renderTags(s);

  const isQ = queued.has(s);
  imgEl.classList.toggle("queued", isQ);
  btnChange.textContent = isQ ? "Unchange" : "Change";
  btnChange.classList.toggle("unchange", isQ);
}

function addLog(msg, cls) {
  const d = document.createElement("div");
  d.className = cls || "";
  d.textContent = msg;
  log.prepend(d);
}

function updateQueueCount() {
  queueCount.textContent = queued.size + " queued";
}

async function queueStem(s) {
  await fetch("/queue", { method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({stem: s}) });
  queued.add(s);
}

async function unqueueStem(s) {
  await fetch("/unqueue", { method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({stem: s}) });
  queued.delete(s);
}

// Per-category flip: queue a rename with only that bit flipped
// We store queued as "stem->newStem" map for multi-bit support
const pendingFlips = {}; // stem -> {cat: newStem, ...}

async function flipCategory(cat) {
  const name = images[idx];
  const s    = stem(name);
  const newStem = flipCatInStem(s, cat);
  if (!newStem) { addLog("Could not parse filename", "err"); return; }

  await fetch("/queue_flip", { method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({stem: s, cat, new_stem: newStem}) });

  queued.add(s);
  updateQueueCount();
  addLog(`Queued [${cat}] flip: ${s} → ${newStem}`, "entry");
  load(idx);
}

document.querySelectorAll(".flip-btn").forEach(btn => {
  btn.addEventListener("click", () => flipCategory(btn.dataset.cat));
});

btnChange.addEventListener("click", async () => {
  await flipCategory("background");
});

btnCommit.addEventListener("click", async () => {
  if (queued.size === 0) { addLog("Nothing queued.", ""); return; }
  const res  = await fetch("/commit", { method: "POST" });
  const data = await res.json();
  data.renamed.forEach(r => addLog("✓ " + r, "ok"));
  data.errors.forEach(e  => addLog("✗ " + e, "err"));
  queued.clear();
  updateQueueCount();
  // Reload page so file list reflects new names
  setTimeout(() => location.reload(), 800);
});

document.getElementById("btn-prev").addEventListener("click", () => load(idx - 1));
document.getElementById("btn-next").addEventListener("click", () => load(idx + 1));

document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "ArrowLeft")  load(idx - 1);
  if (e.key === "ArrowRight") load(idx + 1);
  const keyMap = { i: "item", l: "light", d: "distance", b: "background", s: "sensor" };
  const cat = keyMap[e.key.toLowerCase()];
  if (cat) flipCategory(cat);
});

load(0);
updateQueueCount();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting reviewer at http://localhost:5000")
    print(f"  Images : {IMAGES_DIR}")
    print(f"  Labels : {LABELS_DIR}")
    app.run(debug=False, port=5000)