"""The recording page of `presenter.py` (one self-contained page, served on this Mac only)."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EI · Record your lines</title>
<style>
  :root { --bg:#14161a; --panel:#1d2026; --line:#2c313a; --text:#e9ebef; --dim:#9aa2af; --accent:#4f8cff; --rec:#ff4d4d; --ok:#3ecf8e; --warn:#f5b942; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:15px/1.45 -apple-system, "Helvetica Neue", Arial, sans-serif; }
  button, select { font:inherit; color:inherit; }
  .wrap { display:grid; grid-template-columns:minmax(0,1fr) 320px; gap:16px; max-width:1280px; margin:0 auto; padding:16px; min-height:100vh; }
  @media (max-width:900px) { .wrap { grid-template-columns:1fr; } }
  .main { display:flex; flex-direction:column; gap:12px; min-width:0; }
  .prompt { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 18px; text-align:center; }
  .prompt .where { color:var(--dim); font-size:13px; }
  .prompt .say { font-size:clamp(22px,3.2vw,34px); font-weight:650; line-height:1.25; margin:6px 0 4px; text-wrap:balance; }
  .prompt .hint { color:var(--dim); font-size:13px; min-height:1.4em; }
  .stage { position:relative; background:#000; border-radius:12px; overflow:hidden; aspect-ratio:16/9; border:2px solid var(--line); }
  .stage.rec { border-color:var(--rec); }
  .stage video { width:100%; height:100%; object-fit:cover; display:block; }
  .stage video.mirror { transform:scaleX(-1); }
  .badge { position:absolute; top:10px; left:10px; background:rgba(0,0,0,.6); padding:4px 10px; border-radius:99px; font-size:13px; font-variant-numeric:tabular-nums; }
  .badge.rec::before { content:""; display:inline-block; width:9px; height:9px; border-radius:50%; background:var(--rec); margin-right:6px; }
  .count { position:absolute; inset:0; display:none; align-items:center; justify-content:center; font-size:120px; font-weight:700; background:rgba(0,0,0,.35); }
  .level { position:absolute; left:10px; right:10px; bottom:10px; height:5px; background:rgba(255,255,255,.15); border-radius:3px; overflow:hidden; }
  .level i { display:block; height:100%; width:0; background:var(--ok); }
  .bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  .btn { background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:9px 14px; cursor:pointer; }
  .btn:hover { border-color:#4a5160; }
  .btn:disabled { opacity:.4; cursor:default; }
  .btn.primary { background:var(--accent); border-color:var(--accent); color:#fff; font-weight:600; }
  .btn.rec { background:var(--rec); border-color:var(--rec); color:#fff; font-weight:600; }
  .btn kbd { opacity:.65; font:12px ui-monospace, Menlo, monospace; margin-left:6px; }
  .spacer { flex:1; }
  .opts { display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; color:var(--dim); font-size:13px; }
  .opts select { background:var(--panel); border:1px solid var(--line); border-radius:7px; padding:5px 8px; max-width:230px; }
  .opts label { display:flex; gap:6px; align-items:center; }
  .side { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px; display:flex; flex-direction:column; gap:10px; max-height:calc(100vh - 32px); position:sticky; top:16px; }
  .side h2 { font-size:14px; margin:0; }
  .progress { height:6px; background:var(--line); border-radius:3px; overflow:hidden; }
  .progress i { display:block; height:100%; background:var(--ok); width:0; }
  .list { overflow:auto; display:flex; flex-direction:column; gap:4px; flex:1; min-height:120px; }
  .item { display:grid; grid-template-columns:22px 1fr auto; gap:8px; padding:7px 8px; border-radius:8px; cursor:pointer; font-size:13px; border:1px solid transparent; }
  .item:hover { background:#252931; }
  .item.cur { border-color:var(--accent); background:#222733; }
  .item .dot { width:18px; height:18px; border-radius:50%; border:2px solid var(--line); margin-top:1px; font-size:11px; display:flex; align-items:center; justify-content:center; }
  .item.done .dot { background:var(--ok); border-color:var(--ok); color:#06301f; }
  .item .t { overflow:hidden; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; }
  .item .s { color:var(--dim); font-variant-numeric:tabular-nums; }
  .note { color:var(--dim); font-size:12.5px; }
  .err { color:var(--rec); }
</style>
</head>
<body>
<div class="wrap">
  <div class="main">
    <div class="prompt">
      <div class="where" id="where">&nbsp;</div>
      <div class="say" id="say">Loading…</div>
      <div class="hint" id="hint"></div>
    </div>
    <div class="stage" id="stage">
      <video id="live" class="mirror" autoplay muted playsinline></video>
      <video id="review" playsinline controls style="display:none"></video>
      <div class="badge" id="badge">camera</div>
      <div class="count" id="count"></div>
      <div class="level"><i id="level"></i></div>
    </div>
    <div class="bar">
      <button class="btn" id="listen">▶ Hear the voice<kbd>L</kbd></button>
      <button class="btn rec" id="record">● Record<kbd>Space</kbd></button>
      <button class="btn primary" id="keep" disabled>Keep &amp; next<kbd>Enter</kbd></button>
      <button class="btn" id="retake" disabled>Retake<kbd>R</kbd></button>
      <span class="spacer"></span>
      <button class="btn" id="prev">← Back</button>
      <button class="btn" id="next">Skip →</button>
    </div>
    <div class="opts">
      <label>Camera <select id="cam"></select></label>
      <label>Microphone <select id="mic"></select></label>
      <label><input type="checkbox" id="hearfirst" checked> Play the voice before each take</label>
      <label title="Use headphones, so the voice is not recorded with you"><input type="checkbox" id="along"> Play the voice while I record (headphones)</label>
    </div>
    <div class="note" id="status">Only your picture is used in the video. Your voice is recorded just to find where each word falls.</div>
  </div>
  <div class="side">
    <h2 id="title">Lines</h2>
    <div class="progress"><i id="prog"></i></div>
    <div class="note" id="sum"></div>
    <div class="list" id="list"></div>
    <button class="btn" id="idle">Record the idle look (10 s, no talking)</button>
    <button class="btn primary" id="done">Done</button>
  </div>
</div>
<script>
const K = new URLSearchParams(location.search).get("k") || "";
const api = (p, extra = "") => `${p}?k=${encodeURIComponent(K)}${extra}`;
const $ = id => document.getElementById(id);
let S = null, cur = 0, stream = null, rec = null, chunks = [], blob = null, blobSeconds = 0, t0 = 0, timer = null, mode = "line", busy = false;
const voice = new Audio();
const IDLE_SECONDS = 10, MAX_SECONDS = 90;

function fmt(s) { return s.toFixed(1) + " s"; }

async function load() {
  S = await (await fetch(api("/api/session"))).json();
  $("title").textContent = S.video;
  const first = S.lines.findIndex(l => !l.take);
  cur = first < 0 ? 0 : first;
  draw();
}

function draw() {
  const l = S.lines[cur];
  const done = S.lines.filter(x => x.take).length;
  $("prog").style.width = (100 * done / S.lines.length) + "%";
  $("sum").textContent = `${done} of ${S.lines.length} recorded · preview possible until ${fmt(S.recordedUntil)}` + (S.idle ? " · idle look ✓" : "");
  $("idle").textContent = S.idle ? "Record the idle look again" : "Record the idle look (10 s, no talking)";
  if (mode === "idle") {
    $("where").textContent = "Idle look";
    $("say").textContent = "Just look at the camera. Don't talk.";
    $("hint").textContent = "Breathe, blink, small natural movements. It is shown where nothing is being said. It stops by itself after 10 s.";
  } else {
    $("where").textContent = `Line ${cur + 1} of ${S.lines.length} · the voice takes ${fmt(l.seconds)}` + (l.emotion ? ` · ${l.emotion}` : "");
    $("say").textContent = l.text;
    $("hint").textContent = l.take ? `Recorded (take ${l.take.n}${l.take.seconds ? ", " + fmt(l.take.seconds) : ""}). Record again to replace it.` : "Say it at the voice's pace, looking at the camera.";
  }
  const list = $("list");
  list.innerHTML = "";
  S.lines.forEach((x, i) => {
    const d = document.createElement("div");
    d.className = "item" + (x.take ? " done" : "") + (i === cur && mode === "line" ? " cur" : "");
    d.innerHTML = `<span class="dot">${x.take ? "✓" : ""}</span><span class="t"></span><span class="s">${fmt(x.seconds)}</span>`;
    d.querySelector(".t").textContent = x.text;
    d.onclick = () => { if (!recording()) go(i); };
    list.appendChild(d);
  });
  const c = list.querySelector(".cur");
  if (c) c.scrollIntoView({ block: "nearest" });
  buttons();
}

function recording() { return rec && rec.state === "recording"; }

function buttons() {
  const r = recording();
  $("record").textContent = r ? "■ Stop" : "● Record";
  $("record").insertAdjacentHTML("beforeend", "<kbd>Space</kbd>");
  $("record").disabled = busy || !stream;
  $("keep").disabled = !blob || busy || r;
  $("retake").disabled = !blob || busy || r;
  $("listen").disabled = r || mode === "idle";
  $("prev").disabled = r;
  $("next").disabled = r;
  $("stage").classList.toggle("rec", !!r);
}

function showLive() {
  $("review").pause(); $("review").style.display = "none"; $("live").style.display = "block";
  $("badge").className = "badge"; $("badge").textContent = "camera";
}

function go(i) {
  mode = "line"; blob = null; voice.pause();
  cur = Math.max(0, Math.min(S.lines.length - 1, i));
  showLive(); draw();
}

function hear() {
  return new Promise(res => {
    voice.src = api(`/api/audio/${cur}`, `&t=${Date.now()}`);
    voice.onended = voice.onerror = () => res();
    voice.play().catch(() => res());
  });
}

function countdown() {
  return new Promise(res => {
    const c = $("count"); let n = 3;
    c.style.display = "flex"; c.textContent = n;
    const t = setInterval(() => { n--; if (n <= 0) { clearInterval(t); c.style.display = "none"; res(); } else c.textContent = n; }, 600);
  });
}

function mime() {
  for (const m of ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm", "video/mp4"]) if (MediaRecorder.isTypeSupported(m)) return m;
  return "";
}

async function start() {
  if (!stream || busy || recording()) return;
  busy = true; blob = null; showLive(); buttons();
  if (mode === "line" && $("hearfirst").checked && !$("along").checked) await hear();
  await countdown();
  chunks = [];
  const m = mime();
  rec = new MediaRecorder(stream, { mimeType: m || undefined, videoBitsPerSecond: 6000000, audioBitsPerSecond: 128000 });
  rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
  rec.onstop = () => {
    clearInterval(timer);
    blobSeconds = (performance.now() - t0) / 1000;
    blob = new Blob(chunks, { type: rec.mimeType || m || "video/webm" });
    const v = $("review");
    v.src = URL.createObjectURL(blob); v.style.display = "block"; $("live").style.display = "none"; v.play().catch(() => {});
    $("badge").className = "badge"; $("badge").textContent = "your take · " + fmt(blobSeconds);
    if (mode === "line") {
      const want = S.lines[cur].seconds, off = blobSeconds / want;
      $("hint").textContent = off > 1.6 ? `Your take is much longer than the voice (${fmt(blobSeconds)} against ${fmt(want)}). A retake nearer its pace will sync better.` :
        off < 0.6 ? `Your take is much shorter than the voice (${fmt(blobSeconds)} against ${fmt(want)}). Was the whole line said?` : "Looks fine? Keep it, or retake.";
    }
    buttons();
  };
  rec.start(250);
  t0 = performance.now(); busy = false;
  if (mode === "line" && $("along").checked) hear();
  $("badge").className = "badge rec";
  timer = setInterval(() => {
    const s = (performance.now() - t0) / 1000;
    $("badge").textContent = fmt(s);
    if (s >= (mode === "idle" ? IDLE_SECONDS : MAX_SECONDS)) stop();
  }, 100);
  buttons();
}

function stop() { if (recording()) { voice.pause(); rec.stop(); } }

async function keep() {
  if (!blob || busy) return;
  busy = true; buttons();
  $("status").className = "note"; $("status").textContent = "Saving the take…";
  const ext = blob.type.includes("mp4") ? "mp4" : "webm";
  const which = mode === "idle" ? "idle" : String(cur);
  try {
    const r = await fetch(api(`/api/take/${which}`, `&ext=${ext}&seconds=${blobSeconds.toFixed(2)}`), { method: "POST", body: blob });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || r.status);
    S = j.session;
    $("status").textContent = `Saved ${j.take.file}.`;
    blob = null; busy = false;
    if (mode === "idle") { mode = "line"; go(cur); }
    else { const nxt = S.lines.findIndex((l, i) => i > cur && !l.take); go(nxt < 0 ? Math.min(cur + 1, S.lines.length - 1) : nxt); }
  } catch (e) {
    busy = false; $("status").className = "note err"; $("status").textContent = "The take was not saved: " + e.message + ". It is still here: try Keep again."; buttons();
  }
}

function retake() { if (!blob || busy) return; blob = null; showLive(); draw(); start(); }

async function camera() {
  const cam = $("cam").value, mic = $("mic").value;
  if (stream) stream.getTracks().forEach(t => t.stop());
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: cam ? { exact: cam } : undefined, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      audio: { deviceId: mic ? { exact: mic } : undefined, echoCancellation: false, noiseSuppression: false, autoGainControl: true },
    });
  } catch (e) {
    stream = null; $("status").className = "note err";
    $("status").textContent = "The camera or microphone could not be opened (" + e.message + "). Allow them for this page in the browser, then reload.";
    buttons(); return;
  }
  $("live").srcObject = stream;
  const devs = await navigator.mediaDevices.enumerateDevices();
  for (const [sel, kind, track] of [[$("cam"), "videoinput", stream.getVideoTracks()[0]], [$("mic"), "audioinput", stream.getAudioTracks()[0]]]) {
    const now = track && track.getSettings().deviceId;
    sel.innerHTML = "";
    devs.filter(d => d.kind === kind).forEach(d => { const o = document.createElement("option"); o.value = d.deviceId; o.textContent = d.label || kind; o.selected = d.deviceId === now; sel.appendChild(o); });
  }
  meter(); buttons();
}

let actx = null;
function meter() {
  try {
    if (actx) actx.close();
    actx = new AudioContext();
    const an = actx.createAnalyser(); an.fftSize = 512;
    actx.createMediaStreamSource(stream).connect(an);
    const buf = new Uint8Array(an.fftSize);
    (function tick() {
      if (!actx || actx.state === "closed") return;
      an.getByteTimeDomainData(buf);
      let m = 0; for (const b of buf) m = Math.max(m, Math.abs(b - 128));
      $("level").style.width = Math.min(100, m * 1.6) + "%";
      requestAnimationFrame(tick);
    })();
  } catch (e) {}
}

$("listen").onclick = () => hear();
$("record").onclick = () => recording() ? stop() : start();
$("keep").onclick = keep;
$("retake").onclick = retake;
$("prev").onclick = () => go(cur - 1);
$("next").onclick = () => go(cur + 1);
$("idle").onclick = () => { if (recording()) return; mode = "idle"; blob = null; showLive(); draw(); };
$("cam").onchange = camera; $("mic").onchange = camera;
$("done").onclick = async () => {
  if (recording()) return;
  await fetch(api("/api/done"), { method: "POST" }).catch(() => {});
  if (stream) stream.getTracks().forEach(t => t.stop());
  document.body.innerHTML = '<div style="padding:60px;text-align:center;font-size:20px">Saved. You can close this tab and go back to EI.</div>';
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); recording() ? stop() : start(); }
  else if (e.key === "Enter") { e.preventDefault(); keep(); }
  else if (e.key === "r" || e.key === "R") retake();
  else if (e.key === "l" || e.key === "L") { if (!recording()) hear(); }
  else if (e.key === "ArrowLeft" && !recording()) go(cur - 1);
  else if (e.key === "ArrowRight" && !recording()) go(cur + 1);
});
load().then(camera).catch(e => { $("say").textContent = "This page could not load its lines."; $("status").className = "note err"; $("status").textContent = String(e); });
</script>
</body>
</html>
"""
