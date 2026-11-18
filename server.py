"""HTTP API for Mithi v2 Flash.

GET  /health            -> {"status": "ok"}
GET  /voices            -> languages and voices
POST /tts               -> audio/wav   body: {"text": "...", "lang": "bn", "voice": "ahana", "seed": 0}
GET  /tts?text=..&lang=..&voice=..   -> audio/wav (handy for a browser)

POST /key/rotate        -> {"status": "ok"}  body: {"new_key": "..."}  (needs the current key)

/voices, /tts and /key/rotate need `Authorization: Bearer <key>` (the demo page asks for it);
/health stays open for container health checks.

The key is the server's own — never an environment variable. On first start it is generated,
saved in $MITHI_DATA/api_key (a volume, so it survives rebuilds) and printed once in the logs:

    docker logs mithi-v2-flash            # or: docker exec mithi-v2-flash cat /data/api_key

Paste it into the app (SaaS admin → System Settings → Call Voice). After that the app rotates it
with /key/rotate, which both sides switch to at once.
"""
import hmac
import io
import os
import secrets
import threading

import numpy as np
import soundfile as sf
import torch
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

torch.set_num_threads(int(os.environ.get("MITHI_THREADS", "4")))

from run import SR, Mithi  # noqa: E402

tts = Mithi(os.path.dirname(os.path.abspath(__file__)), device="cpu")
_lock = threading.Lock()  # one synthesis at a time; torch already uses all threads given
app = FastAPI(title="Mithi v2 Flash TTS")
KEY_FILE = os.path.join(os.environ.get("MITHI_DATA", "/data"), "api_key")


def _write_key(key: str) -> None:
    os.makedirs(os.path.dirname(KEY_FILE), exist_ok=True)
    tmp = KEY_FILE + ".tmp"
    with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
        f.write(key)
    os.replace(tmp, KEY_FILE)  # atomic: a crash never leaves half a key


def _load_key() -> str:
    try:
        with open(KEY_FILE) as f:
            key = f.read().strip()
        if key:
            return key
    except FileNotFoundError:
        pass
    key = secrets.token_hex(32)
    _write_key(key)
    print("=" * 72, flush=True)
    print("Mithi API key (generated now, saved in %s):" % KEY_FILE, flush=True)
    print("    " + key, flush=True)
    print("Paste it into SaaS admin → System Settings → Call Voice.", flush=True)
    print("=" * 72, flush=True)
    return key


API_KEY = _load_key()


def require_key(authorization: str | None = Header(default=None)):
    token = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token.encode(), API_KEY.encode()):
        raise HTTPException(401, "invalid or missing API key")


class RotateRequest(BaseModel):
    new_key: str


class TTSRequest(BaseModel):
    text: str
    lang: str = "hi"
    voice: str = "charvi"
    seed: int | None = None
    # 8000 = phone line: 120–3600 Hz band, levelled loudness, 8 kHz (what Asterisk plays as-is)
    sample_rate: int = SR


def _sinc_lowpass(cutoff, sr, taps=255):
    n = np.arange(taps) - (taps - 1) / 2
    h = np.sinc(2 * cutoff / sr * n) * np.blackman(taps)
    return (h / h.sum()).astype(np.float32)


_LP = _sinc_lowpass(3600, SR)


def _phone(audio: np.ndarray) -> np.ndarray:
    """24 kHz → 8 kHz telephone audio without ffmpeg."""
    x = np.convolve(audio, _LP, mode="same")             # anti-alias + top of the phone band
    a = np.exp(-2 * np.pi * 120 / SR)                     # one-pole high-pass at 120 Hz
    y = np.empty_like(x)
    prev_x = prev_y = 0.0
    for i, v in enumerate(x):                             # small arrays (seconds of audio): fine
        prev_y = a * (prev_y + v - prev_x)
        prev_x = v
        y[i] = prev_y
    y = y[::3]                                            # 24000 / 3 = 8000
    rms = float(np.sqrt(np.mean(y ** 2))) or 1.0
    y = y * (10 ** (-16 / 20) / rms)                      # about -16 dBFS RMS
    peak = float(np.max(np.abs(y))) or 1.0
    if peak > 0.79:                                       # keep peaks under -2 dBFS
        y = y * (0.79 / peak)
    return y.astype(np.float32)


def _wav(req: TTSRequest) -> Response:
    if len(req.text) > 5000:
        raise HTTPException(413, "text too long (max 5000 characters)")
    if req.sample_rate not in (SR, 8000):
        raise HTTPException(422, f"sample_rate must be {SR} or 8000")
    try:
        with _lock:
            audio = tts.synthesize(req.text, lang=req.lang, voice=req.voice, seed=req.seed)
    except ValueError as e:
        raise HTTPException(422, str(e))
    rate = req.sample_rate
    if rate == 8000:
        audio = _phone(audio)
    buf = io.BytesIO()
    sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav",
                    headers={"X-Audio-Seconds": f"{len(audio) / rate:.2f}"})


PAGE = """<!doctype html><html lang="bn"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Mithi TTS</title>
<style>body{font-family:system-ui,sans-serif;max-width:760px;margin:24px auto;padding:0 16px;background:#fafafa;color:#222}
textarea{width:100%;height:180px;font-size:18px;padding:10px;box-sizing:border-box}
.row{display:flex;gap:8px;margin:10px 0;flex-wrap:wrap}select,button{font-size:16px;padding:8px 12px}
button{background:#1f6feb;color:#fff;border:0;border-radius:6px;cursor:pointer}button:disabled{opacity:.5}
audio{width:100%;margin-top:10px}#s{color:#666}</style></head><body>
<h2>Mithi v2 Flash</h2>
<textarea id="t">আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।</textarea>
<div class="row"><select id="l"></select><select id="v"></select>
<input id="k" type="password" placeholder="API key" style="font-size:16px;padding:8px">
<button id="b">▶ শোনাও</button><span id="s"></span></div>
<audio id="a" controls></audio>
<script>
k.value=localStorage.getItem('mithi_key')||'';
const H=()=>{localStorage.setItem('mithi_key',k.value);return k.value?{Authorization:'Bearer '+k.value}:{}};
const loadVoices=()=>fetch('/voices',{headers:H()}).then(r=>r.json()).then(d=>{if(!d.voices)return;
 l.innerHTML=d.languages.map(x=>`<option ${x=='bn'?'selected':''}>${x}</option>`).join('');
 v.innerHTML=d.voices.map(x=>`<option ${x=='ahana'?'selected':''}>${x}</option>`).join('');});
loadVoices();k.onchange=loadVoices;
b.onclick=async()=>{b.disabled=true;s.textContent='তৈরি হচ্ছে...';const t0=Date.now();
 try{const r=await fetch('/tts',{method:'POST',headers:{'Content-Type':'application/json',...H()},
  body:JSON.stringify({text:t.value,lang:l.value,voice:v.value})});
  if(!r.ok)throw new Error((await r.json()).detail||r.status);
  a.src=URL.createObjectURL(await r.blob());a.play();
  s.textContent=`${r.headers.get('X-Audio-Seconds')}s অডিও, ${((Date.now()-t0)/1000).toFixed(1)}s এ`;}
 catch(e){s.textContent='Error: '+e.message}finally{b.disabled=false}};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/key/rotate", dependencies=[Depends(require_key)])
def rotate_key(req: RotateRequest):
    global API_KEY
    key = req.new_key.strip()
    if len(key) < 32 or not key.isalnum():
        raise HTTPException(422, "new_key must be at least 32 letters/digits")
    _write_key(key)
    API_KEY = key
    return {"status": "ok"}


@app.get("/voices", dependencies=[Depends(require_key)])
def voices():
    return {"languages": list(Mithi.LANGS), "voices": tts.voices, "sample_rate": SR}


@app.post("/tts", dependencies=[Depends(require_key)])
def tts_post(req: TTSRequest):
    return _wav(req)


@app.get("/tts", dependencies=[Depends(require_key)])
def tts_get(text: str = Query(...), lang: str = "hi", voice: str = "charvi", seed: int | None = None):
    return _wav(TTSRequest(text=text, lang=lang, voice=voice, seed=seed))
