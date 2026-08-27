"""
app.py — Smart Voice Gate (Streamlit Dashboard)

A voice-controlled gate system combining two AI techniques:
  1. Speaker Verification (Voice Biometrics) — verifies WHO is speaking
  2. Speech Recognition (NLP) — verifies WHAT is being said

The gate opens only when both the speaker identity and the
voice command (passphrase) are verified correctly.
"""

import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import base64
from datetime import datetime

# pyrefly: ignore [missing-import]
import soundfile as sf
import io

# pyrefly: ignore [missing-import]
from gtts import gTTS

from voice_embedding import extract_embedding, compare_embeddings
from speech_recognition_module import transcribe_audio, check_passphrase
from gate_logic import verify_access
from user_manager import enroll_user, get_enrolled_users, get_user_names, delete_user, get_user_passphrase, update_user


# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Voice Gate",
    page_icon="🔐",
    layout="wide"
)

# ──────────────────────────────────────────────
# Session State Init — Alerts Log
# ──────────────────────────────────────────────
if "alerts_log" not in st.session_state:
    st.session_state.alerts_log = []

# ──────────────────────────────────────────────
# Premium CSS — Glassmorphism Dark Theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ── Global ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .hero-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .hero-header h1 {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }
    .hero-header .subtitle {
        font-size: 1.05rem;
        color: #9ca3af;
        font-weight: 400;
    }
    .hero-header .badges {
        margin-top: 0.8rem;
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }
    .badge-voice {
        background: rgba(102, 126, 234, 0.15);
        color: #667eea;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .badge-nlp {
        background: rgba(118, 75, 162, 0.15);
        color: #a78bfa;
        border: 1px solid rgba(118, 75, 162, 0.3);
    }

    /* ── Status Banners ── */
    .status-box {
        padding: 1.5rem 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1.2rem 0;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        backdrop-filter: blur(12px);
    }
    .access-granted {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(52, 211, 153, 0.08));
        color: #10b981;
        border: 2px solid rgba(16, 185, 129, 0.4);
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.1);
    }
    .access-denied {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(248, 113, 113, 0.08));
        color: #ef4444;
        border: 2px solid rgba(239, 68, 68, 0.4);
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.1);
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        margin: 0.5rem 0;
    }
    .glass-card h4 {
        margin-top: 0;
        color: #a78bfa;
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .glass-card ul {
        padding-left: 1.2rem;
        color: #d1d5db;
        line-height: 1.9;
    }

    /* ── User Row Card (Manage Users) ── */
    .user-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        transition: all 0.2s ease;
    }
    .user-row:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(102, 126, 234, 0.3);
    }
    .user-info {
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .user-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        font-weight: 700;
        color: white;
    }
    .user-name {
        font-weight: 600;
        font-size: 1rem;
        color: #e5e7eb;
    }
    .user-phrase {
        color: #9ca3af;
        font-size: 0.85rem;
        font-style: italic;
    }

    /* ── Stats Row ── */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .stat-card {
        flex: 1;
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.2rem;
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.8rem;
        padding: 1.5rem 0 0.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 2rem;
    }
    .footer span {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }

    /* ── Alert Row Card ── */
    .alert-row {
        display: flex;
        align-items: flex-start;
        gap: 1rem;
        padding: 1rem 1.4rem;
        margin: 0.6rem 0;
        border-radius: 14px;
        background: rgba(239, 68, 68, 0.06);
        border: 1px solid rgba(239, 68, 68, 0.25);
        transition: all 0.2s ease;
    }
    .alert-row:hover {
        background: rgba(239, 68, 68, 0.10);
        border-color: rgba(239, 68, 68, 0.45);
    }
    .alert-icon { font-size: 1.6rem; flex-shrink: 0; margin-top: 0.1rem; }
    .alert-body { flex: 1; }
    .alert-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #f87171;
        margin-bottom: 0.25rem;
    }
    .alert-detail {
        font-size: 0.82rem;
        color: #9ca3af;
        line-height: 1.7;
    }
    .alert-detail strong { color: #d1d5db; }
    .alert-time {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }
    .alert-badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-wrong-voice {
        background: rgba(251, 146, 60, 0.15);
        color: #fb923c;
        border: 1px solid rgba(251, 146, 60, 0.3);
    }
    .badge-wrong-pass {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-both-failed {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .alerts-empty {
        text-align: center;
        padding: 3rem 1rem;
        color: #6b7280;
        font-size: 1rem;
    }
    .alerts-empty .empty-icon { font-size: 3rem; display: block; margin-bottom: 0.8rem; }

    /* ── Gate shake animation for denial ── */
    @keyframes shake {
        0%   { transform: translateX(0); }
        20%  { transform: translateX(-6px); }
        40%  { transform: translateX(6px); }
        60%  { transform: translateX(-4px); }
        80%  { transform: translateX(4px); }
        100% { transform: translateX(0); }
    }
    @keyframes lockPulse {
        from { opacity: 0.7; transform: scale(1);   }
        to   { opacity: 1;   transform: scale(1.12); }
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🔐 Smart Voice Gate</h1>
    <p class="subtitle">AI-powered access control combining voice biometrics and speech recognition</p>
    <div class="badges">
        <span class="badge badge-voice">🎙️ Speaker Verification (ECAPA-TDNN)</span>
        <span class="badge badge-nlp">🧠 Speech Recognition (NLP)</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Quick Stats Row ──
enrolled_count = len(get_user_names())
alert_count = len(st.session_state.alerts_log)
st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-value">{enrolled_count}</div>
        <div class="stat-label">Enrolled Users</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">2</div>
        <div class="stat-label">AI Techniques</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">{alert_count}</div>
        <div class="stat-label">Security Alerts</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper: Save Uploaded Audio to Temp WAV File
# ──────────────────────────────────────────────
def save_audio_to_wav(audio_bytes):
    """Convert uploaded audio bytes to a WAV file and return the path."""
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
    temp_path = os.path.join(tempfile.gettempdir(), "voice_gate_temp.wav")
    sf.write(temp_path, audio_data, sample_rate)
    return temp_path


def text_to_speech_audio(text):
    """Convert text to speech audio bytes using gTTS."""
    tts = gTTS(text=text, lang="en", slow=True)
    audio_buf = io.BytesIO()
    tts.write_to_fp(audio_buf)
    audio_buf.seek(0)
    return audio_buf


# ──────────────────────────────────────────────
# Helper: Determine failed metric label
# ──────────────────────────────────────────────
def get_fail_reason(speaker_matched, command_matched):
    if not speaker_matched and not command_matched:
        return "Both Voice & Password Failed"
    elif not speaker_matched:
        return "Wrong Voice"
    else:
        return "Wrong Password"


def get_fail_badge_class(reason):
    if "Both" in reason:
        return "badge-both-failed"
    elif "Voice" in reason:
        return "badge-wrong-voice"
    else:
        return "badge-wrong-pass"


# ──────────────────────────────────────────────
# Gate Opening Animation (HTML + Web Audio API)
# ──────────────────────────────────────────────
GATE_OPEN_ANIMATION = """
<div style="display:flex; flex-direction:column; align-items:center; margin:1.5rem 0;">
  <p style="color:#10b981; font-weight:700; font-size:1rem; margin-bottom:0.8rem; letter-spacing:1px;">
    ✅ GATE OPENING…
  </p>
  <div id="gate-container" style="
    width: 260px;
    height: 200px;
    position: relative;
    perspective: 600px;
    border-radius: 8px;
    overflow: hidden;
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border: 2px solid rgba(16,185,129,0.4);
    box-shadow: 0 0 40px rgba(16,185,129,0.15);
  ">
    <!-- Gate frame -->
    <div style="
      position:absolute; inset:0;
      border: 6px solid #334155;
      border-radius: 6px;
      z-index: 5;
      pointer-events:none;
    "></div>
    <!-- Ground -->
    <div style="
      position:absolute; bottom:0; left:0; right:0;
      height:22px;
      background: linear-gradient(180deg, #334155, #1e293b);
      border-top: 2px solid #475569;
    "></div>
    <!-- Left gate panel -->
    <div id="gate-left" style="
      position:absolute; top:0; left:0;
      width:50%; height:calc(100% - 22px);
      background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 40%, #1d4ed8 100%);
      border-right: 3px solid #3b82f6;
      transform-origin: left center;
      transform: rotateY(0deg);
      transition: transform 1.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
      z-index: 3;
      display:flex; align-items:center; justify-content:center;
    ">
      <div style="width:8px; height:70%;
        background: repeating-linear-gradient(180deg, #3b82f6 0px, #1d4ed8 12px, #2563eb 24px);
        border-radius:4px; opacity:0.7;"></div>
      <div style="position:absolute; right:10px; top:50%;
        width:10px; height:10px; border-radius:50%;
        background:radial-gradient(circle,#fbbf24,#d97706);
        box-shadow:0 0 8px #fbbf24; transform:translateY(-50%);"></div>
    </div>
    <!-- Right gate panel -->
    <div id="gate-right" style="
      position:absolute; top:0; right:0;
      width:50%; height:calc(100% - 22px);
      background: linear-gradient(225deg, #1e3a5f 0%, #1e40af 40%, #1d4ed8 100%);
      border-left: 3px solid #3b82f6;
      transform-origin: right center;
      transform: rotateY(0deg);
      transition: transform 1.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
      z-index: 3;
      display:flex; align-items:center; justify-content:center;
    ">
      <div style="width:8px; height:70%;
        background: repeating-linear-gradient(180deg, #3b82f6 0px, #1d4ed8 12px, #2563eb 24px);
        border-radius:4px; opacity:0.7;"></div>
      <div style="position:absolute; left:10px; top:50%;
        width:10px; height:10px; border-radius:50%;
        background:radial-gradient(circle,#fbbf24,#d97706);
        box-shadow:0 0 8px #fbbf24; transform:translateY(-50%);"></div>
    </div>
    <!-- Glow overlay -->
    <div id="gate-glow" style="
      position:absolute; inset:0;
      background: radial-gradient(ellipse at center, rgba(16,185,129,0.0) 0%, transparent 70%);
      z-index:2;
      transition: background 1.2s ease 0.8s;
    "></div>
  </div>
  <p id="gate-status-text" style="color:#6b7280; font-size:0.85rem; margin-top:0.7rem;">Closed — authenticating…</p>
</div>
<script>
(function() {
  function playGateSound() {
    try {
      var AudioCtx = window.AudioContext || window.webkitAudioContext;
      var ctx = new AudioCtx();
      function beep(freq, startTime, duration, type, gainVal) {
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        osc.connect(gain); gain.connect(ctx.destination);
        osc.type = type || 'sawtooth';
        osc.frequency.setValueAtTime(freq, startTime);
        osc.frequency.linearRampToValueAtTime(freq * 0.3, startTime + duration);
        gain.gain.setValueAtTime(gainVal || 0.3, startTime);
        gain.gain.exponentialRampToValueAtTime(0.001, startTime + duration);
        osc.start(startTime); osc.stop(startTime + duration);
      }
      var t = ctx.currentTime;
      beep(180, t, 0.4, 'sawtooth', 0.25);
      beep(90, t+0.1, 0.6, 'square', 0.18);
      beep(140, t+0.35, 0.5, 'sawtooth', 0.20);
      beep(60, t+0.55, 0.8, 'sawtooth', 0.15);
      beep(45, t+0.8, 0.9, 'square', 0.12);
      var buf = ctx.createBuffer(1, ctx.sampleRate * 0.15, ctx.sampleRate);
      var data = buf.getChannelData(0);
      for (var i = 0; i < data.length; i++) {
        data[i] = (Math.random()*2-1) * Math.pow(1 - i/data.length, 3);
      }
      var src = ctx.createBufferSource(); src.buffer = buf;
      var g2 = ctx.createGain(); g2.gain.value = 0.5;
      src.connect(g2); g2.connect(ctx.destination);
      src.start(t + 1.0);
    } catch(e) { console.warn('Gate sound failed:', e); }
  }
  function openGate() {
    var left  = document.getElementById('gate-left');
    var right = document.getElementById('gate-right');
    var glow  = document.getElementById('gate-glow');
    var txt   = document.getElementById('gate-status-text');
    if (left)  left.style.transform  = 'rotateY(-88deg)';
    if (right) right.style.transform = 'rotateY(88deg)';
    if (glow)  glow.style.background = 'radial-gradient(ellipse at center, rgba(16,185,129,0.18) 0%, transparent 70%)';
    if (txt) setTimeout(function(){
      txt.textContent = '\u2705 Gate Opened \u2014 Access Granted!';
      txt.style.color = '#10b981';
    }, 1000);
  }
  setTimeout(function() { playGateSound(); openGate(); }, 150);
})();
</script>
"""

# ──────────────────────────────────────────────
# Gate Locked / Denied Animation
# ──────────────────────────────────────────────
GATE_CLOSED_ANIMATION = """
<style>
  @keyframes shake {
    0%   { transform: translateX(0); }
    20%  { transform: translateX(-6px); }
    40%  { transform: translateX(6px); }
    60%  { transform: translateX(-4px); }
    80%  { transform: translateX(4px); }
    100% { transform: translateX(0); }
  }
  @keyframes lockPulse {
    from { opacity: 0.7; transform: scale(1); }
    to   { opacity: 1;   transform: scale(1.12); }
  }
</style>
<div style="display:flex; flex-direction:column; align-items:center; margin:1.5rem 0;">
  <p style="color:#ef4444; font-weight:700; font-size:1rem; margin-bottom:0.8rem; letter-spacing:1px;">
    ❌ ACCESS DENIED — Gate Locked
  </p>
  <div style="
    width: 260px; height: 200px;
    position: relative; border-radius: 8px; overflow: hidden;
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border: 2px solid rgba(239,68,68,0.4);
    box-shadow: 0 0 40px rgba(239,68,68,0.15);
  ">
    <div style="position:absolute; inset:0; border: 6px solid #334155; border-radius:6px; z-index:5; pointer-events:none;"></div>
    <div style="position:absolute; bottom:0; left:0; right:0; height:22px;
      background: linear-gradient(180deg, #334155, #1e293b); border-top: 2px solid #475569;"></div>
    <!-- Left panel (red, shaking) -->
    <div style="
      position:absolute; top:0; left:0; width:50%; height:calc(100% - 22px);
      background: linear-gradient(135deg, #3b1a1a 0%, #7f1d1d 40%, #991b1b 100%);
      border-right: 3px solid #ef4444;
      display:flex; align-items:center; justify-content:center;
      animation: shake 0.5s ease 0.1s 2;
    ">
      <div style="width:8px; height:70%;
        background: repeating-linear-gradient(180deg,#ef4444 0px,#7f1d1d 12px,#b91c1c 24px);
        border-radius:4px; opacity:0.7;"></div>
      <div style="position:absolute; right:10px; top:50%;
        width:12px; height:12px; border-radius:50%;
        background:radial-gradient(circle,#ef4444,#7f1d1d);
        box-shadow:0 0 10px #ef4444; transform:translateY(-50%);"></div>
    </div>
    <!-- Right panel (red, shaking) -->
    <div style="
      position:absolute; top:0; right:0; width:50%; height:calc(100% - 22px);
      background: linear-gradient(225deg, #3b1a1a 0%, #7f1d1d 40%, #991b1b 100%);
      border-left: 3px solid #ef4444;
      display:flex; align-items:center; justify-content:center;
      animation: shake 0.5s ease 0.1s 2;
    ">
      <div style="width:8px; height:70%;
        background: repeating-linear-gradient(180deg,#ef4444 0px,#7f1d1d 12px,#b91c1c 24px);
        border-radius:4px; opacity:0.7;"></div>
      <div style="position:absolute; left:10px; top:50%;
        width:12px; height:12px; border-radius:50%;
        background:radial-gradient(circle,#ef4444,#7f1d1d);
        box-shadow:0 0 10px #ef4444; transform:translateY(-50%);"></div>
    </div>
    <!-- Lock icon -->
    <div style="position:absolute; inset:0; z-index:4;
      display:flex; align-items:center; justify-content:center; pointer-events:none;">
      <div style="font-size:2.2rem; animation: lockPulse 1s ease-in-out infinite alternate;">🔒</div>
    </div>
  </div>
  <p style="color:#f87171; font-size:0.85rem; margin-top:0.7rem;">Gate remains locked.</p>
</div>
<script>
(function() {
  function playAlertSound() {
    try {
      var AudioCtx = window.AudioContext || window.webkitAudioContext;
      var ctx = new AudioCtx();
      function buzz(freq, start, dur, gain) {
        var osc = ctx.createOscillator();
        var g   = ctx.createGain();
        osc.connect(g); g.connect(ctx.destination);
        osc.type = 'square';
        osc.frequency.setValueAtTime(freq, start);
        g.gain.setValueAtTime(gain, start);
        g.gain.exponentialRampToValueAtTime(0.001, start + dur);
        osc.start(start); osc.stop(start + dur);
      }
      var t = ctx.currentTime;
      buzz(880, t,        0.18, 0.4);
      buzz(660, t + 0.25, 0.18, 0.4);
      buzz(440, t + 0.50, 0.35, 0.4);
    } catch(e) { console.warn('Alert sound failed:', e); }
  }
  setTimeout(playAlertSound, 150);
})();
</script>
"""


# ──────────────────────────────────────────────
# Tabs: Verify / Enroll / Manage / Alerts
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "\U0001f510 Verify & Unlock",
    "\U0001f4cb Enroll Voice",
    "\U0001f465 Manage Users",
    "\U0001f6a8 Security Alerts"
])


# ══════════════════════════════════════════════
# TAB 1: VERIFY & UNLOCK GATE  (shown first)
# ══════════════════════════════════════════════
with tab1:
    st.subheader("🔐 Verify Identity & Unlock Gate")
    st.markdown("Select your name, then record your voice saying **your passphrase**. The system verifies both your identity and your command.")

    # Check if any users are enrolled
    enrolled = get_enrolled_users()
    user_names_list = get_user_names()

    if len(enrolled) == 0:
        st.warning("⚠️ No users enrolled yet! Go to the **Enroll Voice** tab first.")
    else:
        col_v1, col_v2 = st.columns([2, 1])

        with col_v1:
            # User selects their name
            selected_user = st.selectbox(
                "👤 Select your name:",
                options=user_names_list,
                key="verify_user_select"
            )

            # Show the user's passphrase hint (masked)
            if selected_user:
                stored_passphrase = get_user_passphrase(selected_user)
                if stored_passphrase:
                    masked = stored_passphrase[0] + "•" * (len(stored_passphrase) - 1)
                    st.caption(f"🔑 Passphrase hint: **{masked}**")

            # Audio recorder for verification
            verify_audio = st.audio_input(
                "🎤 Record your voice (say your passphrase):",
                key="verify_audio"
            )

        with col_v2:
            st.markdown("""
            <div class="glass-card">
                <h4>🛡️ How It Works</h4>
                <ul>
                    <li>Select your enrolled name</li>
                    <li>Record yourself saying <b>your passphrase</b></li>
                    <li>AI checks <b>WHO</b> you are (voice match)</li>
                    <li>AI checks <b>WHAT</b> you said (phrase match)</li>
                    <li>Both must pass to unlock! 🔓</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🔓 Verify & Unlock", type="primary", use_container_width=True):
            if not verify_audio:
                st.error("❌ Please record your voice first!")
            elif not selected_user:
                st.error("❌ Please select your name!")
            else:
                with st.spinner("🔄 Verifying identity and command..."):
                    try:
                        # Save audio to temp file
                        audio_path = save_audio_to_wav(verify_audio.getvalue())

                        # Get the target user's enrolled embedding and passphrase
                        target_embedding = enrolled[selected_user]
                        target_passphrase = get_user_passphrase(selected_user)

                        # Run the 1-to-1 verification pipeline
                        result = verify_access(
                            audio_path, selected_user,
                            target_embedding, target_passphrase
                        )

                        # ── Display Results ──
                        st.markdown("---")

                        # ── Gate Animation + Sound ──
                        if result["access_granted"]:
                            components.html(GATE_OPEN_ANIMATION, height=280, scrolling=False)
                            st.markdown(
                                '<div class="status-box access-granted">'
                                '\u2705 ACCESS GRANTED \u2014 Gate Opened!'
                                '</div>',
                                unsafe_allow_html=True
                            )
                            st.balloons()
                        else:
                            components.html(GATE_CLOSED_ANIMATION, height=290, scrolling=False)
                            st.markdown(
                                '<div class="status-box access-denied">'
                                '\u274c ACCESS DENIED \u2014 Gate Locked!'
                                '</div>',
                                unsafe_allow_html=True
                            )
                            # ── Log alert ──
                            reason = get_fail_reason(
                                result["speaker_matched"],
                                result["command_matched"]
                            )
                            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.alerts_log.append({
                                "user": selected_user,
                                "reason": reason,
                                "speaker_matched": result["speaker_matched"],
                                "command_matched": result["command_matched"],
                                "speaker_score": result["speaker_score"],
                                "command_score": result["command_score"],
                                "transcribed": result["transcribed_text"],
                                "timestamp": now_str,
                            })

                        # Detail Metrics
                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.markdown("#### 🎙️ Speaker Verification")
                            if result["speaker_matched"]:
                                st.success(
                                    f"✅ Matched: **{result['speaker_name']}**"
                                )
                            else:
                                st.error("❌ Speaker not recognized")
                            st.metric(
                                "Similarity Score",
                                f"{result['speaker_score']:.2%}"
                            )

                        with col_b:
                            st.markdown("#### 📝 Command Verification")
                            if result["command_matched"]:
                                st.success("✅ Passphrase matched!")
                            else:
                                st.error("❌ Wrong passphrase")
                            st.write(
                                f"**You said:** \"{result['transcribed_text']}\""
                            )
                            st.metric(
                                "Match Score",
                                f"{result['command_score']:.2%}"
                            )

                    except Exception as e:
                        st.error(f"❌ Verification failed: {e}")


# ══════════════════════════════════════════════
# TAB 2: ENROLL A NEW USER  (second tab now)
# ══════════════════════════════════════════════
with tab2:
    st.subheader("📋 Enroll a New User")
    st.markdown("Record your voice saying **your custom passphrase** to register your voiceprint.")

    col1, col2 = st.columns([1, 1])

    with col1:
        # User name input
        user_name = st.text_input(
            "Enter your name:",
            placeholder="e.g., Aarav",
            key="enroll_name"
        )

        # Custom passphrase input
        user_passphrase = st.text_input(
            "Set your custom passphrase:",
            placeholder="e.g., open the gate",
            key="enroll_passphrase",
            help="Choose a short phrase you will say to unlock the gate."
        )

        # Audio recorder
        enroll_audio = st.audio_input(
            "🎤 Record your voice (say your passphrase):",
            key="enroll_audio"
        )

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4>📝 Instructions</h4>
            <ul>
                <li>Enter your name</li>
                <li>Type your <b>custom passphrase</b> (e.g., "Open Gate")</li>
                <li>Click the mic and <b>say your passphrase</b> clearly</li>
                <li>Click <b>Enroll</b> to save your voiceprint</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Enroll button
    if st.button("✅ Enroll User", type="primary", use_container_width=True):
        if not user_name:
            st.error("❌ Please enter your name!")
        elif not user_passphrase or len(user_passphrase.strip()) < 2:
            st.error("❌ Please set a custom passphrase (at least 2 characters)!")
        elif not enroll_audio:
            st.error("❌ Please record your voice first!")
        else:
            with st.spinner("🔄 Processing voice enrollment..."):
                try:
                    # Save audio to temp file
                    audio_path = save_audio_to_wav(enroll_audio.getvalue())

                    # Extract voice embedding
                    embedding = extract_embedding(audio_path)

                    # Save user enrollment with custom passphrase
                    enroll_user(user_name, embedding, user_passphrase)

                    st.success(f"✅ **{user_name}** enrolled successfully with passphrase: *\"{user_passphrase}\"*")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Enrollment failed: {e}")


# ══════════════════════════════════════════════
# TAB 3: MANAGE ENROLLED USERS
# ══════════════════════════════════════════════
with tab3:  # noqa: E305
    st.subheader("👥 Manage Enrolled Users")

    manage_names = get_user_names()

    if len(manage_names) == 0:
        st.info("No users enrolled yet. Go to the **Enroll Voice** tab to add a user.")
    else:
        st.markdown(f"**Total Enrolled:** {len(manage_names)}")

        for name in manage_names:
            passphrase = get_user_passphrase(name) or "(not set)"

            # User card
            st.markdown(f"""
            <div class="user-row">
                <div class="user-info">
                    <div class="user-avatar">{name[0].upper()}</div>
                    <div>
                        <div class="user-name">{name}</div>
                        <div class="user-phrase">🔑 "{passphrase}"</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Action buttons row
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

            with btn_col1:
                # Play Passphrase button (TTS)
                if passphrase and passphrase != "(not set)":
                    if st.button("🔊 Play Phrase", key=f"play_{name}"):
                        with st.spinner("Generating pronunciation..."):
                            try:
                                audio_buf = text_to_speech_audio(passphrase)
                                st.audio(audio_buf, format="audio/mp3")
                            except Exception as e:
                                st.error(f"Could not generate audio: {e}")

            with btn_col2:
                if st.button("✏️ Edit", key=f"edit_toggle_{name}"):
                    st.session_state[f"editing_{name}"] = not st.session_state.get(f"editing_{name}", False)

            with btn_col3:
                if st.button("🗑️ Delete", key=f"del_{name}"):
                    delete_user(name)
                    st.success(f"Deleted **{name}**")
                    st.rerun()

            # Expandable edit form (shown when Edit is clicked)
            if st.session_state.get(f"editing_{name}", False):
                with st.container():
                    st.markdown("---")
                    st.markdown(f"**✏️ Editing: {name}**")
                    edit_col1, edit_col2 = st.columns(2)

                    with edit_col1:
                        new_name = st.text_input(
                            "New name:",
                            value=name,
                            key=f"edit_name_{name}"
                        )

                    with edit_col2:
                        new_phrase = st.text_input(
                            "New passphrase:",
                            value=passphrase if passphrase != "(not set)" else "",
                            key=f"edit_phrase_{name}"
                        )

                    save_col, cancel_col = st.columns(2)
                    with save_col:
                        if st.button("💾 Save Changes", key=f"save_{name}", type="primary"):
                            success = update_user(
                                old_name=name,
                                new_name=new_name if new_name != name else None,
                                new_passphrase=new_phrase if new_phrase != passphrase else None
                            )
                            if success:
                                st.session_state[f"editing_{name}"] = False
                                st.success(f"✅ Updated successfully!")
                                st.rerun()
                            else:
                                st.error("❌ Update failed. User not found.")
                    with cancel_col:
                        if st.button("❌ Cancel", key=f"cancel_{name}"):
                            st.session_state[f"editing_{name}"] = False
                            st.rerun()

                    st.markdown("---")


# ══════════════════════════════════════════════
# TAB 4: SECURITY ALERTS
# ══════════════════════════════════════════════
with tab4:
    st.subheader("\U0001f6a8 Security Alerts")
    st.markdown("All failed access attempts are automatically logged here in real-time.")

    alerts = st.session_state.alerts_log

    # ── Summary metrics strip ──
    total_alerts = len(alerts)
    wrong_voice  = sum(1 for a in alerts if not a["speaker_matched"] and a["command_matched"])
    wrong_pass   = sum(1 for a in alerts if a["speaker_matched"] and not a["command_matched"])
    both_failed  = sum(1 for a in alerts if not a["speaker_matched"] and not a["command_matched"])

    al_c1, al_c2, al_c3, al_c4 = st.columns(4)
    with al_c1:
        st.metric("\U0001f534 Total Alerts",   total_alerts)
    with al_c2:
        st.metric("\U0001f3d9\ufe0f Wrong Voice",    wrong_voice)
    with al_c3:
        st.metric("\U0001f511 Wrong Password", wrong_pass)
    with al_c4:
        st.metric("\U0001f480 Both Failed",    both_failed)

    st.markdown("---")

    clear_col, _ = st.columns([1, 5])
    with clear_col:
        if st.button("\U0001f5d1\ufe0f Clear All Alerts", disabled=(total_alerts == 0)):
            st.session_state.alerts_log = []
            st.rerun()

    st.markdown("")

    if total_alerts == 0:
        st.markdown("""
        <div class="alerts-empty">
            <span class="empty-icon">\U0001f6e1\ufe0f</span>
            No security alerts yet \u2014 the gate is secure!
        </div>
        """, unsafe_allow_html=True)
    else:
        for alert in reversed(alerts):
            badge_cls    = get_fail_badge_class(alert["reason"])
            speaker_icon = "\u2705" if alert["speaker_matched"] else "\u274c"
            command_icon = "\u2705" if alert["command_matched"] else "\u274c"
            st.markdown(f"""
            <div class="alert-row">
                <div class="alert-icon">\u26a0\ufe0f</div>
                <div class="alert-body">
                    <div class="alert-title">
                        Access Denied &mdash; {alert['user']}
                        &nbsp;<span class="alert-badge {badge_cls}">{alert['reason']}</span>
                    </div>
                    <div class="alert-detail">
                        <strong>User:</strong> {alert['user']}&nbsp;&nbsp;|&nbsp;&nbsp;
                        {speaker_icon} <strong>Voice:</strong> {alert['speaker_score']:.0%} similarity&nbsp;&nbsp;|&nbsp;&nbsp;
                        {command_icon} <strong>Passphrase:</strong> {alert['command_score']:.0%} match<br/>
                        <strong>Transcribed:</strong> &quot;{alert['transcribed']}&quot;
                    </div>
                    <div class="alert-time">\U0001f550 {alert['timestamp']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <span>Smart Voice Gate</span> \u2014 AI Lab 7 Mini Project<br/>
    Speaker Verification (ECAPA-TDNN) + Speech Recognition (Google API)
</div>
""", unsafe_allow_html=True)
