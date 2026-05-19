import streamlit as st
import numpy as np
import json
import os
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexiShift Explorer",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 860px; }

/* ── top header ── */
.ls-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 0.25rem;
}
.ls-logo {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -2px;
    color: #0a0a0a;
}
.ls-logo-accent { color: #1D9E75; }
.ls-tagline {
    font-size: 0.8rem;
    color: #888;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding-bottom: 3px;
}

/* ── divider ── */
.ls-divider {
    height: 1.5px;
    background: linear-gradient(to right, #1D9E75, #378ADD, transparent);
    margin: 0.75rem 0 1.5rem 0;
    border: none;
}

/* ── metric cards ── */
.metric-grid {
    display: flex;
    gap: 12px;
    margin-bottom: 1.25rem;
}
.metric-card {
    flex: 1;
    background: #f8f8f6;
    border: 1px solid #e8e8e4;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
.metric-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #0a0a0a;
}
.metric-value.changed { color: #1D9E75; }
.metric-value.stable  { color: #378ADD; }

/* ── verdict banner ── */
.verdict-changed {
    background: #e1f5ee;
    border: 1.5px solid #5DCAA5;
    border-left: 5px solid #1D9E75;
    border-radius: 8px;
    padding: 0.85rem 1.2rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.verdict-stable {
    background: #e6f1fb;
    border: 1.5px solid #85B7EB;
    border-left: 5px solid #378ADD;
    border-radius: 8px;
    padding: 0.85rem 1.2rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 12px;
}
.verdict-icon {
    font-size: 1.5rem;
    line-height: 1;
}
.verdict-title {
    font-size: 1rem;
    font-weight: 600;
    color: #0a0a0a;
    margin-bottom: 2px;
}
.verdict-sub {
    font-size: 0.8rem;
    color: #555;
}

/* ── section label ── */
.section-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #aaa;
    margin-bottom: 0.4rem;
    margin-top: 0.25rem;
}

/* ── sample chips ── */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-bottom: 1.5rem;
}
.chip {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    padding: 4px 11px;
    border: 1px solid #d0d0c8;
    border-radius: 20px;
    color: #555;
    background: white;
    cursor: default;
}

/* ── chart cards ── */
.chart-card {
    background: white;
    border: 1px solid #e8e8e4;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.25rem;
}
.chart-card-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #999;
    margin-bottom: 0.75rem;
}

/* ── empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: #bbb;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.empty-text { font-size: 0.9rem; }

/* ── input styling ── */
.stTextInput > div > div > input {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    border-radius: 8px !important;
    border: 1.5px solid #d8d8d0 !important;
    padding: 0.6rem 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1D9E75 !important;
    box-shadow: 0 0 0 2px rgba(29,158,117,0.15) !important;
}
.stButton > button {
    background: #1D9E75 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 1.4rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #0F6E56 !important;
}

/* ── progress bar (gauge) ── */
.gauge-wrap {
    position: relative;
    margin: 0.25rem 0 0.5rem 0;
}
.gauge-track {
    height: 12px;
    background: #eeeee8;
    border-radius: 6px;
    overflow: visible;
    position: relative;
}
.gauge-fill-changed {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(to right, #9FE1CB, #1D9E75);
    transition: width 0.6s ease;
}
.gauge-fill-stable {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(to right, #B5D4F4, #378ADD);
    transition: width 0.6s ease;
}
.gauge-threshold {
    position: absolute;
    top: -5px;
    height: 22px;
    width: 2px;
    background: #E24B4A;
    border-radius: 1px;
}
.gauge-threshold-label {
    position: absolute;
    top: 20px;
    font-size: 0.65rem;
    color: #E24B4A;
    transform: translateX(-50%);
    white-space: nowrap;
}
.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    color: #bbb;
    font-family: 'Space Mono', monospace;
    margin-top: 24px;
}
</style>
""", unsafe_allow_html=True)

# ─── SETTINGS ────────────────────────────────────────────────────────────────
EMBEDDINGS_DIR = "embeddings_by_word"
BINARY_THRESHOLD = 1.16
SAMPLE_WORDS = ["attack_nn", "plane_nn", "player_nn", "tip_vb", "tree_nn", "bag_nn", 
                "land_nn", "head_nn", "edge_nn", "circle_vb", "bit_nn", "record_nn"]

# ─── EMBEDDING HELPERS ───────────────────────────────────────────────────────
def safe_filename(name):
    return re.sub(r'[^a-zA-Z0-9._-]', '_', name)

@st.cache_data(show_spinner=False)
def load_all_embeddings(word):
    filename = safe_filename(word.lower()) + ".json"
    path = os.path.join(EMBEDDINGS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def get_word_embeddings(data, word):
    if data is None:
        return None, None
    word = word.lower()
    if word not in data:
        return None, None
    c1 = np.array(data[word].get("C1", []))
    c2 = np.array(data[word].get("C2", []))
    return c1, c2

# ─── COMPUTATION ─────────────────────────────────────────────────────────────
def compute_shift(c1, c2):
    mean1 = np.mean(c1, axis=0)
    mean2 = np.mean(c2, axis=0)
    cos_sim = np.dot(mean1, mean2) / (np.linalg.norm(mean1) * np.linalg.norm(mean2))
    score = float(1 - cos_sim)
    return round(score, 4), round(float(cos_sim), 4)

# ─── VISUALIZATION ───────────────────────────────────────────────────────────
def plot_pca(c1, c2, changed):
    all_emb = np.vstack([c1, c2])
    labels = ["C1"] * len(c1) + ["C2"] * len(c2)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_emb)
    r1 = reduced[:len(c1)]
    r2 = reduced[len(c1):]

    var_explained = pca.explained_variance_ratio_ * 100

    # Colors
    c1_color  = "#1D9E75"
    c2_color  = "#378ADD"
    c1_center = "#0F6E56"
    c2_center = "#185FA5"

    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafaf8")

    # Grid
    ax.grid(True, color="#e8e8e4", linewidth=0.6, linestyle="--", zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#e0e0da")
        spine.set_linewidth(0.8)

    # Scatter
    ax.scatter(r1[:, 0], r1[:, 1], c=c1_color, alpha=0.45, s=26, zorder=2, label=f"C1 ({len(c1)} contexts)")
    ax.scatter(r2[:, 0], r2[:, 1], c=c2_color, alpha=0.45, s=26, zorder=2, label=f"C2 ({len(c2)} contexts)")

    # Centroids
    cx1, cy1 = r1.mean(axis=0)
    cx2, cy2 = r2.mean(axis=0)
    ax.scatter(cx1, cy1, c=c1_center, s=90, zorder=5, edgecolors="white", linewidths=1.8)
    ax.scatter(cx2, cy2, c=c2_center, s=90, zorder=5, edgecolors="white", linewidths=1.8)

    # Arrow between centroids
    ax.annotate("", xy=(cx2, cy2), xytext=(cx1, cy1),
                arrowprops=dict(arrowstyle="->", color="#E24B4A", lw=1.4,
                                connectionstyle="arc3,rad=0.15"))

    ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)", fontsize=8, color="#888", labelpad=6)
    ax.set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)", fontsize=8, color="#888", labelpad=6)
    ax.tick_params(labelsize=7, colors="#aaa")

    legend = ax.legend(
        loc="upper right", fontsize=7.5, framealpha=0.9,
        edgecolor="#e0e0da", facecolor="white"
    )

    plt.tight_layout(pad=1.2)
    return fig

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ls-header">
  <div class="ls-logo">Lexi<span class="ls-logo-accent">Shift</span></div>
  <div class="ls-tagline">Semantic shift detection · SemEval English</div>
</div>
<div class="ls-divider"></div>
""", unsafe_allow_html=True)

# ─── INPUT ROW ───────────────────────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    word = st.text_input(
        label="word_input",
        placeholder="Enter a target word (e.g. plane_nn, tree_nn, attack_nn...)",
        label_visibility="collapsed",
        key="word_field"
    )
with col_btn:
    st.markdown("<div style='margin-top:4px'>", unsafe_allow_html=True)
    analyze = st.button("Analyze ->")
    st.markdown("</div>", unsafe_allow_html=True)

# ─── SAMPLE CHIPS ────────────────────────────────────────────────────────────
chips_html = '<div class="chip-row">' + \
    "".join(f'<span class="chip">{w}</span>' for w in SAMPLE_WORDS) + \
    '</div>'
st.markdown(chips_html, unsafe_allow_html=True)

# ─── MAIN LOGIC ──────────────────────────────────────────────────────────────
trigger_word = word.strip().lower() if (word and (analyze or word)) else None

if not trigger_word:
    st.markdown("""
    <div class="empty-state">
      <div class="empty-icon">◎</div>
      <div class="empty-text">Enter a word above to detect semantic shift across C1 and C2 corpora</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load
with st.spinner(f"Loading embeddings for '{trigger_word}'..."):
    data = load_all_embeddings(trigger_word)
    c1, c2 = get_word_embeddings(data, trigger_word)

if c1 is None or len(c1) == 0 or len(c2) == 0:
    st.markdown(f"""
    <div style="background:#fcebeb;border:1.5px solid #f09595;border-left:5px solid #E24B4A;
                border-radius:8px;padding:.85rem 1.2rem;margin-bottom:1rem">
      <strong style="color:#A32D2D">Word not found:</strong>
      <span style="color:#555;font-size:.9rem">
        No embeddings available for <code>{trigger_word}</code>.
        Make sure the word exists in your <code>embeddings_by_word/</code> directory.
      </span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Compute
score, cos_sim = compute_shift(c1, c2)
changed = score >= BINARY_THRESHOLD

# ─── METRIC CARDS ────────────────────────────────────────────────────────────
cls = "changed" if changed else "stable"
st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card">
    <div class="metric-label">Shift Score</div>
    <div class="metric-value {cls}">{score:.4f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">Cosine Similarity</div>
    <div class="metric-value">{cos_sim:.4f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">C1 Contexts</div>
    <div class="metric-value">{len(c1)}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">C2 Contexts</div>
    <div class="metric-value">{len(c2)}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── VERDICT BANNER ──────────────────────────────────────────────────────────
if changed:
    st.markdown(f"""
    <div class="verdict-changed">
      <div class="verdict-icon">&harr;</div>
      <div>
        <div class="verdict-title">"{trigger_word}" - Semantic shift detected</div>
        <div class="verdict-sub">Score {score:.4f} exceeds threshold theta = {BINARY_THRESHOLD}.
        This word shows significant meaning change between C1 and C2.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="verdict-stable">
      <div class="verdict-icon">&check;</div>
      <div>
        <div class="verdict-title">"{trigger_word}" - Semantically stable</div>
        <div class="verdict-sub">Score {score:.4f} is below threshold theta = {BINARY_THRESHOLD}.
        This word maintains consistent usage across both time periods.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── SCORE GAUGE ─────────────────────────────────────────────────────────────
pct = min(100.0, (score / 2.0) * 100)
threshold_pct = (BINARY_THRESHOLD / 2.0) * 100
fill_class = "gauge-fill-changed" if changed else "gauge-fill-stable"

st.markdown('<div class="section-label">Shift score gauge</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="chart-card">
  <div class="gauge-wrap">
    <div class="gauge-track">
      <div class="{fill_class}" style="width:{pct:.1f}%"></div>
      <div class="gauge-threshold" style="left:{threshold_pct:.1f}%">
        <div class="gauge-threshold-label">theta = {BINARY_THRESHOLD}</div>
      </div>
    </div>
    <div class="gauge-labels">
      <span>0.0</span>
      <span>0.5</span>
      <span>1.0</span>
      <span>1.5</span>
      <span>2.0</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── PCA PLOT ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">PCA embedding projection</div>', unsafe_allow_html=True)
fig = plot_pca(c1, c2, changed)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid #eeeee8;
            font-size:0.72rem;color:#bbb;text-align:center;letter-spacing:0.04em">
  LexiShift Explorer - SemEval-2020 Task 1 English - Binary threshold theta = 1.16
</div>
""", unsafe_allow_html=True)