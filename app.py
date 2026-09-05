"""
DataForge 2026 Hackathon
Pathway Track: "Explain the Frontier"

Interactive educational explainer:
Key-Value Caching, Limitations, and Fixed-Size Recurrent Alternatives (BDH)

Run:
    streamlit run app.py
"""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="KV Cache vs Fixed-State Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #0b1020;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        .hero {
            padding: 2rem;
            border-radius: 20px;
            background:
                linear-gradient(
                    135deg,
                    rgba(37, 99, 235, 0.20),
                    rgba(124, 58, 237, 0.15)
                );
            border: 1px solid rgba(148, 163, 184, 0.20);
            margin-bottom: 1.5rem;
        }

        .hero h1 {
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }

        .claim {
            font-size: 1.25rem;
            line-height: 1.6;
        }

        .learning {
            padding: 1rem 1.25rem;
            border-left: 4px solid #60a5fa;
            background: rgba(96, 165, 250, 0.08);
            border-radius: 8px;
            margin-top: 1rem;
        }

        .metric-card {
            padding: 1.25rem;
            border-radius: 15px;
            border: 1px solid rgba(148, 163, 184, 0.20);
            background: rgba(15, 23, 42, 0.75);
            min-height: 150px;
        }

        .metric-title {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-bottom: 0.4rem;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
        }

        .metric-subtitle {
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: 0.4rem;
        }

        .status-good {
            color: #34d399;
            font-weight: 700;
        }

        .status-bad {
            color: #fb7185;
            font-weight: 700;
        }

        .section-card {
            padding: 1.5rem;
            border-radius: 16px;
            background: rgba(15, 23, 42, 0.60);
            border: 1px solid rgba(148, 163, 184, 0.15);
            margin: 1rem 0;
        }

        .formula {
            font-family: monospace;
            font-size: 1rem;
            padding: 1rem;
            background: #020617;
            border-radius: 10px;
            overflow-x: auto;
        }

        .small-note {
            color: #94a3b8;
            font-size: 0.85rem;
        }

        .tradeoff {
            padding: 1rem;
            border-radius: 12px;
            background: rgba(245, 158, 11, 0.08);
            border: 1px solid rgba(245, 158, 11, 0.20);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def format_bytes(num_bytes: float) -> str:
    """Convert bytes to a human-readable memory value."""

    gb = num_bytes / (1024 ** 3)
    mb = num_bytes / (1024 ** 2)

    if gb >= 1:
        return f"{gb:,.2f} GB"

    return f"{mb:,.2f} MB"


def kv_cache_bytes(
    seq_len: int,
    batch_size: int,
    n_layers: int,
    n_heads: int,
    d_model: int,
    bytes_per_elem: int,
) -> float:
    """
    Calculate standard Transformer KV-cache memory.

    Formula:

        2 * 2 * n_layers * n_heads * d_head
        * seq_len * batch_size * bytes_per_elem

    Explanation:

        First 2:
            K + V

        Second 2:
            The attention cache contains two tensors:
            conceptually, each stored element occupies
            bytes_per_elem bytes. The formula is conventionally
            written as 2 (K/V) * ...

        n_layers:
            Every Transformer layer maintains its own KV cache.

        n_heads:
            Number of attention heads.

        d_head:
            Dimension of one attention head.

        seq_len:
            Number of tokens stored in the cache.

        batch_size:
            Number of sequences processed simultaneously.

        bytes_per_elem:
            FP16/BF16 -> 2 bytes
            FP8        -> 1 byte
    """

    d_head = d_model / n_heads

    return (
        2
        * n_layers
        * n_heads
        * d_head
        * seq_len
        * batch_size
        * bytes_per_elem
    )


def bdh_state_bytes(
    n_layers: int,
    d_model: int,
    state_multiplier: float = 1.0,
    bytes_per_elem: int = 2,
) -> float:
    """
    Simplified educational model of fixed-size recurrent state.

    Important:
        This is NOT claiming to reproduce a specific BDH
        implementation's exact parameter/state allocation.

    The point of the visualization is asymptotic behavior:
        Transformer KV cache -> O(N)
        Fixed recurrent state -> O(1)

    The state size is therefore modeled as a constant proportional
    to model width and number of layers.
    """

    return (
        n_layers
        * d_model
        * state_multiplier
        * bytes_per_elem
    )


def memory_gb(num_bytes: float) -> float:
    """Convert bytes to GiB."""

    return num_bytes / (1024 ** 3)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Experiment Controls")

    st.markdown(
        "Change the model and hardware assumptions to see "
        "how memory behavior changes."
    )

    st.divider()

    seq_len = st.slider(
        "Sequence Length / Context Size",
        min_value=512,
        max_value=131072,
        value=32768,
        step=512,
        help="Number of tokens represented by the context.",
    )

    batch_size = st.slider(
        "Batch Size",
        min_value=1,
        max_value=64,
        value=1,
        step=1,
    )

    precision = st.selectbox(
        "Precision / Data Type",
        options=[
            "FP16 / BF16 — 2 bytes",
            "FP8 — 1 byte",
        ],
        index=0,
    )

    if precision.startswith("FP8"):
        bytes_per_elem = 1
        precision_label = "FP8"
    else:
        bytes_per_elem = 2
        precision_label = "FP16 / BF16"

    st.divider()

    st.subheader("🧩 Model Specification")

    n_layers = st.slider(
        "Number of Layers",
        min_value=4,
        max_value=128,
        value=32,
        step=4,
    )

    d_model = st.slider(
        "Hidden Dimension / d_model",
        min_value=512,
        max_value=16384,
        value=4096,
        step=512,
    )

    # Number of heads must divide d_model.
    possible_heads = [
        h for h in [8, 16, 32, 64, 128]
        if d_model % h == 0
    ]

    if not possible_heads:
        possible_heads = [8]

    default_head_index = (
        possible_heads.index(32)
        if 32 in possible_heads
        else 0
    )

    n_heads = st.selectbox(
        "Number of Attention Heads",
        options=possible_heads,
        index=default_head_index,
    )

    d_head = d_model // n_heads

    st.caption(
        f"Derived head dimension: d_head = {d_head}"
    )

    st.divider()

    gpu_vram = st.slider(
        "Available GPU VRAM",
        min_value=4,
        max_value=192,
        value=24,
        step=4,
        help=(
            "Used as the OOM threshold for the educational "
            "memory comparison."
        ),
    )

    st.divider()

    st.markdown(
        """
        **Try this experiment:**

        1. Set VRAM to **24 GB**
        2. Use **32 layers**
        3. Use **4096 d_model**
        4. Set context to **131,072**
        5. Watch the KV cache cross the OOM line.
        """
    )


# ============================================================
# CALCULATIONS
# ============================================================

current_kv_bytes = kv_cache_bytes(
    seq_len=seq_len,
    batch_size=batch_size,
    n_layers=n_layers,
    n_heads=n_heads,
    d_model=d_model,
    bytes_per_elem=bytes_per_elem,
)

current_kv_gb = memory_gb(current_kv_bytes)

# Educational fixed-state model.
#
# A multiplier of 1 means:
#     layers × d_model × bytes
#
# This intentionally models a bounded recurrent state rather
# than claiming an exact BDH implementation memory footprint.
bdh_bytes = bdh_state_bytes(
    n_layers=n_layers,
    d_model=d_model,
    state_multiplier=1.0,
    bytes_per_elem=bytes_per_elem,
)

bdh_gb = memory_gb(bdh_bytes)

gpu_vram_bytes = gpu_vram * (1024 ** 3)

transformer_oom = current_kv_bytes > gpu_vram_bytes

# Determine the approximate context length where KV alone
# reaches the selected VRAM threshold.

kv_bytes_per_token = kv_cache_bytes(
    seq_len=1,
    batch_size=batch_size,
    n_layers=n_layers,
    n_heads=n_heads,
    d_model=d_model,
    bytes_per_elem=bytes_per_elem,
)

if kv_bytes_per_token > 0:
    oom_context = int(gpu_vram_bytes / kv_bytes_per_token)
else:
    oom_context = 0


# ============================================================
# HERO SECTION
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🧠 When Memory Becomes the Bottleneck</h1>

        <div class="claim">
            <strong>
            Standard Transformer KV caching grows linearly with
            context length, while fixed-size recurrent alternatives
            such as BDH keep inference memory bounded — trading
            perfect token-level recall for finite state capacity
            and potential interference.
            </strong>
        </div>

        <div class="learning">
            <strong>🎯 Learning objective</strong><br>
            Understand why increasing an LLM's context window can
            become a memory problem, and how recurrent state
            architectures change the memory/recall trade-off.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROBLEM EXPLANATION
# ============================================================

st.markdown("## 1. The KV-Cache Bottleneck")

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        ### Why does a Transformer need a KV cache?

        During autoregressive generation, a Transformer repeatedly
        attends to previous tokens.

        Instead of recomputing the keys and values for every previous
        token at every generation step, implementations cache them.

        Conceptually:

        **Token → Key + Value → Stored in GPU memory**

        The benefit is speed.

        The cost is that the cache grows as more tokens enter the
        context.
        """
    )

with col2:

    st.markdown(
        """
        ### The scaling problem

        For conventional multi-head attention:

        \[
        M_{KV} =
        2 \\times L \\times H \\times d_{head}
        \\times N \\times B \\times S
        \]

        where:

        - \(L\) = number of layers
        - \(H\) = attention heads
        - \(d_{head}\) = dimension per head
        - \(N\) = sequence length
        - \(B\) = batch size
        - \(S\) = bytes per element
        - first **2** = K and V

        Since \(N\) appears directly:

        \[
        M_{KV} = O(N)
        \]

        **Double the context → approximately double the KV memory.**
        """
    )


# ============================================================
# LIVE METRIC CARDS
# ============================================================

st.markdown("## 2. Live Memory Snapshot")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Transformer KV Cache
            </div>
            <div class="metric-value">
                {current_kv_gb:,.2f} GB
            </div>
            <div class="metric-subtitle">
                Exact formula-based estimate
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Fixed Recurrent State
            </div>
            <div class="metric-value">
                {bdh_gb:,.2f} GB
            </div>
            <div class="metric-subtitle">
                Bounded educational model
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m3:

    status_text = (
        "STATUS: OOM"
        if transformer_oom
        else "STATUS: FITS"
    )

    status_class = (
        "status-bad"
        if transformer_oom
        else "status-good"
    )

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Transformer Status
            </div>
            <div class="metric-value {status_class}">
                {status_text}
            </div>
            <div class="metric-subtitle">
                GPU limit: {gpu_vram} GB
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with m4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">
                Recurrent Status
            </div>
            <div class="metric-value status-good">
                BOUNDED
            </div>
            <div class="metric-subtitle">
                State does not grow with N
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PLOT 1 — MEMORY VS CONTEXT
# ============================================================

st.markdown("## 3. Plot 1 — Memory vs. Context Length")

st.markdown(
    """
    The key frontier is visible here:

    **Transformer KV memory rises with every additional token.
    Fixed recurrent state stays approximately flat.**
    """
)

# Generate context points.
context_values = np.unique(
    np.round(
        np.geomspace(
            512,
            131072,
            100
        )
    ).astype(int)
)

# Always include important endpoints.
context_values = np.unique(
    np.concatenate(
        [
            context_values,
            np.array([512, 4096, 8192, 16384, 32768, 65536, 131072])
        ]
    )
)

transformer_memory = np.array(
    [
        memory_gb(
            kv_cache_bytes(
                seq_len=int(n),
                batch_size=batch_size,
                n_layers=n_layers,
                n_heads=n_heads,
                d_model=d_model,
                bytes_per_elem=bytes_per_elem,
            )
        )
        for n in context_values
    ]
)

# Constant recurrent memory.
bdh_memory = np.full(
    len(context_values),
    bdh_gb,
)

fig_memory = go.Figure()

fig_memory.add_trace(
    go.Scatter(
        x=context_values,
        y=transformer_memory,
        mode="lines",
        name="Transformer KV Cache — O(N)",
        line=dict(width=4),
        hovertemplate=(
            "Context: %{x:,} tokens"
            "<br>KV memory: %{y:.2f} GB"
            "<extra></extra>"
        ),
    )
)

fig_memory.add_trace(
    go.Scatter(
        x=context_values,
        y=bdh_memory,
        mode="lines",
        name="Fixed Recurrent State — O(1)",
        line=dict(width=4, dash="dash"),
        hovertemplate=(
            "Context: %{x:,} tokens"
            "<br>State: %{y:.2f} GB"
            "<extra></extra>"
        ),
    )
)

# GPU OOM threshold.
fig_memory.add_hline(
    y=gpu_vram,
    line_dash="dot",
    line_width=3,
    annotation_text=f"GPU VRAM limit = {gpu_vram} GB",
    annotation_position="top left",
)

# Current context marker.
fig_memory.add_vline(
    x=seq_len,
    line_dash="dot",
    line_width=2,
    annotation_text=f"Selected N = {seq_len:,}",
    annotation_position="top right",
)

fig_memory.update_layout(
    height=560,
    template="plotly_dark",
    xaxis=dict(
        title="Context Length / Sequence Length (tokens)",
        type="log",
        tickformat=",",
    ),
    yaxis=dict(
        title="Memory (GB)",
        rangemode="tozero",
    ),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    margin=dict(l=50, r=30, t=80, b=50),
)

st.plotly_chart(
    fig_memory,
    use_container_width=True,
)


# ============================================================
# OOM EXPLANATION
# ============================================================

if transformer_oom:

    st.error(
        f"""
        **STATUS: OOM (Out of Memory)**

        At the selected configuration, the Transformer KV cache
        requires approximately **{current_kv_gb:.2f} GB**, exceeding
        the selected **{gpu_vram} GB** GPU memory threshold.

        The KV cache reaches the threshold at approximately
        **{oom_context:,} tokens** under these assumptions.
        """
    )

else:

    remaining = gpu_vram - current_kv_gb

    st.success(
        f"""
        **STATUS: FITS**

        The KV cache currently requires **{current_kv_gb:.2f} GB**
        of the selected **{gpu_vram} GB** VRAM.

        Approximately **{remaining:.2f} GB** remains before the
        KV-cache-only estimate reaches the threshold.
        """
    )

st.caption(
    "Important: this comparison isolates KV-cache memory. "
    "Real inference also consumes memory for model weights, "
    "activations, CUDA/runtime allocations, temporary tensors, "
    "and other buffers. Therefore, actual OOM can occur earlier."
)


# ============================================================
# EXACT FORMULA SECTION
# ============================================================

st.markdown("## 4. Ground Truth: How the KV Number Is Calculated")

st.markdown(
    """
    The KV-cache number shown above comes directly from the
    following formula.
    """
)

st.markdown(
    """
    <div class="formula">
    KV bytes =
    2 × 2 × n_layers × n_heads × d_head
    × seq_len × batch_size × bytes_per_elem
    </div>
    """,
    unsafe_allow_html=True,
)

formula_col1, formula_col2 = st.columns(2)

with formula_col1:

    st.markdown(
        f"""
        **Current parameters**

        - Layers: `{n_layers}`
        - Heads: `{n_heads}`
        - d_model: `{d_model}`
        - d_head: `{d_head}`
        - Context: `{seq_len:,}`
        - Batch: `{batch_size}`
        - Precision: `{precision_label}`
        - Bytes/element: `{bytes_per_elem}`

        Therefore:

        \[
        d_{{head}} =
        \\frac{{d_{{model}}}}{{H}}
        =
        \\frac{{{d_model}}}{{{n_heads}}}
        =
        {d_head}
        \]
        """
    )

with formula_col2:

    st.markdown(
        f"""
        **Current result**

        \[
        M_{{KV}}
        =
        2 \\times {n_layers} \\times {n_heads}
        \\times {d_head}
        \\times {seq_len}
        \\times {batch_size}
        \\times {bytes_per_elem}
        \]

        = **{current_kv_bytes:,.0f} bytes**

        = **{current_kv_gb:,.2f} GiB**

        This is the **formula-derived KV-cache estimate**, not a
        measurement from a physical GPU.
        """
    )


# ============================================================
# PLOT 2 — INFORMATION RETENTION VS INTERFERENCE
# ============================================================

st.markdown("## 5. Plot 2 — Information Retention vs. Interference")

st.markdown(
    """
    Memory efficiency changes what the model can preserve.

    A KV cache explicitly retains representations for previous
    tokens. A fixed recurrent architecture instead continuously
    updates a bounded internal state.

    That creates a fundamental trade-off:

    **More explicit storage → better direct access to past tokens**

    **More aggressive compression → bounded memory, but greater
    possibility of interference between information.**
    """
)

# Educational conceptual data.
#
# These are NOT benchmark measurements of BDH.
# They visualize the qualitative trade-off requested by the demo.

horizon = np.linspace(0, 100, 101)

# KV cache:
# explicit retention remains high.
kv_retention = np.ones_like(horizon) * 1.0

# Conceptual recurrent retention:
# decreases as more information must share a finite state.
recurrent_retention = (
    0.95 * np.exp(-0.018 * horizon) + 0.05
)

# Conceptual interference:
# increases as the finite state is repeatedly updated.
recurrent_interference = (
    1 - recurrent_retention
)

fig_tradeoff = go.Figure()

fig_tradeoff.add_trace(
    go.Scatter(
        x=horizon,
        y=kv_retention,
        mode="lines",
        name="KV Cache: Explicit token representations",
        line=dict(width=4),
        hovertemplate=(
            "History: %{x:.0f}%"
            "<br>Retention: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)

fig_tradeoff.add_trace(
    go.Scatter(
        x=horizon,
        y=recurrent_retention,
        mode="lines",
        name="Fixed State: Information retained",
        line=dict(width=4, dash="dash"),
        hovertemplate=(
            "History: %{x:.0f}%"
            "<br>Conceptual retention: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)

fig_tradeoff.add_trace(
    go.Scatter(
        x=horizon,
        y=recurrent_interference,
        mode="lines",
        name="Fixed State: Potential interference",
        line=dict(width=3, dash="dot"),
        hovertemplate=(
            "History: %{x:.0f}%"
            "<br>Conceptual interference: %{y:.2f}"
            "<extra></extra>"
        ),
    )
)

fig_tradeoff.update_layout(
    height=500,
    template="plotly_dark",
    xaxis_title="Increasing history / information load",
    yaxis_title="Normalized conceptual quantity",
    yaxis=dict(range=[0, 1.08]),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
)

st.plotly_chart(
    fig_tradeoff,
    use_container_width=True,
)

st.warning(
    """
    **Interpretation warning:** The curves in this plot are
    deliberately conceptual, not empirical BDH benchmark results.
    They illustrate the information-theoretic intuition:
    bounded state capacity means different pieces of information
    must share the same evolving state.
    """
)


# ============================================================
# VISUAL CONCEPT — TWO MEMORY SYSTEMS
# ============================================================

st.markdown("## 6. What Actually Changes?")

left, right = st.columns(2)

with left:

    st.markdown(
        """
        <div class="section-card">

        ### 🗃️ Transformer KV Cache

        **Input sequence**

        `T1 → T2 → T3 → T4 → ... → TN`

        ↓

        **Stored representations**

        `K1,V1`

        `K2,V2`

        `K3,V3`

        `K4,V4`

        `...`

        `KN,VN`

        <br>

        Every additional token adds another set of cached
        representations.

        **Memory: O(N)**

        </div>
        """,
        unsafe_allow_html=True,
    )

with right:

    st.markdown(
        """
        <div class="section-card">

        ### 🧠 Fixed Recurrent State

        **Input sequence**

        `T1 → T2 → T3 → T4 → ... → TN`

        ↓

        **Evolving state**

        `S₀ → S₁ → S₂ → S₃ → ... → Sₙ`

        But the size of each state remains bounded.

        The model does not keep a separate KV pair for every
        historical token.

        **Memory: O(1) with respect to sequence length**

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# BDH CONNECTION
# ============================================================

st.markdown("## 7. The BDH Connection")

st.markdown(
    """
    ### Dragon Hatchling (BDH): replacing an ever-growing cache
    with an evolving state

    The central idea behind fixed-state recurrent approaches is to
    process information through a persistent internal state rather
    than requiring an explicitly growing attention cache.

    Instead of thinking:

    > "Store every previous key and value so that I can attend to it
    > later."

    the recurrent viewpoint is closer to:

    > "Continuously update a state that summarizes the information
    > needed for future computation."

    In BDH-style formulations, this state can be interpreted as a
    **synaptic memory** whose parameters/state are updated as new
    information arrives.

    A simplified conceptual form is:

    \[
    S_{t+1}
    =
    S_t
    +
    \Delta S(x_t, S_t)
    \]

    where \(S_t\) is the current internal/synaptic state and
    \(x_t\) is the current input.

    A Hebbian-style intuition is:

    \[
    \Delta S
    \propto
    \text{pre-synaptic activity}
    \times
    \text{post-synaptic activity}
    \]

    The important architectural consequence is that the state has a
    **fixed capacity** rather than allocating another K/V record for
    every token.

    Therefore, with respect to context length:

    \[
    M_{recurrent} = O(1)
    \]

    while the conventional KV cache behaves as:

    \[
    M_{KV} = O(N)
    \]

    This is the central memory-scaling distinction.
    """
)


# ============================================================
# ASSOCIATIVE RECALL
# ============================================================

st.markdown("### ⚡ Why this can still support useful recall")

recall1, recall2, recall3 = st.columns(3)

with recall1:

    st.markdown(
        """
        #### 1. Local update

        New information modifies the existing state rather than
        allocating a completely new historical memory slot.
        """
    )

with recall2:

    st.markdown(
        """
        #### 2. Distributed representation

        Information can be represented across the state rather than
        being stored as isolated token records.
        """
    )

with recall3:

    st.markdown(
        """
        #### 3. Fast access

        Future computation operates directly on the current state,
        avoiding a cache whose physical size grows with every token.
        """
    )


# ============================================================
# LIMITATIONS
# ============================================================

st.markdown("## 8. Disclosed Limitations & Caveats")

st.markdown(
    """
    <div class="tradeoff">

    ### ⚠️ Fixed memory does NOT mean unlimited memory

    Removing KV-cache growth solves one problem, but introduces
    another.

    A fixed state has finite representational capacity.

    If an extremely long stream contains more information than the
    state can preserve distinctly, multiple pieces of information
    must share that limited state.

    This can lead to:

    - **State interference** — newer updates can alter information
      that was represented earlier.
    - **Loss of exact recall** — verbatim historical details are not
      necessarily preserved in the way explicit KV entries are.
    - **Recency effects** — recent information can have greater
      influence on the evolving state.
    - **Capacity trade-offs** — increasing state capacity costs
      memory and computation.
    - **Task dependence** — the usefulness of compressed recurrent
      memory depends on what information the task actually needs.

    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    ### The honest comparison

    | Property | Transformer KV Cache | Fixed Recurrent State |
    |---|---|---|
    | Memory vs context | **O(N)** | **O(1)** |
    | Explicit historical token representations | ✅ | ❌ |
    | Exact long-range recall | Stronger | Can degrade |
    | Memory exhaustion from context alone | Possible | Bounded |
    | State interference | Lower | Possible |
    | Recency effects | Less fundamental | Important consideration |
    | Main advantage | Direct attention to stored history | Fixed memory footprint |

    The point is **not** that recurrent memory universally replaces
    Transformers.

    The interesting frontier is the trade-off between **explicit
    memory** and **compressed state**.
    """
)


# ============================================================
# INTERACTIVE "WHAT IF?" SECTION
# ============================================================

st.markdown("## 9. What Happens If We Change the Context?")

test_contexts = [
    512,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
]

comparison_rows = []

for n in test_contexts:

    kv = memory_gb(
        kv_cache_bytes(
            seq_len=n,
            batch_size=batch_size,
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            bytes_per_elem=bytes_per_elem,
        )
    )

    comparison_rows.append(
        {
            "Context": f"{n:,}",
            "Transformer KV": f"{kv:.2f} GB",
            "Fixed State": f"{bdh_gb:.2f} GB",
            "Transformer": (
                "OOM"
                if kv > gpu_vram
                else "Fits"
            ),
        }
    )

st.table(comparison_rows)


# ============================================================
# KEY TAKEAWAY
# ============================================================

st.markdown("## 10. The Frontier in One Picture")

final_col1, final_col2 = st.columns(2)

with final_col1:

    st.markdown(
        """
        ### 🏗️ Transformer strategy

        **Keep the history.**

        More context means more stored K/V representations.

        That makes attention over previous tokens efficient, but
        memory grows with sequence length.

        \[
        \boxed{M_{KV}=O(N)}
        \]
        """
    )

with final_col2:

    st.markdown(
        """
        ### 🧠 Fixed-state strategy

        **Compress the history into state.**

        Memory stays bounded as the sequence grows, but the model
        must continuously overwrite/update a finite representation.

        \[
        \boxed{M_{state}=O(1)}
        \]

        The price is that perfect historical recall is no longer
        guaranteed.
        """
    )


st.success(
    """
    **Core takeaway:**

    The question is not simply *"Which architecture uses less
    memory?"*

    It is:

    **How much explicit history should a model retain, and how much
    can it safely compress into a fixed computational state?**
    """
)


# ============================================================
# TECHNICAL FOOTNOTE
# ============================================================

with st.expander("🔬 Technical assumptions behind this demo"):

    st.markdown(
        f"""
        This application deliberately separates **exact arithmetic**
        from **conceptual educational modeling**.

        ### Exact / formula-derived

        The Transformer KV-cache calculation uses:

        \[
        2 \\times L \\times H \\times d_{{head}}
        \\times N \\times B \\times S
        \]

        with:

        - \(L = {n_layers}\)
        - \(H = {n_heads}\)
        - \(d_{{head}} = {d_head}\)
        - \(N = {seq_len:,}\)
        - \(B = {batch_size}\)
        - \(S = {bytes_per_elem}\)

        Result:

        **{current_kv_bytes:,.0f} bytes
        = {current_kv_gb:.2f} GiB**

        ### Educational / conceptual

        The fixed-state memory number is a simplified constant-state
        model:

        \[
        M_{{state}}
        =
        L \\times d_{{model}} \\times S
        \]

        It is intended to demonstrate the **O(1) asymptotic behavior**,
        not to claim that every BDH implementation has exactly this
        byte-level memory requirement.

        Likewise, the information-retention/interference curves are
        conceptual illustrations rather than measured BDH benchmark
        results.

        ### Real GPU memory

        Actual inference memory is larger than KV cache alone because
        it can include:

        - model weights
        - activations
        - temporary attention/workspace tensors
        - CUDA/runtime allocations
        - allocator fragmentation
        - other inference buffers

        Therefore, the GPU line in this visualization should be
        interpreted as a **KV-cache memory threshold**, not a guarantee
        of the maximum deployable context length on a physical GPU.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b;">

    **DataForge 2026 — Explain the Frontier**

    *Key-Value Caching → Memory Scaling → Fixed-State Recurrence*

    </div>
    """,
    unsafe_allow_html=True,
)