import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Monte Carlo Casino Engine Pro", 
    layout="wide", 
    page_icon="🎰"
)

# Custom CSS for UI Polish
st.markdown("""
    <style>
    .card-container {
        transition: transform 0.2s;
        border: 2px solid #EEE;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        background-color: white;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.05);
    }
    .card-container:hover {
        transform: translateY(-10px);
        border-color: #FF4B4B;
        box-shadow: 5px 15px 25px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #f8f9fb;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #eceeef;
    }
    </style>
""", unsafe_allow_html=True)

# Define the Deck
SUITS = ['♠️', '♣️', '♥️', '♦️']
VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
DECK = [f"{v}{s}" for s in SUITS for v in VALUES]

# ==========================================
# LOGIC & SIMULATION
# ==========================================
def calculate_theoretical_probability(hand_size):
    if hand_size > 48: return 1.0 
    return 1.0 - (math.comb(48, hand_size) / math.comb(52, hand_size))

@st.cache_data
def run_simulation(num_sims, hand_size, target_val):
    target_indices = [i for i, card in enumerate(DECK) if card.startswith(target_val)]
    # Efficiently generate all hands at once
    all_hands = [np.random.choice(52, hand_size, replace=False) for _ in range(num_sims)]
    counts = np.array([sum(1 for card in hand if card in target_indices) for hand in all_hands])
    return counts, all_hands

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("🎰 Control Room")
    target_card = st.selectbox("🎯 Target Rank to Find", VALUES, index=0)
    num_sims = st.select_slider("⚡ Simulation Intensity", options=[100, 1000, 5000, 10000, 25000], value=10000)
    hand_size = st.slider("🃏 Hand Size", 1, 12, 5)
    
    st.divider()
    st.write("**Current Targets in Deck:**")
    cols = st.columns(4)
    for i, s in enumerate(SUITS):
        color = "red" if s in ['♥️', '♦️'] else "#31333F"
        cols[i].markdown(f"<h3 style='color:{color}; text-align:center;'>{target_card}{s}</h3>", unsafe_allow_html=True)

# Run Logic
true_prob = calculate_theoretical_probability(hand_size)
counts, all_hands = run_simulation(num_sims, hand_size, target_card)
sim_prob = np.mean(counts > 0)

# ==========================================
# MAIN DASHBOARD
# ==========================================
st.title("🃏 Monte Carlo Card Analytics")

# --- ROW 1: SAMPLE HAND ---
st.subheader(f"🎴 Live Draw Sample (Looking for {target_card})")
card_cols = st.columns(hand_size)
sample_hand = all_hands[0]
for i, idx in enumerate(sample_hand):
    val = DECK[idx]
    is_target = val.startswith(target_card)
    color = "#FF4B4B" if "♥️" in val or "♦️" in val else "#31333F"
    border = "3px solid #FFD700" if is_target else "2px solid #EEE"
    card_cols[i].markdown(f'<div class="card-container" style="border: {border}; color: {color}; font-size: 30px; font-weight: bold;">{val}</div>', unsafe_allow_html=True)

# --- ROW 2: METRICS ---
st.markdown("<br>", unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Theoretical %", f"{true_prob:.2%}")
m2.metric("Simulation %", f"{sim_prob:.2%}")
m3.metric("Math Accuracy", f"{100 - (abs(true_prob-sim_prob)*100):.2f}%")
m4.metric("Avg Hits/Hand", f"{np.mean(counts):.2f}")

st.divider()

# --- ROW 3: ADVANCED PROBABILITY CHARTS (The New Additions) ---
st.subheader("🧪 Advanced Probability Deep-Dive")
tab1, tab2, tab3 = st.tabs(["📉 Convergence & Variance", "📊 Distribution Analytics", "🎲 Game Theory (Streaks)"])

with tab1:
    col_a, col_b = st.columns(2)
    # Fixed Step Size for Slicing
    step = max(1, num_sims // 200)
    cum_probs = np.cumsum(counts > 0) / np.arange(1, num_sims + 1)
    
    with col_a:
        # CHART 1: CONVERGENCE
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(y=cum_probs[::step], name="Simulated", line=dict(color='#FF4B4B', width=3)))
        fig_conv.add_hline(y=true_prob, line_dash="dash", line_color="green", annotation_text="Math Goal")
        fig_conv.update_layout(title="Law of Large Numbers Convergence", xaxis_title="Trials (Sampled)", yaxis_title="Probability")
        st.plotly_chart(fig_conv, width="stretch")
        
    with col_b:
        # CHART 2: RELATIVE VARIANCE (Z-Score Style)
        error_pct = ((cum_probs - true_prob) / true_prob) * 100
        fig_var = px.line(y=error_pct[::step], title="Relative Variance % (Simulation 'Wobble')", labels={'y': '% Deviation', 'x': 'Trials'})
        fig_var.add_hline(y=0, line_color="black")
        st.plotly_chart(fig_var, width="stretch")

with tab2:
    col_c, col_d = st.columns(2)
    with col_c:
        # CHART 3: HIT DENSITY
        hit_counts = pd.Series(counts).value_counts(normalize=True).sort_index()
        fig_hits = px.bar(x=hit_counts.index, y=hit_counts.values, title=f"Chance of Drawing Exactly X {target_card}s",
                          labels={'x': 'Count', 'y': 'Frequency'}, color=hit_counts.values)
        st.plotly_chart(fig_hits, width="stretch")
    with col_d:
        # CHART 4: CUMULATIVE INVENTORY
        fig_inv = px.area(y=np.cumsum(counts), title=f"Total {target_card}s Discovered Over Time", 
                          labels={'y': 'Total Found', 'x': 'Trial #'})
        st.plotly_chart(fig_inv, width="stretch")

with tab3:
    col_e, col_f = st.columns(2)
    # Calculate Losing Streaks
    streaks, current_streak = [], 0
    for win in (counts > 0):
        if not win: current_streak += 1
        else:
            if current_streak > 0: streaks.append(current_streak)
            current_streak = 0
            
    with col_e:
        # CHART 5: STREAK HISTOGRAM
        fig_streak = px.histogram(streaks, title="Losing Streak Frequency (Hands between Wins)",
                                  labels={'value': 'Wait Time'}, color_discrete_sequence=['#FF4B4B'])
        st.plotly_chart(fig_streak, width="stretch")
        
    with col_f:
        # CHART 6: EXPECTED VALUE GAUGE
        theoretical_ev = (4/52) * hand_size
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta", value = np.mean(counts),
            delta = {'reference': theoretical_ev},
            title = {'text': "Expected Value (EV) per Hand"},
            gauge = {'axis': {'range': [0, 1]}, 'bar': {'color': "#FF4B4B"}}
        ))
        st.plotly_chart(fig_gauge, width="stretch")

# --- ROW 4: SUIT ANALYTICS ---
st.divider()
st.subheader("🧩 Suit Bias & Performance")
successful_hands = [all_hands[i] for i in range(num_sims) if counts[i] > 0]
suit_counts = {'♠️': 0, '♣️': 0, '♥️': 0, '♦️': 0}
for hand in successful_hands:
    for idx in hand:
        for s in suit_counts:
            if s in DECK[idx]: suit_counts[s] += 1

fig_suit = px.funnel_area(names=list(suit_counts.keys()), values=list(suit_counts.values()), title="Suit Frequency in Winning Hands")
st.plotly_chart(fig_suit, width="stretch")