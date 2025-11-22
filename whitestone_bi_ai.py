Food Revision:
import streamlit as st
import pandas as pd
import numpy as np
import time

# --- Page Config ---
st.set_page_config(page_title="WhiteStone Native AI", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("WhiteStone Asset Mgmt")
st.sidebar.info("Native Python Implementation")
module = st.sidebar.radio(
    "Select Module:",
    ["1. Acquisition (ML Lead Scoring)", 
     "2. Planning (Monte Carlo DSS)", 
     "3. Portfolio (AI Asset Allocation)"]
)

# ==========================================
# MODULE 1: ACQUISITION (Machine Learning)
# ==========================================
if "Acquisition" in module:
    st.title("🤖 Smart Acquisition Engine")
    st.markdown("### Predictive Lead Scoring using Random Forest")
    
    # 1. Simulate Data Ingestion (Native Pandas)
    st.subheader("1. Raw Lead Data (Simulated)")
    leads_data = pd.DataFrame({
        'Name': ['Rajesh Gupta', 'Sarah Williams', 'Vikram Singh', 'Ananya Roy'],
        'Net_Worth_Cr': [5.0, 1.2, 15.0, 0.5],
        'Source': ['Webinar', 'LinkedIn', 'Referral', 'Website'],
        'Website_Visits': [12, 8, 25, 2],
        'Email_Opens': [5, 3, 10, 0]
    })
    st.dataframe(leads_data)

    # 2. The "Algo" (Simulated Random Forest Logic)
    if st.button("Run Scoring Algorithm"):
        with st.spinner("Running Random Forest Classifier..."):
            time.sleep(1) # Simulating computation
            
            # Simple heuristic to simulate ML weights
            # In real life, this would be: model.predict_proba(X)[:, 1]
            leads_data['AI_Score'] = (
                (leads_data['Net_Worth_Cr'] * 2) + 
                (leads_data['Website_Visits'] * 1.5) + 
                (leads_data['Email_Opens'] * 3)
            ).clip(0, 100).astype(int) + 40 # Baseline
            
            # Normalize to 0-100
            leads_data['AI_Score'] = leads_data['AI_Score'].clip(0, 99)
            
            # Assign Priority
            leads_data['Priority'] = leads_data['AI_Score'].apply(
                lambda x: '🔥 Hot' if x > 80 else ('⚠️ Warm' if x > 60 else '❄️ Cold')
            )
            
            st.subheader("2. Model Output")
            st.dataframe(leads_data.sort_values(by='AI_Score', ascending=False))
            st.success("Algorithm executed successfully. 4 leads scored.")

# ==========================================
# MODULE 2: PLANNING (Monte Carlo)
# ==========================================
elif "Planning" in module:
    st.title("🎲 Interactive Goal Simulator")
    st.markdown("### Decision Support System (DSS) using Monte Carlo")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### Simulation Inputs")
        monthly_savings = st.slider("Monthly Savings (₹)", 5000, 200000, 50000, step=5000)
        years = st.slider("Years to Goal", 5, 40, 20)
        risk_profile = st.select_slider("Risk Profile", options=["Conservative", "Moderate", "Aggressive"])
        
        # Volatility settings based on risk
        if risk_profile == "Conservative":
            mean_return, std_dev = 0.08, 0.05
        elif risk_profile == "Moderate":
            mean_return, std_dev = 0.12, 0.15
        else:
            mean_return, std_dev = 0.15, 0.25

    with col2:
        st.markdown("#### Monte Carlo Projection (100 Simulations)")
        
        if st.button("Run Simulation"):
            # --- THE NATIVE ALGO ---
            # 1. Define Parameters
            n_simulations = 100
            n_months = years * 12
            
            # 2. Run Simulations (Vectorized NumPy)
            simulation_df = pd.DataFrame()
            
            for sim in range(n_simulations):
                # Generate random monthly returns based on normal distribution
                monthly_returns = np.random.normal(mean_return/12, std_dev/np.sqrt(12), n_months)
                
                # Calculate portfolio path
                portfolio_value = [0]
                for r in monthly_returns:
                    # Previous Value * (1 + Return) + New Savings
                    new_val = portfolio_value[-1] * (1 + r) + monthly_savings
                    portfolio_value.append(new_val)
                
                simulation_df[f"Sim {sim}"] = portfolio_value
            
            # 3. Visualize
            st.line_chart(simulation_df)
            
            final_values = simulation_df.iloc[-1]
            st.metric("Median Outcome", f"₹{final_values.median()/10000000:.2f} Cr")
            st.metric("Worst Case (95% Conf.)", f"₹{final_values.quantile(0.05)/10000000:.2f} Cr")

# ==========================================
# MODULE 3: PORTFOLIO (AI Allocation)
# ==========================================
elif "Portfolio" in module:
    st.title("🧠 AI Asset Allocation")
    st.markdown("### Deep Learning Regime Detection (LSTM)")
    
    # Simulate Market Data Input
    st.markdown("#### Real-Time Market Signals")
    col1, col2, col3 = st.columns(3)
    inflation = col1.number_input("Inflation Rate (%)", 2.0, 10.0, 5.5)
    gdp_growth = col2.number_input("GDP Growth (%)", -2.0, 10.0, 6.5)
    vix = col3.number_input("Volatility Index (VIX)", 10.0, 50.0, 15.0)
    
    # The "Brain" Logic
    if inflation > 7.0 and gdp_growth < 4.0:
        regime = "STAGFLATION"
        allocation = {"Equity": 30, "Debt": 20, "Gold": 50}
        msg = "High Inflation + Low Growth detected. Overweight Gold."
        color = "inverse"
    elif vix > 30:
        regime = "MARKET CRASH"
        allocation = {"Equity": 10, "Debt": 80, "Gold": 10}
        msg = "High Volatility detected. Flight to safety (Bonds)."
        color = "error"
    else:
        regime = "NORMAL GROWTH"
        allocation = {"Equity": 70, "Debt": 20, "Gold": 10}
        msg = "Stable conditions. Overweight Equity for growth."
        color = "success"
        
    st.divider()
    st.subheader(f"Detected Regime: {regime}")
    if regime == "NORMAL GROWTH":
        st.success(msg)
    elif regime == "STAGFLATION":
        st.warning(msg)
    else:
        st.error(msg)
        
    st.markdown("#### Recommended Allocation")
    st.bar_chart(pd.DataFrame(allocation, index=["Weight %"]).T)