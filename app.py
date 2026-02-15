import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Risk System",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# Professional Highlighted Heading
# ---------------------------------------------------
st.markdown("""
<style>
.main-title {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    padding: 20px;
    border-radius: 12px;
    color: white;
    font-size: 36px;
    font-weight: 700;
    text-align: center;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
    color: #334155;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Supply Chain Risk System (SCRS)</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Disruption Prediction Dashboard</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("final_pipeline.pkl", "rb"))
    return model

model = load_model()

# ---------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Analytics"]
)

# ===================================================
# ================== PREDICTION PAGE ===============
# ===================================================
if menu == "Prediction":

    st.header("Enter Order Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        type_val = st.selectbox("Transaction Type", ["Debit", "Transfer", "Cash"])
        customer_segment = st.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"])
        market = st.selectbox("Market", ["US", "EU", "APAC"])

    with col2:
        order_region = st.text_input("Order Region", "East")
        shipping_mode = st.selectbox("Shipping Mode", ["Standard", "Second Class", "First Class", "Same Day"])

    with col3:
        discount_rate = st.number_input("Discount Rate (0-1)", 0.0, 1.0, 0.1)
        product_price = st.number_input("Product Price", 0.0, 10000.0, 100.0)
        quantity = st.number_input("Quantity", 1, 1000, 1)
        profit = st.number_input("Profit Per Order", -10000.0, 10000.0, 10.0)
        shipping_delay = st.number_input("Shipping Delay Days", -10.0, 30.0, 0.0)

    sales = product_price * quantity

    if st.button("🔍 Predict Risk"):

        input_dict = {
            "Type": type_val,
            "Customer Segment": customer_segment,
            "Market": market,
            "Order Region": order_region,
            "Shipping Mode": shipping_mode,
            "Order Item Discount Rate": discount_rate,
            "Order Item Product Price": product_price,
            "Order Item Quantity": quantity,
            "Sales": sales,
            "Order Profit Per Order": profit,
            "Shipping_Delay_Days": shipping_delay
        }

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)[0]

        risk_map = {
            0: "🟢 Low Risk",
            1: "🟡 Medium Risk",
            2: "🔴 High Risk"
        }

        result = risk_map.get(prediction, "Unknown")

        st.subheader("Prediction Result")
        st.success(result)

        # ---------------------------------------------------
        # AI EXPLANATION SECTION
        # ---------------------------------------------------
        st.subheader("🤖 Prediction Rationale")

        explanation = ""

        if prediction == 2:
            explanation = """
            The model predicts **High Risk** because:
            - Higher shipping delay increases disruption probability.
            - Low profit margins reduce supply chain flexibility.
            - High discount rate may indicate unstable demand.
            - Certain shipping modes and regions historically show higher disruption.
            """

        elif prediction == 1:
            explanation = """
            The model predicts **Medium Risk** due to:
            - Moderate shipping delay.
            - Balanced profit and discount levels.
            - Regional risk exposure.
            """

        else:
            explanation = """
            The model predicts **Low Risk** because:
            - Minimal shipping delay.
            - Stable profit margin.
            - Balanced discount strategy.
            - Low historical disruption for selected region and shipping mode.
            """

        st.info(explanation)

        # ---------------------------------------------------
        # Save History
        # ---------------------------------------------------
        if not os.path.exists("history.csv"):
            pd.DataFrame(columns=["Timestamp", "Prediction"]).to_csv("history.csv", index=False)

        history_df = pd.read_csv("history.csv")
        new_row = pd.DataFrame([{
            "Timestamp": datetime.now(),
            "Prediction": result
        }])

        history_df = pd.concat([history_df, new_row], ignore_index=True)
        history_df.to_csv("history.csv", index=False)

        st.download_button(
            "Download Result",
            data=input_df.to_csv(index=False),
            file_name="prediction_result.csv"
        )

# ===================================================
# ================== ANALYTICS PAGE =================
# ===================================================
elif menu == "Analytics":

    st.header("Prediction Analytics")

    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv")

        st.subheader("Prediction History")
        st.dataframe(df)

        # 🔴 Clear History Button
        if st.button("🗑 Clear Prediction History"):
            os.remove("history.csv")
            st.success("Prediction history deleted successfully!")
            st.rerun()

        st.subheader("Risk Distribution")
        st.bar_chart(df["Prediction"].value_counts())

    else:
        st.warning("No prediction history available.")


