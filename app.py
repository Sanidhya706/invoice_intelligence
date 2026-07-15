import streamlit as st
import pandas as pd

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Vendor Invoice Intelligence Portal")

st.markdown("""
Welcome to the **Vendor Invoice Intelligence Portal**.

This application uses Machine Learning to:

- Predict Freight Cost
- Predict whether an Invoice requires Manual Review

Select a module from the sidebar to get started.
""")

st.sidebar.title("Navigation")

app_mode = st.sidebar.radio(
    "Select Module",
    (
        "Freight Cost Prediction",
        "Invoice Manual Approval Prediction"
    )
)

# ===========================================================
# Freight Cost Prediction
# ===========================================================

if app_mode == "Freight Cost Prediction":

    st.header("🚚 Freight Cost Prediction")

    st.markdown("""
Predict the expected freight cost based on the invoice amount.
""")

    col1, col2 = st.columns(2)

    with col1:

        dollars = st.number_input(
            "Invoice Amount ($)",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

    with col2:

        st.metric(
            label="Prediction Model",
            value="Linear Regression"
        )

    if st.button("Predict Freight Cost"):

        input_data = {
            "Dollars": [dollars]
        }

        result = predict_freight_cost(input_data)

        predicted_cost = result["Predicted_Freight"].iloc[0]

        st.success(
            f"Predicted Freight Cost: ${predicted_cost:.2f}"
        )

        st.markdown("---")

        st.subheader("Prediction Details")

        st.dataframe(result)

        st.markdown("""
### Business Impact

This prediction helps businesses:

- Estimate logistics cost before shipment.
- Compare expected freight cost with actual freight cost.
- Detect unusually high freight charges.
- Improve budgeting and vendor management.
""")

# ===========================================================
# Invoice Manual Approval Prediction
# ===========================================================

elif app_mode == "Invoice Manual Approval Prediction":

    st.header("📝 Invoice Manual Approval Prediction")

    st.markdown("""
Predict whether a vendor invoice should be **Flagged for Manual Review**.

Enter the invoice details below.
""")

    col1, col2 = st.columns(2)

    with col1:

        invoice_quantity = st.number_input(
            "Invoice Quantity",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        invoice_dollars = st.number_input(
            "Invoice Amount ($)",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

        freight = st.number_input(
            "Freight Cost ($)",
            min_value=0.0,
            value=100.0,
            step=10.0
        )

    with col2:

        total_item_quantity = st.number_input(
            "Total Item Quantity",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        total_item_dollars = st.number_input(
            "Total Item Amount ($)",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

        st.metric(
            label="Prediction Model",
            value="Random Forest Classifier"
        )

    if st.button("Predict Invoice Status"):

        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars],
        }

        result = predict_invoice_flag(input_data)

        prediction = result["Predicted_Flag"].iloc[0]

        if prediction == 1:

            st.error("⚠️ Invoice should be Flagged for Manual Review.")

        else:

            st.success("✅ Invoice looks Normal. No Manual Review Required.")

        st.markdown("---")

        st.subheader("Prediction Details")

        st.dataframe(result)

        st.markdown("""
### Business Impact

This prediction helps organizations:

- Detect suspicious invoices automatically.
- Reduce manual verification time.
- Minimize financial risk.
- Improve invoice processing efficiency.
- Assist finance teams in prioritizing high-risk invoices.
""")