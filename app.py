
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

st.set_page_config(page_title="Company Upsell & Capital Suggestor", layout="wide")

# === Sidebar info ===
st.sidebar.header("🧠 Model Thresholds")
st.sidebar.markdown("""
- **📦 Insurance**: `avg_total > 500`  
- **💳 Engage360**: `cod_ratio > 0.5`  
- **🚨 RTO Alert**: `(rto_ratio + remote_ratio) > 0.05`  
- **🏦 Capital**: `total_order_value > 100000`  
""")

st.sidebar.header("📊 Features Used")
st.sidebar.markdown("""
- `avg_total`: Average order value  
- `cod_ratio`: Ratio of COD orders  
- `rto_ratio`: RTO-prone order status ratio  
- `remote_ratio`: Remote pincode ratio  
- `order_count`: Total order volume  
""")

# === Title ===
st.title("📈 Company-Level Upsell & Capital Suggestion Engine")

# === File Upload ===
uploaded_file = st.file_uploader("📤 Upload your order CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🧾 Raw Data Preview")
    st.dataframe(df.head())

    # Remote pincode mock list
    remote_pincodes = {"173212", "796007", "431002"}

    # Aggregate per company
    grouped = df.groupby("company_id").agg(
        avg_total=('total', 'mean'),
        cod_ratio=('payment_method', lambda x: (x.str.lower() == 'cod').mean()),
        rto_ratio=('status', lambda x: x.isin([15, 16]).mean()),
        remote_ratio=('customer_pincode', lambda x: x.astype(str).isin(remote_pincodes).mean()),
        order_count=('channel_order_id', 'count')
    ).reset_index()

    # Add recommendation labels
    grouped['recommend_insurance'] = grouped['avg_total'] > 500
    grouped['recommend_engage'] = grouped['cod_ratio'] > 0.5
    grouped['recommend_rto_alert'] = (grouped['rto_ratio'] + grouped['remote_ratio']) > 0.05
    grouped['recommend_capital'] = grouped['avg_total'] * grouped['order_count'] > 100000

    # Train model
    X = grouped[['avg_total', 'cod_ratio', 'rto_ratio', 'remote_ratio', 'order_count']]
    Y = grouped[['recommend_insurance', 'recommend_engage', 'recommend_rto_alert', 'recommend_capital']]
    model = MultiOutputClassifier(RandomForestClassifier(random_state=42))
    model.fit(X, Y)
    preds = model.predict(X)

    pred_df = pd.DataFrame(preds, columns=['pred_insurance', 'pred_engage', 'pred_rto_alert', 'pred_capital'])
    result = pd.concat([grouped[['company_id']], pred_df], axis=1)

    st.subheader("✅ Model Predictions")
    st.dataframe(result)

    # Download button
    csv = result.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions", csv, "company_predictions.csv", "text/csv")
else:
    st.info("Upload a file to begin analysis.")
