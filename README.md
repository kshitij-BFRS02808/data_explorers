
# 🚀 Company-Level Upsell & Capital Suggestion Engine

This project provides a **Streamlit-based interface** to predict smart upsell opportunities for each company using aggregated order data. The system uses rule-based thresholds and a trained RandomForest model to recommend:

- 📦 Insurance
- 💳 Engage360 (for high COD usage)
- 🚨 RTO Risk Alert
- 🏦 Capital Suggestion (for high GMV companies)

---

## 📸 UI Screenshots

### 📥 Upload CSV & See Insights

![Upload Screenshot](images/upload.png)

### 📊 Recommendations by Company

![Prediction Screenshot](images/predictions.png)

---

## 🧠 Prediction Logic

| Recommendation      | Rule |
|---------------------|------|
| 📦 Insurance         | `avg_total > 500` |
| 💳 Engage360         | `cod_ratio > 0.5` |
| 🚨 RTO Alert         | `rto_ratio + remote_ratio > 0.05` |
| 🏦 Capital           | `avg_total * order_count > 100000` |

---

## 🛠️ Features Used
- `avg_total`: Average order value  
- `cod_ratio`: Ratio of COD orders  
- `rto_ratio`: Ratio of orders with status 15/16  
- `remote_ratio`: Ratio of orders from remote pincodes  
- `order_count`: Number of orders per company  

---

## 📦 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Folder Structure

```
.
├── app.py
├── requirements.txt
├── README.md
└── images/
    ├── upload.png
    └── predictions.png
```

---

## 🧪 Sample Input Format (CSV Columns)

```csv
company_id,channel_id,shipping_method,sla,channel_order_id,pickup_address_id,
customer_city,customer_state,customer_state_code,customer_pincode,
customer_country,payment_method,net_total,total,status,status_code,
is_return,purpose_of_shipment,date_created_at,rto_prediction
```

---

## 📈 Model
The engine uses a `MultiOutputClassifier(RandomForestClassifier)` to train on all 4 labels simultaneously.

---

## 📬 Want to Contribute?
Feel free to fork and PR! This can be extended to include:
- XGBoost models
- Model versioning
- Real-time scoring API

