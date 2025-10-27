import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


st.set_page_config(
    page_title="SmartShop Inventory",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ---------- Main Header ---------- */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* ---------- Metrics Cards ---------- */
    div[data-testid="stMetric"] {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(0, 0, 0, 0.03);
        transition: transform 0.3s ease, box-shadow 0.3s ease, background 0.3s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-8px) scale(1.03);
        background: #ffffff;
        box-shadow: 0 8px 16px rgba(255, 223, 100, 0.5), 0 12px 24px rgba(0, 0, 0, 0.12);
    }

    div[data-testid="stMetric"] label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #6b7280;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.875rem;
        font-weight: 700;
        color: #374151;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 20px;
        background-color: #e5e7eb;
        border-radius: 10px;
        color: #374151;
        box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease, background-color 0.3s ease;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 0 12px rgba(255, 223, 100, 0.5), 4px 4px 8px rgba(0, 0, 0, 0.12);
        background-color: #d1d5db;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #667eea, #764ba2);
        color: #ffffff;
        box-shadow: inset 2px 2px 5px rgba(0, 0, 0, 0.2);
    }

    /* ---------- DataFrame ---------- */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    }

    /* ---------- Sidebar Buttons ---------- */
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(145deg, #667eea, #764ba2);
        border: none;
        border-radius: 10px;
        box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        font-weight: 600;
        color: #ffffff;
        cursor: pointer;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 0 12px rgba(255, 223, 100, 0.5), 4px 4px 10px rgba(0, 0, 0, 0.25);
    }

    section[data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(0);
        box-shadow: inset 2px 2px 5px rgba(0, 0, 0, 0.3);
    }
    
    /* ---------- Expiry Alerts ---------- */
    .expiry-critical {
        background-color: #fee2e2;
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .expiry-warning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    .expiry-soon {
        background-color: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# -------------------- HELPER FUNCTIONS --------------------
def calculate_dynamic_price(base_price, expiry_date, current_date):
    """Calculate price based on days until expiry"""
    try:
        expiry = pd.to_datetime(expiry_date)
        current = pd.to_datetime(current_date)
        days_left = (expiry - current).days
        
        if days_left < 0:  # Expired
            return base_price * 0.1, 90  # 90% off
        elif days_left <= 2:  # Critical (0-2 days)
            return base_price * 0.3, 70  # 70% off
        elif days_left <= 5:  # Urgent (3-5 days)
            return base_price * 0.5, 50  # 50% off
        elif days_left <= 10:  # Warning (6-10 days)
            return base_price * 0.7, 30  # 30% off
        elif days_left <= 15:  # Soon (11-15 days)
            return base_price * 0.85, 15  # 15% off
        else:  # Fresh
            return base_price, 0
    except Exception as e:
        return base_price, 0


def get_expiry_status(days_left):
    """Get expiry status and color"""
    if days_left < 0:
        return "EXPIRED", "🔴"
    elif days_left <= 2:
        return "CRITICAL", "🔴"
    elif days_left <= 5:
        return "URGENT", "🟠"
    elif days_left <= 10:
        return "WARNING", "🟡"
    elif days_left <= 15:
        return "SOON", "🔵"
    else:
        return "FRESH", "🟢"


def get_current_inventory(conn):
    """Get current inventory state from my_table"""
    query = """
    WITH LatestRecords AS (
        SELECT 
            product_id,
            product_name,
            stock_left,
            expiry_date,
            base_price,
            date,
            ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY date DESC, ROWID DESC) as rn
        FROM my_table
    ),
    TotalSold AS (
        SELECT 
            product_id,
            SUM(stock_sold) as stock_sold_total
        FROM my_table
        GROUP BY product_id
    )
    SELECT 
        lr.product_id,
        lr.product_name,
        lr.stock_left,
        lr.expiry_date,
        lr.base_price,
        COALESCE(ts.stock_sold_total, 0) as stock_sold_total
    FROM LatestRecords lr
    LEFT JOIN TotalSold ts ON lr.product_id = ts.product_id
    WHERE lr.rn = 1
    ORDER BY lr.product_id
    """
    return pd.read_sql(query, conn)


# -------------------- DATABASE SETUP --------------------
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "my_database.db")  

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# Create table if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS my_table (
        product_id INTEGER,
        product_name TEXT,
        date TEXT,
        day TEXT,
        stock_sold INTEGER,
        stock_left INTEGER,
        expiry_date TEXT,
        base_price REAL,
        adjusted_price REAL,
        discount_percent INTEGER
    )
""")

conn.commit()


# -------------------- SIDEBAR --------------------
st.sidebar.markdown("## Inventory Operations")

st.sidebar.markdown("---")
selected_date = st.sidebar.date_input(
    "Transaction Date", 
    value=datetime.now(),
    help="Select the date for this transaction"
)
transaction_date = selected_date.strftime("%Y-%m-%d")
transaction_day = selected_date.strftime("%A")

st.sidebar.info(f"📅 {transaction_date}\n\n📆 {transaction_day}")

st.sidebar.markdown("---")

operation = st.sidebar.radio(
    "Operation Type",
    ["Sell", "Buy"],
    help="Choose whether to sell or replenish stock."
)

with st.sidebar.expander("Product Details", expanded=True):
    product_id = st.number_input("Product ID", min_value=1, step=1)
    quantity = st.number_input("Quantity", min_value=0, step=1)
    
    if operation == "Buy":
        expiry_date = st.date_input("Expiry Date", value=datetime.now() + timedelta(days=30))
        expiry_str = expiry_date.strftime("%Y-%m-%d")
        base_price = st.number_input("Base Price per Unit (₹)", min_value=0.0, step=1.0, value=10.0)

action_btn = st.sidebar.button("Confirm Transaction", use_container_width=True, type="primary")

# -------------------- OPERATIONS --------------------
if action_btn:
    if quantity == 0:
        st.sidebar.error("Quantity must be greater than 0!")
    else:
        # Get current state of the product
        cursor.execute("""
            SELECT product_name, stock_left, expiry_date, base_price
            FROM my_table
            WHERE product_id = ?
            ORDER BY date DESC, ROWID DESC
            LIMIT 1
        """, (product_id,))
        
        current_state = cursor.fetchone()
        
        if operation == "Sell":
            if not current_state:
                st.sidebar.warning("Cannot sell: Product ID not found in inventory. Add stock first!")
            else:
                product_name, current_stock, expiry_date_db, base_price_db = current_state
                
                if quantity > current_stock:
                    st.sidebar.warning(f"Cannot sell {quantity} units. Only {current_stock} available.")
                else:
                    stock_left = current_stock - quantity
                    
                    # Calculate dynamic price
                    adjusted_price, discount = calculate_dynamic_price(base_price_db, expiry_date_db, transaction_date)
                    total_revenue = adjusted_price * quantity

                    cursor.execute("""
                        INSERT INTO my_table (product_id, product_name, date, day, stock_sold, stock_left, 
                                            expiry_date, base_price, adjusted_price, discount_percent)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (product_id, product_name, transaction_date, transaction_day, quantity, stock_left,
                          expiry_date_db, base_price_db, adjusted_price, discount))

                    if discount > 0:
                        st.sidebar.success(f"""
                        ✅ Sold {quantity} units of **{product_name}**
                        
                        💰 Base Price: ₹{base_price_db:.2f}
                        🏷️ Adjusted Price: ₹{adjusted_price:.2f} ({discount}% off)
                        💵 Total Revenue: ₹{total_revenue:.2f}
                        
                        📦 Stock Remaining: **{stock_left}**
                        """)
                    else:
                        st.sidebar.success(f"""
                        ✅ Sold {quantity} units of **{product_name}**
                        
                        💰 Price: ₹{base_price_db:.2f}
                        💵 Total Revenue: ₹{total_revenue:.2f}
                        
                        📦 Stock Remaining: **{stock_left}**
                        """)
                    
                    conn.commit()
                    st.rerun()

        elif operation == "Buy":
            if not current_state:
                # New product
                product_name = st.sidebar.text_input("Product Name (Required for new products)")
                if not product_name:
                    st.sidebar.error("Please enter a product name!")
                else:
                    cursor.execute("""
                        INSERT INTO my_table (product_id, product_name, date, day, stock_sold, stock_left,
                                            expiry_date, base_price, adjusted_price, discount_percent)
                        VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, 0)
                    """, (product_id, product_name, transaction_date, transaction_day, quantity, 
                          expiry_str, base_price, base_price))
                    
                    st.sidebar.success(f"""
                    ✅ Added new product: **{product_name}**
                    
                    📦 Quantity: {quantity}
                    💰 Base Price: ₹{base_price:.2f}
                    📅 Expiry: {expiry_str}
                    """)
                    conn.commit()
                    st.rerun()
            else:
                product_name = current_state[0]
                stock_left = current_state[1] + quantity

                cursor.execute("""
                    INSERT INTO my_table (product_id, product_name, date, day, stock_sold, stock_left,
                                        expiry_date, base_price, adjusted_price, discount_percent)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, 0)
                """, (product_id, product_name, transaction_date, transaction_day, stock_left,
                      expiry_str, base_price, base_price))

                st.sidebar.success(f"""
                ✅ Added {quantity} units of **{product_name}**
                
                📦 Total Stock: {stock_left}
                💰 Base Price: ₹{base_price:.2f}
                📅 Expiry: {expiry_str}
                """)
                conn.commit()
                st.rerun()

# -------------------- MAIN DASHBOARD --------------------
st.markdown('<h1 class="main-header">SmartShop Inventory Dashboard</h1>', unsafe_allow_html=True)

df_transactions = pd.read_sql("SELECT * FROM my_table ORDER BY date DESC, ROWID DESC", conn)
df_inventory = get_current_inventory(conn)

# Add expiry calculations to inventory
if not df_inventory.empty:
    df_inventory['expiry_date'] = pd.to_datetime(df_inventory['expiry_date'], errors='coerce')
    df_inventory['expiry_date'].fillna(pd.Timestamp.now() + timedelta(days=30), inplace=True)
    df_inventory['base_price'].fillna(10.0, inplace=True)
    
    df_inventory['days_to_expiry'] = (df_inventory['expiry_date'] - pd.to_datetime(selected_date)).dt.days
    df_inventory['adjusted_price'] = df_inventory.apply(
        lambda row: calculate_dynamic_price(row['base_price'], row['expiry_date'], selected_date)[0], axis=1
    )
    df_inventory['discount_percent'] = df_inventory.apply(
        lambda row: calculate_dynamic_price(row['base_price'], row['expiry_date'], selected_date)[1], axis=1
    )

# -------------------- EXPIRY ALERTS --------------------
if not df_inventory.empty:
    critical = df_inventory[df_inventory['days_to_expiry'] <= 2]
    urgent = df_inventory[(df_inventory['days_to_expiry'] > 2) & (df_inventory['days_to_expiry'] <= 5)]
    warning = df_inventory[(df_inventory['days_to_expiry'] > 5) & (df_inventory['days_to_expiry'] <= 10)]
    
    if not critical.empty:
        st.markdown('<div class="expiry-critical">', unsafe_allow_html=True)
        st.error(f"🔴 **CRITICAL: {len(critical)} product(s) expiring in ≤2 days!** (70% discount applied)")
        for _, row in critical.iterrows():
            st.write(f"• {row['product_name']}: {row['days_to_expiry']} days left | ₹{row['base_price']:.2f} → ₹{row['adjusted_price']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if not urgent.empty:
        st.markdown('<div class="expiry-warning">', unsafe_allow_html=True)
        st.warning(f"🟠 **URGENT: {len(urgent)} product(s) expiring in 3-5 days** (50% discount applied)")
        for _, row in urgent.iterrows():
            st.write(f"• {row['product_name']}: {row['days_to_expiry']} days left | ₹{row['base_price']:.2f} → ₹{row['adjusted_price']:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    if not warning.empty:
        st.markdown('<div class="expiry-soon">', unsafe_allow_html=True)
        st.info(f"🟡 **WARNING: {len(warning)} product(s) expiring in 6-10 days** (30% discount applied)")
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------- KPI METRICS --------------------
if not df_inventory.empty:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Products", value=len(df_inventory))
    
    with col2:
        total_stock = df_inventory['stock_left'].sum()
        st.metric(label="Total Stock", value=f"{total_stock:,}")
    
    with col3:
        total_sold = df_inventory['stock_sold_total'].sum()
        st.metric(label="Total Sold", value=f"{total_sold:,}")
    
    with col4:
        if not df_transactions.empty and 'adjusted_price' in df_transactions.columns:
            df_trans_temp = df_transactions.copy()
            df_trans_temp['date'] = pd.to_datetime(df_trans_temp['date'], errors='coerce')
            today = selected_date
            today_revenue = df_trans_temp[df_trans_temp['date'] == pd.to_datetime(today)]
            if not today_revenue.empty:
                revenue = (today_revenue['adjusted_price'] * today_revenue['stock_sold']).sum()
                st.metric(label="Today's Revenue", value=f"₹{revenue:,.2f}")
            else:
                st.metric(label="Today's Revenue", value="₹0.00")
        else:
            st.metric(label="Today's Revenue", value="₹0.00")

st.markdown("---")

tabs = st.tabs(["Overview", "Expiry Management", "AI Analytics", "Transactions", "Inventory"])

# -------------------- OVERVIEW TAB --------------------
with tabs[0]:
    if not df_inventory.empty and not df_transactions.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Stock Levels by Product")
            chart_data = df_inventory.set_index("product_name")['stock_left']
            st.bar_chart(chart_data, height=400)
        
        with col2:
            st.markdown("### Total Sales by Product")
            chart_data = df_inventory.set_index("product_name")['stock_sold_total']
            st.bar_chart(chart_data, height=400)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Price Adjustments Overview")
            discount_dist = df_inventory['discount_percent'].value_counts().sort_index()
            st.bar_chart(discount_dist, height=350)
        
        with col2:
            st.markdown("### Revenue: Base vs Adjusted")
            if not df_transactions.empty and 'base_price' in df_transactions.columns:
                df_trans_revenue = df_transactions[df_transactions['stock_sold'] > 0].copy()
                if not df_trans_revenue.empty:
                    revenue_data = pd.DataFrame({
                        'Base Revenue': [(df_trans_revenue['base_price'] * df_trans_revenue['stock_sold']).sum()],
                        'Actual Revenue': [(df_trans_revenue['adjusted_price'] * df_trans_revenue['stock_sold']).sum()]
                    })
                    st.bar_chart(revenue_data, height=350)
    else:
        st.info("No data available. Start by adding products to your inventory!")

# -------------------- EXPIRY MANAGEMENT TAB --------------------
with tabs[1]:
    st.markdown("### Expiry-Based Pricing Strategy")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Discount Tiers")
        st.markdown("""
        - 🟢 **Fresh** (>15 days): 0% off
        - 🔵 **Soon** (11-15 days): 15% off
        - 🟡 **Warning** (6-10 days): 30% off
        - 🟠 **Urgent** (3-5 days): 50% off
        - 🔴 **Critical** (0-2 days): 70% off
        - ⚫ **Expired**: 90% off
        """)
    
    with col2:
        if not df_inventory.empty:
            st.markdown("#### Products by Expiry Status")
            df_inv_display = df_inventory.copy()
            df_inv_display['Status'] = df_inv_display['days_to_expiry'].apply(
                lambda x: get_expiry_status(x)[1] + " " + get_expiry_status(x)[0]
            )
            
            status_counts = df_inv_display.groupby('Status').size()
            st.bar_chart(status_counts, height=300)
    
    st.markdown("---")
    
    if not df_inventory.empty:
        st.markdown("### Detailed Expiry Status")
        
        df_expiry_display = df_inventory[[
            'product_id', 'product_name', 'stock_left', 'expiry_date', 
            'days_to_expiry', 'base_price', 'adjusted_price', 'discount_percent'
        ]].copy()
        
        df_expiry_display['Status'] = df_expiry_display['days_to_expiry'].apply(
            lambda x: get_expiry_status(x)[1] + " " + get_expiry_status(x)[0]
        )
        
        df_expiry_display = df_expiry_display.sort_values('days_to_expiry')
        df_expiry_display['expiry_date'] = df_expiry_display['expiry_date'].dt.strftime('%Y-%m-%d')
        df_expiry_display.columns = ['ID', 'Product', 'Stock', 'Expiry Date', 'Days Left', 
                                      'Base Price', 'Current Price', 'Discount %', 'Status']
        
        st.dataframe(df_expiry_display, use_container_width=True, height=400)

# -------------------- AI ANALYTICS TAB --------------------
with tabs[2]:
    if not df_transactions.empty and len(df_transactions) >= 10:
        df = df_transactions.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['product_id_code'] = df['product_id'].astype('category').cat.codes

        # ========== FORECASTING (Regressor) ==========
        st.markdown("### 📈 Sales Forecasting (Regression)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Next Month Sales Predictions")
            forecast_data = []
            
            for prod in df['product_name'].unique():
                prod_data = df[df['product_name']==prod]
                if len(prod_data) >= 5:
                    X = prod_data[['product_id_code', 'month', 'day_of_month']]
                    y = prod_data['stock_sold']
                    
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                    model.fit(X, y)
                    
                    next_month = (selected_date.month % 12) + 1
                    pred = model.predict([[prod_data['product_id_code'].iloc[0], next_month, 15]])
                    forecast_data.append({
                        'Product': prod,
                        'Predicted_Sales': max(0, int(pred[0] * 30))
                    })
            
            if forecast_data:
                forecast_df = pd.DataFrame(forecast_data)
                st.bar_chart(forecast_df.set_index('Product')['Predicted_Sales'], height=350)
            else:
                st.info("Need more data for forecasting")
        
        with col2:
            st.markdown("#### Discount Impact on Sales")
            if 'discount_percent' in df.columns:
                discount_sales = df.groupby('discount_percent')['stock_sold'].sum().sort_index()
                st.bar_chart(discount_sales, height=350)
        
        st.markdown("---")
        
        # ========== CLASSIFICATION: Sales Performance Prediction ==========
        st.markdown("### 🎯 Sales Performance Classification")
        
        # Prepare classification data
        df_class = df[df['stock_sold'] > 0].copy()
        
        if len(df_class) >= 15:  # Need sufficient data for classification
            # Create sales performance categories
            df_class['sales_performance'] = pd.cut(
                df_class['stock_sold'], 
                bins=[0, df_class['stock_sold'].quantile(0.33), 
                      df_class['stock_sold'].quantile(0.66), 
                      df_class['stock_sold'].max()],
                labels=['Low', 'Medium', 'High'],
                include_lowest=True
            )
            
            # Features for classification
            X_class = df_class[['product_id_code', 'month', 'day_of_month', 'discount_percent']]
            y_class = df_class['sales_performance']
            
            # Train-test split
            if len(X_class) >= 20:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_class, y_class, test_size=0.3, random_state=42
                )
                
                # Train classifier
                classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                classifier.fit(X_train, y_train)
                
                # Predictions
                y_pred = classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Classification Accuracy", f"{accuracy*100:.1f}%")
                
                with col2:
                    # Feature importance
                    feature_importance = pd.DataFrame({
                        'Feature': ['Product', 'Month', 'Day', 'Discount'],
                        'Importance': classifier.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.markdown("**Top Feature**")
                    st.info(f"🏆 {feature_importance.iloc[0]['Feature']}")
                
                with col3:
                    # Predicted distribution
                    pred_counts = pd.Series(y_pred).value_counts()
                    st.markdown("**Predicted Classes**")
                    for idx, count in pred_counts.items():
                        st.write(f"• {idx}: {count}")
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Feature Importance")
                    st.bar_chart(feature_importance.set_index('Feature')['Importance'], height=300)
                
                with col2:
                    st.markdown("#### Performance Distribution")
                    performance_dist = df_class['sales_performance'].value_counts()
                    st.bar_chart(performance_dist, height=300)
                
                # Predictions for each product
                st.markdown("---")
                st.markdown("#### Product Performance Predictions (Next Month)")
                
                product_predictions = []
                for prod in df['product_name'].unique():
                    prod_data = df_class[df_class['product_name'] == prod]
                    if len(prod_data) >= 3:
                        # Predict for next month with average discount
                        next_month = (selected_date.month % 12) + 1
                        avg_discount = prod_data['discount_percent'].mean()
                        
                        pred_features = [[
                            prod_data['product_id_code'].iloc[0],
                            next_month,
                            15,  # mid-month
                            avg_discount
                        ]]
                        
                        predicted_class = classifier.predict(pred_features)[0]
                        confidence = classifier.predict_proba(pred_features).max()
                        
                        product_predictions.append({
                            'Product': prod,
                            'Predicted Performance': predicted_class,
                            'Confidence': f"{confidence*100:.1f}%"
                        })
                
                if product_predictions:
                    pred_df = pd.DataFrame(product_predictions)
                    st.dataframe(pred_df, use_container_width=True)
                
            else:
                st.info("Need at least 20 transactions for train-test split")
        else:
            st.info("Need at least 15 sales transactions for classification analysis")
        
        st.markdown("---")
        
        # ========== REVENUE ANALYSIS ==========
        st.markdown("### 💰 Revenue Analysis")
        if 'adjusted_price' in df.columns and 'base_price' in df.columns:
            df_revenue = df[df['stock_sold'] > 0].copy()
            if not df_revenue.empty:
                df_revenue['base_revenue'] = df_revenue['base_price'] * df_revenue['stock_sold']
                df_revenue['actual_revenue'] = df_revenue['adjusted_price'] * df_revenue['stock_sold']
                df_revenue['revenue_loss'] = df_revenue['base_revenue'] - df_revenue['actual_revenue']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Base Revenue", f"₹{df_revenue['base_revenue'].sum():,.2f}")
                with col2:
                    st.metric("Total Actual Revenue", f"₹{df_revenue['actual_revenue'].sum():,.2f}")
                with col3:
                    loss = df_revenue['revenue_loss'].sum()
                    base_total = df_revenue['base_revenue'].sum()
                    if base_total > 0:
                        loss_percent = (loss / base_total * 100)
                        st.metric("Revenue Lost to Discounts", f"₹{loss:,.2f}", delta=f"-{loss_percent:.1f}%")
                    else:
                        st.metric("Revenue Lost to Discounts", f"₹{loss:,.2f}")
        
    else:
        st.info("Need at least 10 transactions to generate AI analytics.")

# -------------------- TRANSACTIONS TAB --------------------
with tabs[3]:
    st.markdown("### Transaction History")
    
    if not df_transactions.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_product = st.selectbox("Filter by Product", ["All"] + list(df_transactions['product_name'].unique()))
        with col2:
            date_range = st.selectbox("Date Range", ["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days"])
        with col3:
            show_rows = st.slider("Rows to display", 10, 100, 25)
        
        filtered_df = df_transactions.copy()
        filtered_df['date'] = pd.to_datetime(filtered_df['date'], errors='coerce')
        
        if filter_product != "All":
            filtered_df = filtered_df[filtered_df['product_name'] == filter_product]
        
        if date_range != "All Time":
            days_map = {"Last 7 Days": 7, "Last 30 Days": 30, "Last 90 Days": 90}
            cutoff_date = pd.to_datetime(selected_date) - timedelta(days=days_map[date_range])
            filtered_df = filtered_df[filtered_df['date'] >= cutoff_date]
        
        # Add revenue column
        if 'adjusted_price' in filtered_df.columns:
            filtered_df['revenue'] = filtered_df['adjusted_price'] * filtered_df['stock_sold']
        
        st.dataframe(filtered_df.head(show_rows), use_container_width=True, height=400)
        st.caption(f"Showing {min(show_rows, len(filtered_df))} of {len(filtered_df)} transactions")
    else:
        st.info("No transactions recorded yet")

# -------------------- INVENTORY TAB --------------------
with tabs[4]:
    st.markdown("### Current Inventory Status")
    
    if not df_inventory.empty:
        df_display = df_inventory.copy()
        
        def stock_status(stock):
            if stock == 0:
                return "❌ Out of Stock"
            elif stock < 10:
                return "⚠️ Low Stock"
            else:
                return "✅ In Stock"
        
        df_display['Stock Status'] = df_display['stock_left'].apply(stock_status)
        df_display['Expiry Status'] = df_display['days_to_expiry'].apply(
            lambda x: get_expiry_status(x)[1] + " " + get_expiry_status(x)[0]
        )
        
        df_display['expiry_date'] = df_display['expiry_date'].dt.strftime('%Y-%m-%d')
        
        df_display = df_display[[
            'product_id', 'product_name', 'stock_left', 'Stock Status', 
            'expiry_date', 'days_to_expiry', 'Expiry Status',
            'base_price', 'adjusted_price', 'discount_percent', 'stock_sold_total'
        ]]
        
        df_display.columns = [
            'ID', 'Product', 'Stock Left', 'Stock Status', 
            'Expiry Date', 'Days to Expiry', 'Expiry Status',
            'Base Price (₹)', 'Current Price (₹)', 'Discount %', 'Total Sold'
        ]
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        col1, col2 = st.columns(2)
        
        with col1:
            low_stock = df_inventory[df_inventory['stock_left'] < 10]
            if not low_stock.empty:
                st.warning(f"**{len(low_stock)} product(s) have low stock!**")
                with st.expander("View Low Stock Products"):
                    st.dataframe(
                        low_stock[['product_name', 'stock_left']].rename(
                            columns={'product_name': 'Product', 'stock_left': 'Stock'}
                        ), 
                        use_container_width=True
                    )
        
        with col2:
            expiring_soon = df_inventory[df_inventory['days_to_expiry'] <= 10]
            if not expiring_soon.empty:
                st.warning(f"**{len(expiring_soon)} product(s) expiring within 10 days!**")
                with st.expander("View Expiring Products"):
                    st.dataframe(
                        expiring_soon[['product_name', 'days_to_expiry', 'discount_percent']].rename(
                            columns={
                                'product_name': 'Product', 
                                'days_to_expiry': 'Days Left',
                                'discount_percent': 'Discount %'
                            }
                        ), 
                        use_container_width=True
                    )
    else:
        st.info("Inventory is empty. Add products using the sidebar!")

conn.close()