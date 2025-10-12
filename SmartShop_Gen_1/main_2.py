import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import altair as alt
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
</style>
""", unsafe_allow_html=True)



BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "my_database.db")  

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()


st.sidebar.markdown("## Inventory Operations")

st.sidebar.markdown("---")
selected_date = st.sidebar.date_input(
    "Transaction Date", 
    value=datetime.now(),
    help="Select the date for this transaction"
)
transaction_date = selected_date.strftime("%Y-%m-%d")
transaction_day = selected_date.strftime("%A")

st.sidebar.info(f"{transaction_date}\n\n{transaction_day}")

st.sidebar.markdown("---")

operation = st.sidebar.radio(
    "Operation Type",
    ["Sell", "Buy"],
    help="Choose whether to sell or replenish stock."
)

with st.sidebar.expander("Product Details", expanded=True):
    product_id = st.number_input("Product ID", min_value=1, step=1)
    quantity = st.number_input("Quantity", min_value=0, step=1)

action_btn = st.sidebar.button("Confirm Transaction", use_container_width=True, type="primary")

# -------------------- OPERATIONS --------------------
if action_btn:
    if quantity == 0:
        st.sidebar.error("Quantity must be greater than 0!")
    else:
        cursor.execute("SELECT product_name, stock_sold_total, stock_left FROM inventory WHERE product_id=?", (product_id,))
        inv_row = cursor.fetchone()
        
        if operation == "Sell":
            if not inv_row:
                st.sidebar.warning("Cannot sell: Product ID not found in inventory. Add stock first!")
            else:
                product_name = inv_row[0]
                current_stock = inv_row[2]
                
                if quantity > current_stock:
                    st.sidebar.warning(f"Cannot sell {quantity} units. Only {current_stock} available.")
                else:
                    stock_left = current_stock - quantity
                    stock_sold_total = inv_row[1] + quantity

                    cursor.execute("""
                        INSERT INTO my_table (product_id, product_name, date, day, stock_sold, stock_left)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (product_id, product_name, transaction_date, transaction_day, quantity, stock_left))

                    cursor.execute("""
                        UPDATE inventory
                        SET stock_sold_total=?, stock_left=?
                        WHERE product_id=?
                    """, (stock_sold_total, stock_left, product_id))

                    st.sidebar.success(f"Sold {quantity} units of **{product_name}**\n\nStock remaining: **{stock_left}**")
                    conn.commit()
                    st.rerun()

        elif operation == "Buy":
            if not inv_row:
                st.sidebar.error("Product ID not found! Please add this product to inventory first with a product name.")
                st.sidebar.info("Tip: Use the database to add new products with both ID and name.")
            else:
                product_name = inv_row[0]
                stock_left = inv_row[2] + quantity
                stock_sold_total = inv_row[1]

                cursor.execute("""
                    INSERT INTO my_table (product_id, product_name, date, day, stock_sold, stock_left)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (product_id, product_name, transaction_date, transaction_day, 0, stock_left))

                cursor.execute("""
                    UPDATE inventory
                    SET stock_left=?
                    WHERE product_id=?
                """, (stock_left, product_id))

                st.sidebar.success(f"Added {quantity} units of **{product_name}**\n\nTotal stock: **{stock_left}**")
                conn.commit()
                st.rerun()

# -------------------- MAIN DASHBOARD --------------------
st.markdown('<h1 class="main-header">SmartShop Inventory Dashboard</h1>', unsafe_allow_html=True)

df_transactions = pd.read_sql("SELECT * FROM my_table ORDER BY date DESC, ROWID DESC", conn)
df_inventory = pd.read_sql("SELECT * FROM inventory ORDER BY product_id", conn)

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
        if not df_transactions.empty:
            df_trans_temp = df_transactions.copy()
            df_trans_temp['date'] = pd.to_datetime(df_trans_temp['date'], errors='coerce')
            today = selected_date
            today_sales = df_trans_temp[df_trans_temp['date'] == pd.to_datetime(today)]['stock_sold'].sum()
            st.metric(label="Today's Sales", value=f"{int(today_sales):,}")

st.markdown("---")

tabs = st.tabs(["Overview", "AI Analytics", "Transactions", "Inventory"])

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
        
        st.markdown("### Sales Trend (Last 30 Days)")
        df_temp = df_transactions.copy()
        df_temp['date'] = pd.to_datetime(df_temp['date'], errors='coerce')
        
        cutoff_date = pd.to_datetime(selected_date) - timedelta(days=30)
        last_30_days = df_temp[df_temp['date'] >= cutoff_date]
        
        if not last_30_days.empty:
            pivot_data = last_30_days.pivot_table(
                index='date', columns='product_name', values='stock_sold', aggfunc='sum'
            ).fillna(0)
            st.line_chart(pivot_data, height=400)
            st.caption(f"Showing sales from {cutoff_date.date()} to {selected_date}")
        else:
            st.info(f"No sales data in the last 30 days from {selected_date}")
    else:
        st.info("No data available. Start by adding products to your inventory!")

# -------------------- AI ANALYTICS TAB --------------------
with tabs[1]:
    if not df_transactions.empty and len(df_transactions) >= 10:
        df = df_transactions.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['product_id_code'] = df['product_id'].astype('category').cat.codes

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Sales Forecast (Next Month)")
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
            st.markdown("### Sales by Day of Week")
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_sales = df.groupby('day_of_week')['stock_sold'].sum()
            dow_sales = dow_sales.reindex(dow_order).fillna(0)
            st.bar_chart(dow_sales, height=350)
        
        st.markdown("---")
        
        st.markdown("### Sales Heatmap: Product × Day")
        heatmap_data = df.groupby(['day_of_week','product_name'])['stock_sold'].sum().reset_index()
        
        if not heatmap_data.empty:
            chart = alt.Chart(heatmap_data).mark_rect().encode(
                x=alt.X('day_of_week:N', title='Day of Week', sort=dow_order),
                y=alt.Y('product_name:N', title='Product'),
                color=alt.Color('stock_sold:Q', scale=alt.Scale(scheme='viridis'), title='Units Sold'),
                tooltip=['product_name', 'day_of_week', 'stock_sold']
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)
        
    else:
        st.info("Need at least 10 transactions to generate AI analytics. Keep selling!")

# -------------------- TRANSACTIONS TAB --------------------
with tabs[2]:
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
        
        st.dataframe(filtered_df.head(show_rows), use_container_width=True, height=400)
        st.caption(f"Showing {min(show_rows, len(filtered_df))} of {len(filtered_df)} transactions")
    else:
        st.info("No transactions recorded yet")

# -------------------- INVENTORY TAB --------------------
with tabs[3]:
    st.markdown("### Current Inventory Status")
    
    if not df_inventory.empty:
        df_display = df_inventory.copy()
        
        def stock_status(stock):
            if stock == 0:
                return "Out of Stock"
            elif stock < 10:
                return "Low Stock"
            else:
                return "In Stock"
        
        df_display['Status'] = df_display['stock_left'].apply(stock_status)
        df_display = df_display[['product_id', 'product_name', 'stock_left', 'stock_sold_total', 'Status']]
        df_display.columns = ['ID', 'Product', 'Stock Left', 'Total Sold', 'Status']
        
        st.dataframe(df_display, use_container_width=True, height=400)
        
        low_stock = df_inventory[df_inventory['stock_left'] < 10]
        if not low_stock.empty:
            st.warning(f"**{len(low_stock)} product(s) have low stock!**")
            with st.expander("View Low Stock Products"):
                st.dataframe(low_stock[['product_name', 'stock_left']], use_container_width=True)
    else:
        st.info("Inventory is empty. Add products using the sidebar!")

conn.close()