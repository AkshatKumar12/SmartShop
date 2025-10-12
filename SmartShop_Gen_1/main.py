
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import altair as alt
import os

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "my_database.db")  
DB_PATH = os.path.join(BASE_DIR, "my_database.db") 

# sqlite
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

st.sidebar.header("Inventory Operations")
operation = st.sidebar.selectbox("Choose Operation", ["Sell", "Buy / Replenish"])
product_id = st.sidebar.number_input("Product ID", min_value=1)
product_name = st.sidebar.text_input("Product Name")
quantity = st.sidebar.number_input("Quantity", min_value=0)
action_btn = st.sidebar.button("Confirm")


now = datetime.now()
date = now.strftime("%Y-%m-%d") 
day = now.strftime("%A")

if action_btn:
    cursor.execute("SELECT stock_sold_total, stock_left FROM inventory WHERE product_id=?", (product_id,))
    inv_row = cursor.fetchone()
    
    if operation == "Sell":
        if not inv_row:
            st.warning("Cannot sell: Product not in inventory. Add stock first!")
        else:
            current_stock = inv_row[1]
            if quantity > current_stock:
                st.warning(f"Cannot sell {quantity} units. Only {current_stock} units available.")
            else:
                stock_left = current_stock - quantity
                stock_sold_total = inv_row[0] + quantity

                
                cursor.execute("""
                    INSERT INTO my_table (product_id, product_name, date, day, stock_sold, stock_left)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (product_id, product_name, date, day, quantity, stock_left))

               
                cursor.execute("""
                    UPDATE inventory
                    SET stock_sold_total=?, stock_left=?, product_name=?
                    WHERE product_id=?
                """, (stock_sold_total, stock_left, product_name, product_id))

                st.success(f"Sold {quantity} units of {product_name}. Stock left: {stock_left}")

    elif operation == "Buy / Replenish":
        if inv_row:
            stock_left = inv_row[1] + quantity
            stock_sold_total = inv_row[0]
        else:
            stock_left = quantity
            stock_sold_total = 0

        cursor.execute("""
            INSERT INTO my_table (product_id, product_name, date, day, stock_sold, stock_left)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (product_id, product_name, date, day, 0, stock_left))

        cursor.execute("""
            INSERT INTO inventory (product_id, product_name, stock_sold_total, stock_left)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(product_id) DO UPDATE SET
                product_name=excluded.product_name,
                stock_sold_total=excluded.stock_sold_total,
                stock_left=excluded.stock_left
        """, (product_id, product_name, stock_sold_total, stock_left))

        st.success(f"Replenished {quantity} units of {product_name}. Current stock: {stock_left}")

    conn.commit()

st.title("🛒 SmartShop Dashboard")
tabs = st.tabs(["Transactions Table", "Inventory Table", "AI Forecast"])

with tabs[0]:
    st.subheader("Transaction History")
    df_transactions = pd.read_sql("SELECT * FROM my_table ORDER BY ROWID DESC", conn)
    st.dataframe(df_transactions)
    if not df_transactions.empty:
        pivot_data = df_transactions.pivot_table(
            index='date', columns='product_name', values='stock_sold', aggfunc='sum'
        ).fillna(0)
        st.line_chart(pivot_data)

with tabs[1]:
    st.subheader("Current Inventory")
    df_inventory = pd.read_sql("SELECT * FROM inventory ORDER BY product_id", conn)
    st.dataframe(df_inventory)
    if not df_inventory.empty:
        st.bar_chart(df_inventory.set_index("product_name")['stock_left'])
        st.line_chart(df_inventory.set_index("product_name")['stock_sold_total'])

#ml
with tabs[2]:
    st.subheader("AI & Forecast Dashboard")
    df = pd.read_sql("SELECT * FROM my_table", conn)

    if not df.empty:
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month
        df['product_id_code'] = df['product_id'].astype('category').cat.codes

        
        st.markdown("### Sales Over Time per Product")
        line_chart_data = df.pivot_table(
            index='date', columns='product_name', values='stock_sold', aggfunc='sum'
        ).fillna(0)
        st.line_chart(line_chart_data)

        st.markdown("### Total Stock Sold per Product")
        total_sold = df.groupby('product_name')['stock_sold'].sum()
        st.bar_chart(total_sold)

        # 3️⃣ Current Stock Left
        # stock lft
        st.markdown("### Current Stock Left per Product")
        if not df_inventory.empty:
            st.bar_chart(df_inventory.set_index("product_name")['stock_left'])

        # 4️⃣ Predicted Sales Next Month
        # preediction
        st.markdown("### Predicted Sales Next Month per Product")
        X = df[['product_id_code', 'month']]
        y = df['stock_sold']
        model = RandomForestRegressor()
        model.fit(X, y)

        pred_df = pd.DataFrame()
        for prod in df['product_name'].unique():
            prod_code = df[df['product_name']==prod]['product_id_code'].iloc[0]
            next_month = datetime.now().month + 1
            pred = model.predict([[prod_code, next_month]])
            pred_df = pd.concat([pred_df, pd.DataFrame({'Product':[prod], 'Predicted_Sales':[int(pred[0])]})])
        st.bar_chart(pred_df.set_index('Product')['Predicted_Sales'])

        #heatmap
        st.markdown("### Sales by Day of Week")
        heatmap_data = df.groupby(['day_of_week','product_name'])['stock_sold'].sum().reset_index()
        chart = alt.Chart(heatmap_data).mark_rect().encode(
            x='day_of_week:N',
            y='product_name:N',
            color='stock_sold:Q'
        )
        st.altair_chart(chart, use_container_width=True)

    else:
        st.info("No transaction data yet to show charts or make predictions.")
