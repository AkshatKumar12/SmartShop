# import streamlit as st
# import sqlite3
# import pandas as pd
# from datetime import datetime

# DB_PATH = r"C:\Users\AKSHAT\Desktop\MONU_KI_DUKAN\smartshop\smartshop\Database\my_database.db"

# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cursor = conn.cursor()

# # -------------------- SIDEBAR --------------------
# st.sidebar.header("Inventory Operations")
# operation = st.sidebar.selectbox("Choose Operation", ["Sell", "Buy / Replenish"])
# product_id = st.sidebar.number_input("Product ID", min_value=1)
# product_name = st.sidebar.text_input("Product Name")
# quantity = st.sidebar.number_input("Quantity", min_value=0)
# action_btn = st.sidebar.button("Confirm")

# # System date
# now = datetime.now()
# date = now.strftime("%d-%m-%Y")
# day = now.strftime("%A")

# if action_btn:
#     # Get current inventory if exists
#     cursor.execute("SELECT stock_sold_total, stock_left FROM inventory WHERE product_id=?", (product_id,))
#     inv_row = cursor.fetchone()
    
#     if operation == "Sell":
#         # Get latest stock_left from transactions
#         cursor.execute("SELECT stock_left FROM my_table WHERE product_id=? ORDER BY ROWID DESC LIMIT 1", (product_id,))
#         trans_row = cursor.fetchone()
#         previous_stock_left = trans_row[0] if trans_row else (inv_row[1] if inv_row else None)

#         if previous_stock_left is None:
#             st.warning("New product! Please enter initial stock first.")
#             total_stock = st.sidebar.number_input("Initial Stock for New Product", min_value=quantity)
#             previous_stock_left = total_stock

#         stock_left = max(previous_stock_left - quantity, 0)

#         # Insert transaction
#         cursor.execute("""
#             INSERT INTO my_table (product_id, product_name, date, day, stock_sold, stock_left)
#             VALUES (?, ?, ?, ?, ?, ?)
#         """, (product_id, product_name, date, day, quantity, stock_left))

#         # Update inventory
#         stock_sold_total = (inv_row[0] if inv_row else 0) + quantity
#         cursor.execute("""
#             INSERT INTO inventory (product_id, product_name, stock_sold_total, stock_left)
#             VALUES (?, ?, ?, ?)
#             ON CONFLICT(product_id) DO UPDATE SET
#             product_name=excluded.product_name,
#             stock_sold_total=excluded.stock_sold_total,
#             stock_left=excluded.stock_left
#         """, (product_id, product_name, stock_sold_total, stock_left))

#         st.success(f"Sold {quantity} units of {product_name}. Stock left: {stock_left}")

#     elif operation == "Buy / Replenish":
#         if inv_row:
#             stock_left = inv_row[1] + quantity
#             stock_sold_total = inv_row[0]
#         else:
#             stock_left = quantity
#             stock_sold_total = 0

#         cursor.execute("""
#             INSERT INTO inventory (product_id, product_name, stock_sold_total, stock_left)
#             VALUES (?, ?, ?, ?)
#             ON CONFLICT(product_id) DO UPDATE SET
#             product_name=excluded.product_name,
#             stock_sold_total=excluded.stock_sold_total,
#             stock_left=excluded.stock_left
#         """, (product_id, product_name, stock_sold_total, stock_left))

#         st.success(f"Replenished {quantity} units of {product_name}. Current stock: {stock_left}")

#     conn.commit()

# # -------------------- DASHBOARD --------------------
# st.title("🛒 SmartShop Dashboard")
# tabs = st.tabs(["Transactions Table", "Inventory Table"])

# with tabs[0]:
#     st.subheader("Transaction History")
#     df_transactions = pd.read_sql("SELECT * FROM my_table ORDER BY ROWID DESC", conn)
#     st.dataframe(df_transactions)
#     if not df_transactions.empty:
#         st.bar_chart(df_transactions.set_index("product_name")["stock_sold"])

# with tabs[1]:
#     st.subheader("Current Inventory")
#     df_inventory = pd.read_sql("SELECT * FROM inventory ORDER BY product_id", conn)
#     st.dataframe(df_inventory)
#     if not df_inventory.empty:
#         st.bar_chart(df_inventory.set_index("product_name")["stock_left"])
#         st.line_chart(df_inventory.set_index("product_name")["stock_sold_total"])


########################################################################################################################################################################################################################################

# import streamlit as st
# import sqlite3
# import pandas as pd
# from datetime import datetime
# from sklearn.ensemble import RandomForestRegressor

# DB_PATH = r"C:\Users\AKSHAT\Desktop\MONU_KI_DUKAN\smartshop\smartshop\Database\my_database.db"

# conn = sqlite3.connect(DB_PATH, check_same_thread=False)
# cursor = conn.cursor()

# # -------------------- SIDEBAR --------------------
# st.sidebar.header("Inventory Operations")
# operation = st.sidebar.selectbox("Choose Operation", ["Sell", "Buy / Replenish"])
# product_id = st.sidebar.number_input("Product ID", min_value=1)
# product_name = st.sidebar.text_input("Product Name")
# quantity = st.sidebar.number_input("Quantity", min_value=0)
# action_btn = st.sidebar.button("Confirm")

# # System date
# now = datetime.now()
# date = now.strftime("%d-%m-%Y")
# day = now.strftime("%A")

# if action_btn:
#     # Get current inventory if exists
#     cursor.execute("SELECT stock_sold_total, stock_left FROM inventory WHERE product_id=?", (product_id,))
#     inv_row = cursor.fetchone()
    
#     if operation == "Sell":
#         if not inv_row:
#             st.warning("Cannot sell: Product not in inventory. Add stock first!")
#         else:
#             current_stock = inv_row[1]
#             if quantity > current_stock:
#                 st.warning(f"Cannot sell {quantity} units. Only {current_stock} units available.")
#             else:
#                 stock_left = current_stock - quantity
#                 stock_sold_total = inv_row[0] + quantity

#                 # Insert transaction
#                 cursor.execute("""
#                     INSERT INTO my_table (product_id, product_name, date, day, stock_sold, stock_left)
#                     VALUES (?, ?, ?, ?, ?, ?)
#                 """, (product_id, product_name, date, day, quantity, stock_left))

#                 # Update inventory
#                 cursor.execute("""
#                     UPDATE inventory
#                     SET stock_sold_total=?, stock_left=?, product_name=?
#                     WHERE product_id=?
#                 """, (stock_sold_total, stock_left, product_name, product_id))

#                 st.success(f"Sold {quantity} units of {product_name}. Stock left: {stock_left}")

#     elif operation == "Buy / Replenish":
#         if inv_row:
#             stock_left = inv_row[1] + quantity
#             stock_sold_total = inv_row[0]
#         else:
#             stock_left = quantity
#             stock_sold_total = 0

#         # Insert a transaction for buy (stock_sold = 0)
#         cursor.execute("""
#             INSERT INTO my_table (product_id, product_name, date, day, stock_sold, stock_left)
#             VALUES (?, ?, ?, ?, ?, ?)
#         """, (product_id, product_name, date, day, 0, stock_left))

#         # Update inventory
#         cursor.execute("""
#             INSERT INTO inventory (product_id, product_name, stock_sold_total, stock_left)
#             VALUES (?, ?, ?, ?)
#             ON CONFLICT(product_id) DO UPDATE SET
#                 product_name=excluded.product_name,
#                 stock_sold_total=excluded.stock_sold_total,
#                 stock_left=excluded.stock_left
#         """, (product_id, product_name, stock_sold_total, stock_left))

#         st.success(f"Replenished {quantity} units of {product_name}. Current stock: {stock_left}")

#     conn.commit()

# # -------------------- DASHBOARD --------------------
# st.title("🛒 SmartShop Dashboard")
# tabs = st.tabs(["Transactions Table", "Inventory Table", "AI Forecast"])

# # -------------------- Transactions Tab --------------------
# with tabs[0]:
#     st.subheader("Transaction History")
#     df_transactions = pd.read_sql("SELECT * FROM my_table ORDER BY ROWID DESC", conn)
#     st.dataframe(df_transactions)
#     if not df_transactions.empty:
#         st.bar_chart(df_transactions.set_index("product_name")["stock_sold"])

# # -------------------- Inventory Tab --------------------
# with tabs[1]:
#     st.subheader("Current Inventory")
#     df_inventory = pd.read_sql("SELECT * FROM inventory ORDER BY product_id", conn)
#     st.dataframe(df_inventory)
#     if not df_inventory.empty:
#         st.bar_chart(df_inventory.set_index("product_name")["stock_left"])
#         st.line_chart(df_inventory.set_index("product_name")["stock_sold_total"])

# # -------------------- AI / Forecast Tab --------------------
# with tabs[2]:
#     st.subheader("ML Forecast: Predict Next Sale")
#     df = pd.read_sql("SELECT * FROM my_table", conn)

#     if not df.empty:
#         # Feature engineering
#         # df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
#         # Auto-detect format
#         df['date'] = pd.to_datetime(df['date'], errors='coerce')
#         # Handle day-first dates and ISO dates
#         df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')


#         df['day_of_week'] = df['date'].dt.dayofweek
#         df['month'] = df['date'].dt.month
#         df['product_id_code'] = df['product_id'].astype('category').cat.codes

#         X = df[['product_id_code', 'day_of_week', 'month', 'stock_left']]
#         y = df['stock_sold']

#         model = RandomForestRegressor()
#         model.fit(X, y)

#         st.markdown("### Predict Sales for a Product")
#         product_id_input = st.number_input("Product ID", min_value=1, value=1, key="ai_product")
#         day_input = st.selectbox("Day of Week (0=Mon)", list(range(7)), key="ai_day")
#         month_input = st.selectbox("Month", list(range(1,13)), key="ai_month")
#         # Get current stock_left
#         cursor.execute("SELECT stock_left FROM inventory WHERE product_id=?", (product_id_input,))
#         stock_left_input = cursor.fetchone()[0] if cursor.fetchone() else 0

#         pred = model.predict([[product_id_input, day_input, month_input, stock_left_input]])
#         st.write(f"Predicted sales: {int(pred[0])}")
#     else:
#         st.info("No transaction data yet to make predictions.")

########################################################################################################################################################################################################################################

import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import altair as alt

DB_PATH = r"C:\Users\AKSHAT\Desktop\DBMS_PBL\SmartShop\SmartShop_Gen_1\my_database.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()

# -------------------- SIDEBAR --------------------
st.sidebar.header("Inventory Operations")
operation = st.sidebar.selectbox("Choose Operation", ["Sell", "Buy / Replenish"])
product_id = st.sidebar.number_input("Product ID", min_value=1)
product_name = st.sidebar.text_input("Product Name")
quantity = st.sidebar.number_input("Quantity", min_value=0)
action_btn = st.sidebar.button("Confirm")

# System date
now = datetime.now()
date = now.strftime("%Y-%m-%d")  # use ISO format for consistency
day = now.strftime("%A")

if action_btn:
    # Get current inventory if exists
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

                # Insert transaction
                cursor.execute("""
                    INSERT INTO my_table (product_id, product_name, date, day, stock_sold, stock_left)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (product_id, product_name, date, day, quantity, stock_left))

                # Update inventory
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

        # Insert a transaction for buy (stock_sold = 0)
        cursor.execute("""
            INSERT INTO my_table (product_id, product_name, date, day, stock_sold, stock_left)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (product_id, product_name, date, day, 0, stock_left))

        # Update inventory
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

# -------------------- DASHBOARD --------------------
st.title("🛒 SmartShop Dashboard")
tabs = st.tabs(["Transactions Table", "Inventory Table", "AI Forecast"])

# -------------------- Transactions Tab --------------------
with tabs[0]:
    st.subheader("Transaction History")
    df_transactions = pd.read_sql("SELECT * FROM my_table ORDER BY ROWID DESC", conn)
    st.dataframe(df_transactions)
    if not df_transactions.empty:
        st.bar_chart(df_transactions.pivot_table(index='date', columns='product_name', values='stock_sold', aggfunc='sum').fillna(0))

# -------------------- Inventory Tab --------------------
with tabs[1]:
    st.subheader("Current Inventory")
    df_inventory = pd.read_sql("SELECT * FROM inventory ORDER BY product_id", conn)
    st.dataframe(df_inventory)
    if not df_inventory.empty:
        st.bar_chart(df_inventory.set_index("product_name")['stock_left'])
        st.line_chart(df_inventory.set_index("product_name")['stock_sold_total'])

# -------------------- AI / Forecast Tab --------------------
with tabs[2]:
    st.subheader("AI & Forecast Dashboard")
    df = pd.read_sql("SELECT * FROM my_table", conn)

    if not df.empty:
        # Parse dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['day_of_week'] = df['date'].dt.day_name()
        df['month'] = df['date'].dt.month
        df['product_id_code'] = df['product_id'].astype('category').cat.codes

        # 1️⃣ Sales Over Time per Product
        st.markdown("### Sales Over Time per Product")
        line_chart_data = df.pivot_table(index='date', columns='product_name', values='stock_sold', aggfunc='sum').fillna(0)
        st.line_chart(line_chart_data)

        # 2️⃣ Total Stock Sold per Product
        st.markdown("### Total Stock Sold per Product")
        total_sold = df.groupby('product_name')['stock_sold'].sum()
        st.bar_chart(total_sold)

        # 3️⃣ Current Stock Left per Product
        st.markdown("### Current Stock Left per Product")
        if not df_inventory.empty:
            st.bar_chart(df_inventory.set_index("product_name")['stock_left'])

        # 4️⃣ Predicted Sales Next Month per Product
        st.markdown("### Predicted Sales Next Month per Product")
        X = df[['product_id_code', 'month']]
        y = df['stock_sold']
        model = RandomForestRegressor()
        model.fit(X, y)

        products = df['product_name'].unique()
        pred_df = pd.DataFrame()
        for prod in products:
            prod_code = df[df['product_name']==prod]['product_id_code'].iloc[0]
            pred = model.predict([[prod_code, datetime.now().month + 1]])  # next month
            pred_df = pd.concat([pred_df, pd.DataFrame({'Product':[prod], 'Predicted_Sales':[int(pred[0])]})])
        st.bar_chart(pred_df.set_index('Product')['Predicted_Sales'])

        # 5️⃣ Sales by Day of Week Heatmap
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
