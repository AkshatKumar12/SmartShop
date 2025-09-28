import sqlite3

DB_PATH = r"C:\Users\AKSHAT\Desktop\MONU_KI_DUKAN\smartshop\smartshop\Database\my_database.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# -------------------- Aggregate my_table --------------------
cursor.execute("""
SELECT product_id, product_name, SUM(stock_sold) as stock_sold_total, MAX(date) as last_date
FROM my_table
GROUP BY product_id, product_name
""")
rows = cursor.fetchall()

# -------------------- Update inventory table --------------------
for product_id, product_name, stock_sold_total, last_date in rows:
    # Get latest stock_left from the most recent transaction for this product
    cursor.execute("""
        SELECT stock_left 
        FROM my_table
        WHERE product_id=? AND date=?
        ORDER BY ROWID DESC
        LIMIT 1
    """, (product_id, last_date))
    latest_stock_left = cursor.fetchone()[0]

    # Check if product exists in inventory
    cursor.execute("SELECT * FROM inventory WHERE product_id=?", (product_id,))
    existing = cursor.fetchone()
    if existing:
        cursor.execute("""
            UPDATE inventory
            SET product_name=?, stock_sold_total=?, stock_left=?
            WHERE product_id=?
        """, (product_name, stock_sold_total, latest_stock_left, product_id))
    else:
        cursor.execute("""
            INSERT INTO inventory (product_id, product_name, stock_sold_total, stock_left)
            VALUES (?, ?, ?, ?)
        """, (product_id, product_name, stock_sold_total, latest_stock_left))

conn.commit()
conn.close()

print("Inventory table updated from my_table!")
