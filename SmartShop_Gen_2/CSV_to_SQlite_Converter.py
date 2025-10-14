import sqlite3
import pandas as pd

path = r"C:\Users\AKSHAT\Desktop\DBMS_PBL\SmartShop\SmartShop_Gen_2\dataset2.csv"
df = pd.read_csv(path)

conn = sqlite3.connect("my_database.db")
df.to_sql("my_table", conn, if_exists="append", index=False)
conn.close()
