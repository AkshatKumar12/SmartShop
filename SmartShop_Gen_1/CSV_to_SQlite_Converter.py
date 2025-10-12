
import sqlite3
import os
import pandas as pd

path = os.path.join("C:/Users/AKSHAT/Desktop/MONU_KI_DUKAN/smartshop/smartshop/Database", "smartshop_dataset.csv")
df = pd.read_csv(path)


conn = sqlite3.connect("my_database.db")

df.to_sql("my_table", conn, if_exists="append", index=False)

conn.close()
