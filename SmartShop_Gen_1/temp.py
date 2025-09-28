import sqlite3

conn = sqlite3.connect('my_database.db')
cursor = conn.cursor()

# Delete rows where salt_sale is 1000
cursor.execute("DELETE FROM my_table WHERE salt_sale = ?", (1025,))
conn.commit()

conn.close()
