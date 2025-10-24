import pymysql, sys
try:
    conn = pymysql.connect(host="localhost", port=3306, user="root", password="1234", autocommit=True)
    with conn.cursor() as cur:
        cur.execute("CREATE DATABASE IF NOT EXISTS veloxquant_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
    print("✅ Base `veloxquant_db` créée / déjà présente.")
except Exception as e:
    print("❌ ERREUR:", e); sys.exit(1)
