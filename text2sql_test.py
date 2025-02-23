from llama_index.core import download_loader

from llama_index.readers.database import DatabaseReader

DB_SCHEME = "MySQL"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_USER = "root"
DB_PASS = "123456"
DB_NAME = "test"

reader = DatabaseReader(
    scheme=DB_SCHEME,
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASS,
    dbname=DB_NAME,
)

query = "SELECT * FROM users"
documents = reader.load_data(query=query)
print(documents)