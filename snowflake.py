import snowflake.connector
from snow_config import conn_params  # Importing the parameters

# Connect to Snowflake
conn = snowflake.connector.connect(**conn_params)
cur = conn.cursor()

# Your script for loading images goes here...

# Close connection
cur.close()
conn.close()
