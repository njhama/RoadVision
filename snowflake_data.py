import snowflake.connector
from snow_config import conn_params  

def load_images_to_snowflake(stage_name, table_name):

    conn = snowflake.connector.connect(**conn_params)
    cur = conn.cursor()

    copy_command = f"COPY INTO {table_name} FROM @{stage_name} FILE_FORMAT = (TYPE = 'CSV' FIELD_OPTIONALLY_ENCLOSED_BY = '\"');"
    cur.execute(copy_command)

   
    cur.close()
    conn.close()


load_images_to_snowflake('your_cloud_stage', 'image_data.images')
