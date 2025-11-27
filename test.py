from db_handler import WeaviateHandler

handler = WeaviateHandler({})

if handler.is_connected():
    schema_info = handler.get_schema_info(print_output=True)