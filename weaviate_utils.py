import weaviate

weaviate_client = weaviate.connect_to_local(
    host="localhost",
    port=9000,
    grpc_port=50055
)

print(weaviate_client.is_ready())