import weaviate
from weaviate.classes.query import MetadataQuery
from schemas import TextChunk
import json

class WeaviateHandler:
    def __init__(self, config: dict):
        self.url = config.get('host', 'localhost')
        self.http_port = config.get('http_port', '9000')
        self.grpc_port = config.get('grpc_port', '50055')
        self.client = weaviate.connect_to_local(
            host=self.url,
            port=int(self.http_port),
            grpc_port=int(self.grpc_port)
        )
        
    def is_connected(self) -> bool:
        return self.client.is_connected()
    
    def close(self):
        self.client.close()
        
    def get_schema_info(self, print_output: bool = True) -> dict:
        """
        Stáhne informace o všech kolekcích (schématech) v DB.
        
        Args:
            print_output (bool): Pokud je True, vytiskne hezky formátovaný JSON do konzole.
            
        Returns:
            dict: Slovník obsahující konfigurace všech kolekcí.
        """
        # Získáme všechny kolekce
        collections = self.client.collections.list_all()
        
        schema_summary = {}

       # Iterujeme přes názvy kolekcí
        for name in collections.keys():
            collection_obj = self.client.collections.get(name)
            
            config = collection_obj.config.get()
            
            schema_summary[name] = {
                "properties": [p.name for p in config.properties],
                # Ošetření pro případ, že vectorizer config není nastaven
                "vectorizer": config.vectorizer_config.vectorizer.value if config.vectorizer_config else "None",
                "full_config": config.to_dict()
            }

        if print_output:
            print(json.dumps(schema_summary, indent=2, default=str))

        return schema_summary