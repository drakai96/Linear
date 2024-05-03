from qdrant_client import QdrantClient, models
from constant import DATA_INFO_COLLECTION
class QdrantQuery():
    
    def __init__(self):
        self.client = QdrantClient(url="http://localhost:6333")
        if not self.client.collection_exists("data_info"):
            self.client.create_collection(
                collection_name=DATA_INFO_COLLECTION)

    def add_data(self,id, payload:dict):
        self.client.upsert(collection_name=DATA_INFO_COLLECTION,points=[
        models.PointStruct(
            id=id,
            payload = payload)])

    def search_with_id(self,id):
        data = self.client.retrieve(
            collection_name="{DATA_INFO_COLLECTION}",
            ids=[id]
        )
        return data