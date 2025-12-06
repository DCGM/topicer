from pydantic import BaseModel
from uuid import UUID

class DBRequest(BaseModel):
    collection_id: UUID | None = None