from pydantic import BaseModel
from uuid import UUID

class DBRequest(BaseModel):
    user_collection_id: UUID | None = None