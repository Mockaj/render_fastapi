from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional

class SemanticSearchLog(BaseModel):
    id: Optional[int] = None
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    query: str
    law_name: str
    paragraph: int
    law_year: int
    law_id: int
    score: float

    class Config:
        from_attributes = True 