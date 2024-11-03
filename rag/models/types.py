# models/types.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional


class Paragraf(BaseModel):
    cislo: str
    zneni: str


class Law(BaseModel):
    nazev: str
    id: str
    year: str
    category: Optional[str] = None
    date: Optional[str] = None
    staleURL: Optional[str] = None
    paragrafy: List[Paragraf] = Field(default_factory=list)


class QueryRequest(BaseModel):
    query: str

    @validator('query')
    def query_must_be_non_empty(cls, v):
        if not v.strip():
            raise ValueError('Query must be a non-empty string')
        return v


class RelevantDocument(BaseModel):
    law_nazev: str
    law_id: str
    law_year: str
    law_category: Optional[str]
    law_date: Optional[str]
    law_staleURL: Optional[str]
    paragraph_cislo: str
    paragraph_zneni: str


class QueryResponse(BaseModel):
    relevant_docs: List[RelevantDocument]


class SeedLawRequest(BaseModel):
    url: str

    @validator('url')
    def validate_url(cls, v):
        if not v.startswith('https://www.zakonyprolidi.cz'):
            raise ValueError('URL must be from zakonyprolidi.cz')
        return v
