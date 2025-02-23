from pydantic import BaseModel

class ErrorResponse(BaseModel):
    detail: str

class QueryResponse(BaseModel):
    answer: str
    question: str

class IndexResponse(BaseModel):
    message: str
    filename: str

class StatusResponse(BaseModel):
    status: dict 