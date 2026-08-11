"""Data contracts. Everything downstream produces or consumes these shapes."""

from typing import Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    source_type: Literal["youtube", "article"]
    url: str
    title: str
    text: str  # normalized full text
    anchors: list[str] = []  # timestamps or section markers, optional in v1


class Exercise(BaseModel):
    question: str
    options: list[str] = Field(min_length=3, max_length=4)
    answer_index: int  # index into options
    why: str  # one-line explanation of the answer


class Lesson(BaseModel):
    order: int
    title: str
    explanation: str  # the ~3-min read
    mermaid: str  # a mermaid diagram spec, as text
    exercise: Exercise
    citation: str  # where in the source this came from


class Path(BaseModel):
    document_title: str
    source_url: str
    lessons: list[Lesson] = Field(min_length=3, max_length=6)
