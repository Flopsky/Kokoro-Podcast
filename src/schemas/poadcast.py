# src/schemas/poadcast.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import re

class PodcastParagraph(BaseModel):
    """
    Represents a single paragraph of text for the podcast script.
    Ensures text is plain and ready for TTS.
    """
    text: str = Field(
        ...,
        description="A single paragraph of text, without markdown or bullet points, ready for TTS.",
    )

    @field_validator('text')
    @classmethod
    def text_must_be_valid(cls, v: str) -> str:
        if not v or v.isspace():
            raise ValueError('Paragraph text must not be empty or whitespace.')
        # Check for common markdown/list patterns: **, __, _, italics, * list, - list
        # This regex looks for **, __, _ not preceded by word char, and * or - followed by space at start of line/after space
        # Corrected regex:
        if re.search(r'(\*\*|__|(?<!\w)_|(?:^|\s)[*-]\s)', v):
             # Removed the print statement here
             raise ValueError('Paragraph text must not contain markdown symbols (**, __, _, italics) or list markers (* , - ).')
        # Ensure no leading/trailing whitespace which might affect TTS
        v = v.strip()
        if not v: # Check again after stripping
             raise ValueError('Paragraph text must not be empty after stripping whitespace.')
        return v

class PodcastSegment(BaseModel):
    """
    Represents a logical segment or section within the podcast.
    Contains an ordered list of paragraphs.
    """
    segment_title: Optional[str] = Field(
        None,
        description="An optional title for this segment, something meaningfull.",
    )
    paragraphs: List[PodcastParagraph] = Field(
        ...,
        description="An ordered list of paragraphs that make up this segment.",
        min_length=1 # A segment should have at least one paragraph
    )

class Poadcast(BaseModel):
    """
    Defines the structure for generating a podcast script from a source text.
    The output script is organized into segments and paragraphs,
    designed to be exhaustive, non-repetitive, and free of markdown,
    capable of handling extensive source material.
    """
    title: str = Field(
        ...,
        description="The overall title of the podcast episode.",
        examples=["Exploring the Deep Sea"]
    )
    summary: Optional[str] = Field(
        None,
        description="An optional brief summary of the podcast episode's content.",
        examples=["A deep dive into the mariana trench and the creatures that inhabit it."]
    )
    script_segments: List[PodcastSegment] = Field(
        ...,
        description="The main podcast script, structured as an ordered list of segments. Should comprehensively cover the source_text, potentially resulting in many segments and paragraphs.",
        min_length=1 # A podcast should have at least one segment
    )

    @field_validator('title')
    @classmethod
    def title_must_not_be_empty(cls, v: str) -> str:
        if not v or v.isspace():
            raise ValueError('Podcast title must not be empty or whitespace.')
        return v
