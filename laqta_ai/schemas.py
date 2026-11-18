"""Pydantic contracts for untrusted multimodal model output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class VisualDimension(StrictModel):
    score: int = Field(ge=1, le=10, description="Subjective visual assessment, not CTR prediction")
    rationale: str = Field(min_length=8, max_length=600)
    evidence: str = Field(min_length=3, max_length=350, description="A visible detail or explicit uncertainty")


class DesignPoint(StrictModel):
    element: str = Field(min_length=2, max_length=100)
    observation: str = Field(min_length=8, max_length=500)
    why_it_matters: str = Field(min_length=8, max_length=350)


class PrioritizedRecommendation(StrictModel):
    priority: Literal["High", "Medium", "Low"]
    issue: str = Field(min_length=5, max_length=200)
    action: str = Field(min_length=8, max_length=500)
    expected_design_effect: str = Field(min_length=8, max_length=300)


class VisualAnalysis(StrictModel):
    visual_hierarchy: VisualDimension
    subject_visibility: VisualDimension
    composition_balance: VisualDimension
    color_harmony: VisualDimension
    text_readability: VisualDimension
    background_clutter: VisualDimension
    visual_focus: VisualDimension
    title_relevance: VisualDimension


class ThumbnailAnalysis(StrictModel):
    overall_summary: str = Field(min_length=20, max_length=1200)
    visual_analysis: VisualAnalysis
    strengths: list[DesignPoint] = Field(min_length=3, max_length=3)
    improvement_opportunities: list[DesignPoint] = Field(min_length=3, max_length=3)
    prioritized_recommendations: list[PrioritizedRecommendation] = Field(min_length=1, max_length=7)
    image_editing_prompt: str = Field(min_length=40, max_length=3000)

    @field_validator("strengths", "improvement_opportunities")
    @classmethod
    def require_distinct_elements(cls, values: list[DesignPoint]) -> list[DesignPoint]:
        labels = [item.element.casefold() for item in values]
        if len(set(labels)) != len(labels):
            raise ValueError("Each item must refer to a distinct visible design element.")
        return values
