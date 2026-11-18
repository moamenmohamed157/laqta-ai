"""Presentation helpers for validated AI recommendations."""

from __future__ import annotations

from .schemas import ThumbnailAnalysis


DIMENSION_LABELS = {
    "visual_hierarchy": "Visual hierarchy",
    "subject_visibility": "Subject visibility",
    "composition_balance": "Composition & balance",
    "color_harmony": "Color harmony",
    "text_readability": "Text readability",
    "background_clutter": "Background clutter",
    "visual_focus": "Visual focus",
    "title_relevance": "Title relevance",
}


def ordered_recommendations(analysis: ThumbnailAnalysis):
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(analysis.prioritized_recommendations, key=lambda item: order[item.priority])


def calculate_experimental_score(analysis: ThumbnailAnalysis) -> int:
    """Unweighted editorial heuristic for navigation only; never a CTR estimate."""
    dimensions = list(analysis.visual_analysis.model_dump().values())
    return round(sum(item["score"] for item in dimensions) / len(dimensions))
