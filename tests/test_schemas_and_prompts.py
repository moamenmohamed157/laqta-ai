import pytest
from pydantic import ValidationError

from laqta_ai.prompts import build_analysis_messages
from laqta_ai.schemas import ThumbnailAnalysis


def test_prompt_includes_user_context_and_language():
    system, user = build_analysis_messages(
        title="I built a tiny house",
        category="Vlog / Lifestyle",
        desired_style="Bright cinematic",
        language="Arabic",
    )
    assert "Arabic" in system
    assert "I built a tiny house" in user
    assert "Bright cinematic" in user


def test_schema_rejects_invalid_dimension_score():
    sample = {
        "overall_summary": "A sufficiently detailed summary of the visual design in this thumbnail.",
        "visual_analysis": {
            key: {"score": 11, "rationale": "A sufficiently long rationale.", "evidence": "A visible detail."}
            for key in (
                "visual_hierarchy", "subject_visibility", "composition_balance", "color_harmony",
                "text_readability", "background_clutter", "visual_focus", "title_relevance",
            )
        },
        "strengths": [
            {"element": f"Element {i}", "observation": "A sufficiently detailed visible observation.", "why_it_matters": "It supports a clear focal point."}
            for i in range(3)
        ],
        "improvement_opportunities": [
            {"element": f"Issue {i}", "observation": "A sufficiently detailed visible observation.", "why_it_matters": "It can improve thumbnail clarity."}
            for i in range(3)
        ],
        "prioritized_recommendations": [
            {"priority": "High", "issue": "Issue", "action": "A sufficiently detailed action to improve this element.", "expected_design_effect": "A clearer and more readable design."}
        ],
        "image_editing_prompt": "A sufficiently long and actionable image editing prompt for the image.",
    }
    with pytest.raises(ValidationError):
        ThumbnailAnalysis.model_validate(sample)
