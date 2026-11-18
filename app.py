from __future__ import annotations

import hashlib
import html
import json
from dataclasses import asdict

import streamlit as st

from laqta_ai.analysis_service import analyze_locally, analyze_with_vision
from laqta_ai.config import Settings
from laqta_ai.image_analyzer import ImageValidationError
from laqta_ai.recommendations import DIMENSION_LABELS, calculate_experimental_score, ordered_recommendations
from laqta_ai.vision_client import VisionAPIError


st.set_page_config(
    page_title="Laqta AI · Thumbnail Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root { --mint: #5DB891; --mint-soft: rgba(93,184,145,.13); --ink: #0A1019; --panel: #111B27; --panel-2: #172331; --line: rgba(184,205,220,.12); --muted: #91A1B2; --white: #F3F7FA; }
.stApp { background: radial-gradient(ellipse at 10% -10%, rgba(49,112,92,.16), transparent 34%), linear-gradient(145deg,#0A1019 0%,#0D1520 52%,#0A111A 100%); color: var(--white); }
[data-testid="stHeader"] { background: rgba(10,16,25,.78); }
[data-testid="stSidebar"] { background: #0D1722; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.25rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { background: linear-gradient(145deg,rgba(21,33,47,.96),rgba(15,25,37,.96)); border:1px solid var(--line); border-radius:19px; padding:18px 20px; box-shadow:0 16px 50px rgba(0,0,0,.13); }
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 4rem; }
.eyebrow { color: var(--mint); font-size: .72rem; font-weight: 800; letter-spacing: .19em; text-transform: uppercase; margin-bottom: .45rem; }
.hero-title { font-size: clamp(2.2rem,4vw,3.6rem); font-weight: 780; line-height: 1.02; letter-spacing: -.055em; color: #F5F8FA; margin: 0; }
.hero-title span { color: var(--mint); }
.hero-copy { color: #A7B6C4; font-size: 1.02rem; margin-top: .75rem; max-width: 760px; }
.brand-lockup { display:flex; align-items:center; gap:10px; margin-bottom: 2rem; }
.brand-mark { width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:linear-gradient(145deg,#73D3A9,#358B69);color:#08130F;font-weight:900;font-size:22px;box-shadow:0 8px 25px rgba(93,184,145,.19); }
.brand-name { color:#F4F8FA;font-weight:800;letter-spacing:-.035em;font-size:1.12rem; }
.brand-sub { color:#8395A6;font-size:.69rem;letter-spacing:.15em;text-transform:uppercase;margin-top:1px; }
.panel { background: linear-gradient(145deg,rgba(21,33,47,.96),rgba(15,25,37,.96)); border:1px solid var(--line); border-radius:19px; padding:20px 22px; box-shadow:0 16px 50px rgba(0,0,0,.13); }
.panel-kicker { color:var(--mint);font-size:.68rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase;margin-bottom:6px; }
.panel-title { color:#EDF4F7;font-size:1.04rem;font-weight:750;letter-spacing:-.02em;margin-bottom:5px; }
.panel-copy { color:#8FA1B1;font-size:.85rem;line-height:1.5; }
.metric-card { background:linear-gradient(145deg,#172636,#111D29);border:1px solid var(--line);border-radius:15px;padding:16px 17px;min-height:112px; }
.metric-label { color:#93A5B5;font-size:.76rem;font-weight:650; }
.metric-value { color:#F2F7F9;font-size:1.65rem;font-weight:780;letter-spacing:-.04em;margin-top:7px; }
.metric-foot { color:#718597;font-size:.69rem;margin-top:2px; }
.section-heading { color:#F1F6F8;font-size:1.27rem;font-weight:760;letter-spacing:-.03em;margin:1.5rem 0 .2rem; }
.section-caption { color:#8FA1B1;font-size:.84rem;margin-bottom:1rem; }
.insight-card { background:#131F2B;border:1px solid var(--line);border-radius:14px;padding:15px 16px;margin-bottom:10px; }
.insight-title { color:#F0F5F7;font-size:.9rem;font-weight:720;margin-bottom:6px; }
.insight-body { color:#A7B5C1;font-size:.82rem;line-height:1.5; }
.priority { font-size:.65rem;font-weight:850;letter-spacing:.1em;text-transform:uppercase;border-radius:6px;padding:4px 7px;display:inline-block;margin-bottom:9px; }
.priority-high { color:#FFB4A8;background:rgba(234,110,94,.12); }
.priority-medium { color:#F2CC83;background:rgba(214,163,74,.12); }
.priority-low { color:#9AD8BD;background:rgba(93,184,145,.13); }
.color-chip { display:flex;align-items:center;gap:9px;margin:8px 0;color:#BDCAD3;font-size:.77rem; }
.swatch {width:26px;height:26px;border-radius:8px;border:1px solid rgba(255,255,255,.2);flex:0 0 auto;}
.small-note { color:#778B9D;font-size:.74rem;line-height:1.5; }
.score-pill { color:#8FE0B9;background:rgba(93,184,145,.12);border:1px solid rgba(93,184,145,.25);border-radius:99px;padding:5px 10px;font-weight:750;font-size:.78rem; }
.stButton > button { border-radius:11px;border:1px solid rgba(93,184,145,.42);background:linear-gradient(135deg,#6BC59B,#4AA77E);color:#08140F;font-weight:800;min-height:2.75rem; }
.stButton > button:hover { border-color:#8BE2B7;background:#80D3AA;color:#07130E; }
.stButton > button:disabled { opacity:.42; }
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div { background:#111E2A!important;border-color:rgba(184,205,220,.16)!important;color:#ECF3F6!important;border-radius:10px!important; }
.stTabs [data-baseweb="tab-list"] { gap:7px;border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color:#8FA1B1;padding:10px 15px; }
.stTabs [aria-selected="true"] { color:#88DAB2!important;border-bottom-color:#5DB891!important; }
[data-testid="stFileUploader"] section { background:#101D29;border:1px dashed rgba(93,184,145,.5);border-radius:16px; }
[data-testid="stFileUploader"] section small { color:#8194A4!important; }
hr { border-color:var(--line)!important; }
footer { visibility:hidden; }
@media (max-width: 760px) { .block-container{padding:1.1rem 1rem 3rem;} .panel{padding:16px;} }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def get_settings() -> Settings:
    try:
        return Settings.from_env(st.secrets)
    except Exception:
        return Settings.from_env()


def panel_header(kicker: str, title: str, description: str = "") -> None:
    st.markdown(
        f'<div class="panel-kicker">{html.escape(kicker)}</div>'
        f'<div class="panel-title">{html.escape(title)}</div>'
        f'<div class="panel-copy">{html.escape(description)}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, foot: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div><div class="metric-foot">{html.escape(foot)}</div></div>',
        unsafe_allow_html=True,
    )


def show_design_point(item, number: int) -> None:
    st.markdown(
        f'<div class="insight-card"><div class="insight-title">{number:02d} · {html.escape(item.element)}</div>'
        f'<div class="insight-body">{html.escape(item.observation)}<br><br>'
        f'<span style="color:#75CBA0;font-weight:700">Why it matters</span> · {html.escape(item.why_it_matters)}</div></div>',
        unsafe_allow_html=True,
    )


settings = get_settings()

with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><div class="brand-mark">✦</div><div><div class="brand-name">Laqta AI</div>'
        '<div class="brand-sub">Analyze · Improve · Create</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("#### Analysis workspace")
    video_title = st.text_input("Video title", placeholder="e.g. I Tried Living in the Desert for 24 Hours")
    category = st.selectbox(
        "Content category",
        ["Vlog / Lifestyle", "Education", "Tech / AI", "Gaming", "Business / Finance", "Travel", "Food", "Documentary", "Entertainment", "Other"],
    )
    visual_style = st.text_area(
        "Desired visual style",
        placeholder="e.g. Bright cinematic lighting, bold subject separation, natural colors…",
        height=92,
    )
    language = st.radio("Recommendation language", ["English", "Arabic"], horizontal=True)
    st.divider()
    st.markdown("#### Vision AI connection")
    if settings.vision_is_configured:
        st.success(f"Connected configuration · {settings.vision_model}", icon="✦")
        st.caption("Credentials are read from environment variables / Streamlit secrets and are never displayed.")
    else:
        st.warning("Vision AI is not configured. Local computer-vision analysis still works.", icon="⚙️")
        st.caption("Set LAQTA_VISION_API_KEY in your .env file. See README for provider setup.")
    with st.expander("Advanced provider settings"):
        st.caption("These values are configured outside the app.")
        st.code(
            "LAQTA_VISION_BASE_URL=https://api.openai.com/v1\n"
            "LAQTA_VISION_MODEL=gpt-4.1-mini\n"
            "LAQTA_VISION_TIMEOUT_SECONDS=90",
            language="bash",
        )
    st.markdown('<div class="small-note">Your upload is processed in memory for this session. Laqta AI does not write it to disk or retain it as a project file.</div>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">AI-powered thumbnail intelligence</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Make every frame <span>count.</span></h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">A practical thumbnail review studio combining measurable image signals with grounded multimodal creative direction. Analyze the pixels. Then improve the story.</div>',
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1.12, 0.88], gap="large")
with left:
    with st.container(border=True):
        panel_header("01 / Upload", "Start with your thumbnail", "JPG, PNG, or WEBP · Max file size and pixel limits are configurable")
        uploaded = st.file_uploader("Upload thumbnail", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if uploaded is None:
            st.markdown(
                '<div style="padding:22px 8px 8px;text-align:center;color:#7F93A4">'
                '<div style="font-size:2.25rem;color:#5DB891">↥</div><b style="color:#DCE7EC">Drop your thumbnail here</b>'
                '<div style="font-size:.77rem;margin-top:6px">Your image stays in session memory and is not saved to disk.</div></div>',
                unsafe_allow_html=True,
            )

local_analysis = None
raw_bytes = None
image_hash = None
if uploaded is not None:
    raw_bytes = uploaded.getvalue()
    image_hash = hashlib.sha256(raw_bytes).hexdigest()
    if st.session_state.get("image_hash") != image_hash:
        st.session_state["image_hash"] = image_hash
        st.session_state.pop("ai_result", None)
        st.session_state.pop("ai_context_hash", None)
    try:
        with st.spinner("Validating image and calculating local metrics…"):
            local_analysis = analyze_locally(raw_bytes, uploaded.name, settings)
    except ImageValidationError as exc:
        st.error(str(exc), icon="⚠️")
    except Exception as exc:
        st.error("Local image analysis failed safely. Try re-exporting the image as JPG or PNG.", icon="⚠️")

with right:
    with st.container(border=True):
        panel_header("02 / Preview", "Thumbnail preview", "Original proportions are preserved; the AI copy is resized only when needed")
        if local_analysis:
            st.image(local_analysis.processed_image.image, use_container_width=True)
            meta = local_analysis.processed_image.metadata
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;margin-top:10px">'
                f'<span class="small-note">{meta.width:,} × {meta.height:,} px</span>'
                f'<span class="small-note">{html.escape(meta.aspect_ratio_label)}</span>'
                f'<span class="small-note">{meta.file_size_bytes / 1024 / 1024:.2f} MB</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="aspect-ratio:16/9;border:1px dashed rgba(184,205,220,.2);border-radius:13px;display:grid;place-items:center;color:#6F8496">'
                '<div style="text-align:center"><div style="font-size:2rem">▧</div><div style="font-size:.8rem">Your preview will appear here</div></div></div>',
                unsafe_allow_html=True,
            )

if local_analysis:
    m = local_analysis.metrics
    st.markdown('<div class="section-heading">Technical image report</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Measured directly from the uploaded pixels using OpenCV and NumPy. These are not AI estimates.</div>', unsafe_allow_html=True)
    metric_cols = st.columns(4, gap="small")
    with metric_cols[0]: metric_card("Brightness", f"{m.brightness:.1f}/100", "Mean Rec. 709 luminance")
    with metric_cols[1]: metric_card("Contrast", f"{m.contrast:.1f}/100", "Luminance standard deviation")
    with metric_cols[2]: metric_card("Saturation", f"{m.saturation:.1f}%", "Mean HSV saturation")
    with metric_cols[3]: metric_card("Sharpness", f"{m.sharpness_variance:,.0f}", "Laplacian variance · relative")

    tab_cv, tab_colors, tab_ai, tab_actions = st.tabs(["Image signals", "Color palette", "AI visual review", "Recommendations"])
    with tab_cv:
        col_a, col_b = st.columns([1.1, .9], gap="large")
        with col_a:
            st.markdown("##### Pixel distribution")
            chart_data = {
                "Metric": ["Brightness", "Contrast", "Saturation"],
                "Value": [m.brightness, m.contrast, m.saturation],
            }
            st.bar_chart(chart_data, x="Metric", y="Value", height=220, use_container_width=True)
            st.caption("All three signals are normalized to a 0–100 display scale. They describe pixel characteristics, not design quality.")
        with col_b:
            st.markdown("##### Technical details")
            st.markdown(f"**Dimensions:** {m.dimensions} px  ")
            st.markdown(f"**Aspect ratio:** {m.aspect_ratio:.3f}  ")
            st.markdown(f"**Format:** {local_analysis.processed_image.metadata.format}  ")
            st.markdown(f"**Pixel count:** {local_analysis.processed_image.metadata.pixel_count:,}")
            st.markdown(f"**Near-black pixels:** {m.underexposed_percent:.2f}%  ")
            st.markdown(f"**Near-white pixels:** {m.overexposed_percent:.2f}%  ")
        st.markdown("##### Basic quality checks")
        for note in m.quality_notes:
            st.markdown(f"- {note}")
        st.markdown('<div class="small-note">Sharpness variance is content- and resolution-dependent. It should only be compared across similar images processed with the same method; it is not a universal sharpness grade.</div>', unsafe_allow_html=True)

    with tab_colors:
        st.markdown("##### Dominant color distribution")
        st.caption("OpenCV k-means clustering on a downsampled copy. Percentages are approximate pixel shares.")
        for color in m.dominant_colors:
            st.markdown(
                f'<div class="color-chip"><div class="swatch" style="background:{color.hex}"></div>'
                f'<b style="width:78px">{color.hex}</b><span>{color.percentage:.1f}%</span>'
                f'<span style="color:#718597">RGB {color.rgb[0]}, {color.rgb[1]}, {color.rgb[2]}</span></div>',
                unsafe_allow_html=True,
            )

    context_hash = hashlib.sha256(
        f"{image_hash}|{video_title}|{category}|{visual_style}|{language}".encode("utf-8")
    ).hexdigest()
    previous_context = st.session_state.get("ai_context_hash")
    if previous_context != context_hash:
        st.session_state.pop("ai_result", None)

    with tab_ai:
        st.markdown("##### Multimodal creative review")
        st.caption("The model inspects the actual thumbnail plus your title and creative context. Scores are subjective editorial heuristics, not CTR predictions.")
        if not settings.vision_is_configured:
            st.info("Local computer-vision results are ready. Configure a multimodal Vision API key to enable visual hierarchy, title relevance, and grounded creative recommendations.")
        run_ai = st.button("✦  Analyze with Vision AI", key="run_ai", disabled=not settings.vision_is_configured, use_container_width=True)
        if run_ai:
            try:
                with st.spinner("Reviewing composition, hierarchy, color, and title relevance…"):
                    result = analyze_with_vision(
                        local_analysis.processed_image.image,
                        settings,
                        title=video_title,
                        category=category,
                        desired_style=visual_style,
                        language=language,
                    )
                st.session_state["ai_result"] = result.model_dump()
                st.session_state["ai_context_hash"] = context_hash
                st.success("Vision analysis completed and validated.", icon="✦")
            except VisionAPIError as exc:
                st.error(str(exc), icon="⚠️")
            except Exception as exc:
                st.error("Unexpected analysis error. No AI result was saved. Check the configured provider and application logs.", icon="⚠️")

        ai_data = st.session_state.get("ai_result") if st.session_state.get("ai_context_hash") == context_hash else None
        if ai_data:
            from laqta_ai.schemas import ThumbnailAnalysis
            analysis = ThumbnailAnalysis.model_validate(ai_data)
            score = calculate_experimental_score(analysis)
            score_col, summary_col = st.columns([.25, .75], gap="large")
            with score_col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Editorial heuristic</div><div class="metric-value">{score}<span style="font-size:.85rem;color:#8395A6">/10</span></div><div class="metric-foot">Not a CTR score</div></div>', unsafe_allow_html=True)
            with summary_col:
                st.markdown(f'<div class="insight-card"><div class="insight-title">Creative director summary</div><div class="insight-body">{html.escape(analysis.overall_summary)}</div></div>', unsafe_allow_html=True)
            st.markdown("##### Visual dimensions")
            dims = analysis.visual_analysis.model_dump()
            for key, value in dims.items():
                label = DIMENSION_LABELS[key]
                c1, c2, c3 = st.columns([.29, .51, .20])
                with c1: st.markdown(f"**{label}**")
                with c2: st.progress(value["score"] / 10, text=value["rationale"])
                with c3: st.markdown(f'<span class="score-pill">{value["score"]}/10</span>', unsafe_allow_html=True)
                st.caption(f"Visible evidence: {value['evidence']}")
        else:
            st.markdown('<div class="small-note">No AI analysis is available for this image and context yet. Laqta AI will not fabricate a model response.</div>', unsafe_allow_html=True)

    with tab_actions:
        ai_data = st.session_state.get("ai_result") if st.session_state.get("ai_context_hash") == context_hash else None
        if ai_data:
            from laqta_ai.schemas import ThumbnailAnalysis
            analysis = ThumbnailAnalysis.model_validate(ai_data)
            st.markdown("##### What is already working")
            for index, point in enumerate(analysis.strengths, 1):
                show_design_point(point, index)
            st.markdown("##### Highest-value improvements")
            for index, point in enumerate(analysis.improvement_opportunities, 1):
                show_design_point(point, index)
            st.markdown("##### Prioritized action plan")
            for item in ordered_recommendations(analysis):
                level = item.priority.lower()
                st.markdown(
                    f'<div class="insight-card"><span class="priority priority-{level}">{html.escape(item.priority)} priority</span>'
                    f'<div class="insight-title">{html.escape(item.issue)}</div>'
                    f'<div class="insight-body"><b>Action:</b> {html.escape(item.action)}<br><br>'
                    f'<b>Expected design effect:</b> {html.escape(item.expected_design_effect)}</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown("##### Professional image-editing prompt")
            st.caption("Generated from this image review and your creative brief. Copy it into your preferred image-editing tool.")
            st.code(analysis.image_editing_prompt, language="text")
            report = {
                "product": "Laqta AI",
                "image": {"filename": uploaded.name, "sha256": image_hash, "metadata": asdict(local_analysis.processed_image.metadata)},
                "computer_vision_metrics": m.to_dict(),
                "ai_analysis": ai_data,
                "disclaimer": "AI dimension scores are subjective heuristics, not objective quality measures or YouTube CTR predictions.",
            }
            st.download_button(
                "Download analysis report · JSON",
                data=json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="laqta-ai-analysis.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("Run Vision AI analysis to unlock image-specific strengths, improvement opportunities, prioritized actions, and an editing prompt.")

else:
    with st.container(border=True):
        st.markdown('<div class="panel-kicker">Ready when you are</div><div class="panel-title">Your analysis workspace is waiting.</div><div class="panel-copy">Upload a thumbnail to calculate real image metrics. Connect a vision model to unlock image-aware creative feedback.</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#65798B;font-size:.72rem">'
    '<span>LAQTA AI · Analyze. Improve. Create.</span><span>Measured pixels ≠ subjective visual judgment ≠ predicted CTR</span></div>',
    unsafe_allow_html=True,
)
