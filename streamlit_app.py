import streamlit as st
from app import run_research


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="Grounded Research Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
        }

        .hero {
            padding: 1.5rem 0 1rem 0;
        }

        .hero-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            color: #777;
            margin-bottom: 1.5rem;
        }

        .status-card {
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 10px;
            background: #fafafa;
            margin-bottom: 1rem;
        }

        .answer-card {
            padding: 1.25rem;
            border: 1px solid #ddd;
            border-radius: 12px;
            background: #ffffff;
        }

        .metric-card {
            padding: 1rem;
            border: 1px solid #ddd;
            border-radius: 10px;
            text-align: center;
            background: #fafafa;
        }

        .small-label {
            font-size: 0.8rem;
            color: #777;
        }

        .small-value {
            font-size: 1.15rem;
            font-weight: 600;
        }

        .evidence-item {
            padding: 0.8rem;
            border-left: 3px solid #888;
            background: #fafafa;
            margin-bottom: 0.7rem;
            border-radius: 4px;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🔎 Grounded Research Agent</div>
        <div class="hero-subtitle">
            Agentic AI research with live external evidence, source routing,
            security validation, and grounded answers.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:
    st.header("About")

    st.write(
        """
        This system uses an AI-driven routing process to decide which
        external source is appropriate for a research question.
        """
    )

    st.divider()

    st.subheader("Available Sources")

    st.write("🔴 Stack Exchange")
    st.write("🌤️ Open-Meteo Weather")
    st.write("🌐 Tavily Web Search")

    st.divider()

    st.subheader("Security")

    st.write("✓ Evidence sanitization")
    st.write("✓ Prompt-injection protection")
    st.write("✓ Citation validation")
    st.write("✓ Grounding checks")
    st.write("✓ Fail-closed behavior")

    st.divider()

    st.caption("Grounded Research Agent")


# ---------------------------------------------------------
# QUESTION INPUT
# ---------------------------------------------------------

st.subheader("Ask a research question")

question = st.text_area(
    "Question",
    placeholder=(
        "Examples:\n"
        "• What is the weather in Chennai today?\n"
        "• How do I read a file in Python?\n"
        "• What are the latest developments in cloud security?"
    ),
    height=150,
    max_chars=2000,
    label_visibility="collapsed",
)


# ---------------------------------------------------------
# RESEARCH BUTTON
# ---------------------------------------------------------

research_clicked = st.button(
    "🔎 Research",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# VALIDATE INPUT
# ---------------------------------------------------------

if research_clicked:

    if not question.strip():
        st.warning("Please enter a research question.")
        st.stop()

    # -----------------------------------------------------
    # RUN AGENT
    # -----------------------------------------------------

    with st.spinner("Researching and validating evidence..."):

        try:
            result = run_research(question.strip())

        except Exception:
            st.error(
                "The research request could not be completed. "
                "Please try again."
            )
            st.stop()

    # -----------------------------------------------------
    # NORMALIZE RESULT
    # -----------------------------------------------------

    if not isinstance(result, dict):
        st.error("The research agent returned an invalid response.")
        st.stop()

    success = result.get("success", False)
    stage = result.get("stage", "unknown")
    route = result.get("route", "unknown")

    answer = result.get(
        "answer",
        "I don't have enough grounded evidence to answer that safely.",
    )

    evidence = result.get("evidence", [])
    citations = result.get("citations", [])

    retrieval_status = result.get("retrieval_status", {})
    security_status = result.get("security_status", {})

    blocked_count = result.get("blocked_count", 0)

    quality = result.get("quality", "unknown")
    relevance = result.get("relevance")
    sufficiency = result.get("sufficiency")

    # -----------------------------------------------------
    # PIPELINE
    # -----------------------------------------------------

    st.divider()
    st.subheader("Research Pipeline")

    pipeline_cols = st.columns(4)

    with pipeline_cols[0]:
        st.success("01 ✓ Router")

    with pipeline_cols[1]:
        if retrieval_status:
            st.success("02 ✓ Retrieval")
        else:
            st.warning("02 Retrieval")

    with pipeline_cols[2]:
        if security_status:
            st.success("03 ✓ Evidence")
        else:
            st.warning("03 Evidence")

    with pipeline_cols[3]:
        if answer:
            st.success("04 ✓ Answer")
        else:
            st.warning("04 Answer")

    # -----------------------------------------------------
    # STATUS METRICS
    # -----------------------------------------------------

    st.divider()

    metric_cols = st.columns(5)

    with metric_cols[0]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Status</div>
                <div class="small-value">
                    {"Success" if success else "Declined"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_cols[1]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Route</div>
                <div class="small-value">{route}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_cols[2]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Evidence</div>
                <div class="small-value">{len(evidence)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_cols[3]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Quality</div>
                <div class="small-value">{quality}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_cols[4]:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Blocked</div>
                <div class="small-value">{blocked_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------
    # ANSWER
    # -----------------------------------------------------

    st.divider()
    st.subheader("Grounded Answer")

    st.markdown(
        f"""
        <div class="answer-card">
            {answer}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # GROUNDING ASSESSMENT
    # -----------------------------------------------------

    st.divider()
    st.subheader("Grounding Assessment")

    assessment_cols = st.columns(3)

    with assessment_cols[0]:
        if relevance is True:
            st.success("Relevance: Yes")
        elif relevance is False:
            st.error("Relevance: No")
        else:
            st.info("Relevance: Not reported")

    with assessment_cols[1]:
        if sufficiency is True:
            st.success("Sufficiency: Yes")
        elif sufficiency is False:
            st.error("Sufficiency: No")
        else:
            st.info("Sufficiency: Not reported")

    with assessment_cols[2]:
        st.info(f"Stage: {stage}")

    # -----------------------------------------------------
    # CITATIONS
    # -----------------------------------------------------

    if citations:

        st.divider()
        st.subheader("Sources")

        for index, citation in enumerate(citations, start=1):

            if isinstance(citation, dict):

                title = citation.get(
                    "title",
                    citation.get("name", f"Source {index}"),
                )

                url = citation.get("url", "")

                if url:
                    st.markdown(
                        f"{index}. [{title}]({url})"
                    )
                else:
                    st.write(f"{index}. {title}")

            elif isinstance(citation, str):

                if citation.startswith("http"):
                    st.markdown(
                        f"{index}. [{citation}]({citation})"
                    )
                else:
                    st.write(f"{index}. {citation}")

    # -----------------------------------------------------
    # EVIDENCE
    # -----------------------------------------------------

    st.divider()

    with st.expander(
        f"📚 Retrieved Evidence ({len(evidence)} item(s))",
        expanded=False,
    ):

        if not evidence:

            st.info(
                "No grounded evidence was retrieved for this question."
            )

        else:

            for index, item in enumerate(evidence, start=1):

                if not isinstance(item, dict):
                    st.write(item)
                    continue

                title = item.get(
                    "title",
                    item.get(
                        "question_title",
                        f"Evidence {index}",
                    ),
                )

                source = item.get(
                    "source",
                    item.get("site", "Unknown source"),
                )

                url = item.get("url", "")

                score = item.get("score")

                st.markdown(
                    f"""
                    <div class="evidence-item">
                        <strong>{index}. {title}</strong><br>
                        <span class="small-label">
                            Source: {source}
                            {" | Score: " + str(score) if score is not None else ""}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if url:
                    st.markdown(f"🔗 {url}")

                # Show useful evidence text without dumping everything
                body = (
                    item.get("answer_body")
                    or item.get("answer")
                    or item.get("body")
                    or item.get("question_body")
                    or ""
                )

                if body:
                    st.write(body[:3000])

    # -----------------------------------------------------
    # RETRIEVAL DETAILS
    # -----------------------------------------------------

    with st.expander("⚙️ Technical Details", expanded=False):

        st.write("**Route:**", route)
        st.write("**Stage:**", stage)
        st.write("**Success:**", success)
        st.write("**Evidence count:**", len(evidence))
        st.write("**Blocked evidence:**", blocked_count)
        st.write("**Quality:**", quality)

        if retrieval_status:
            st.write("**Retrieval status:**")
            st.json(retrieval_status)

        if security_status:
            st.write("**Security status:**")
            st.json(security_status)


# ---------------------------------------------------------
# INITIAL STATE
# ---------------------------------------------------------

else:

    st.info(
        "Enter a question above and click **Research**. "
        "The agent will choose the appropriate source, retrieve "
        "live evidence, validate it, and generate a grounded answer."
    )