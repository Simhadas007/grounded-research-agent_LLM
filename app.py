from guardrails.safety import check_user_input
from router import create_search_plan

from tools.stackexchange import search_stackexchange
from tools.weather import get_weather

from utils.research_answer import generate_research_answer

from security.tool_security import ToolSecurityController


def run_research(question: str) -> dict:
    """
    Run the complete grounded research pipeline.

    Flow:
        User Question
            -> Input Guardrail
            -> AI Router + Planner
            -> Tool Permission Control
            -> Source Retrieval
            -> Grounded Answer
            -> Final Result

    Security #7:
        Every real tool call must be authorized before execution.
        A fresh ToolSecurityController is created for every request.
    """

    # =========================================================
    # 1. INPUT GUARDRAIL
    # =========================================================

    input_check = check_user_input(question)

    if not input_check["allowed"]:
        return {
            "success": False,
            "stage": "input_guardrail",
            "error": input_check["reason"],
        }

    # =========================================================
    # 2. CREATE FRESH TOOL SECURITY STATE
    # =========================================================
    #
    # IMPORTANT:
    # A new controller is created for every user request.
    #
    # This prevents tool-call state from leaking between
    # different requests/users.
    #
    # Security #7
    # =========================================================

    tool_security = ToolSecurityController()

    # =========================================================
    # 3. AI ROUTER + PLANNER
    # =========================================================

    try:
        plan = create_search_plan(question)

    except Exception as exc:
        return {
            "success": False,
            "stage": "router",
            "error": f"Router failed: {exc}",
        }

    # =========================================================
    # 4. RETRIEVE EVIDENCE
    # =========================================================

    evidence = []

    # =========================================================
    # 4A. STACK EXCHANGE
    # =========================================================

    if plan.route in {"stackexchange", "both"}:

        # -----------------------------------------------------
        # Validate that the router actually supplied a query.
        # -----------------------------------------------------

        if plan.search_query:

            # -------------------------------------------------
            # SECURITY #7
            #
            # Authorize the tool BEFORE calling it.
            # -------------------------------------------------

            authorization = tool_security.authorize_tool(
                route=plan.route,
                tool="stackexchange",
            )

            if not authorization.allowed:
                return {
                    "success": False,
                    "stage": "tool_authorization",
                    "route": plan.route,
                    "error": (
                        "Stack Exchange tool execution denied: "
                        f"{authorization.reason}"
                    ),
                }

            # -------------------------------------------------
            # REAL TOOL EXECUTION
            #
            # This line cannot be reached unless authorization
            # succeeded.
            # -------------------------------------------------

            try:
                result = search_stackexchange(
                    plan.search_query
                )

                if result.get("success"):

                    stack_results = result.get(
                        "results",
                        []
                    )

                    for item in stack_results:

                        if isinstance(item, dict):

                            item["source_type"] = (
                                "stackexchange"
                            )

                    evidence.extend(stack_results)

            except Exception as exc:

                print(
                    f"Stack Exchange retrieval error: {exc}"
                )

    # =========================================================
    # 4B. WEATHER
    # =========================================================

    if plan.route in {"weather", "both"}:

        # -----------------------------------------------------
        # Validate that the router actually supplied a location.
        # -----------------------------------------------------

        if plan.location:

            # -------------------------------------------------
            # SECURITY #7
            #
            # Authorize the tool BEFORE calling it.
            # -------------------------------------------------

            authorization = tool_security.authorize_tool(
                route=plan.route,
                tool="weather",
            )

            if not authorization.allowed:
                return {
                    "success": False,
                    "stage": "tool_authorization",
                    "route": plan.route,
                    "error": (
                        "Weather tool execution denied: "
                        f"{authorization.reason}"
                    ),
                }

            # -------------------------------------------------
            # REAL TOOL EXECUTION
            #
            # This line cannot be reached unless authorization
            # succeeded.
            # -------------------------------------------------

            try:
                result = get_weather(
                    plan.location
                )

                if result.get("success"):

                    weather_data = result.get(
                        "data"
                    )

                    if weather_data:

                        weather_data["source_type"] = (
                            "weather"
                        )

                        evidence.append(
                            weather_data
                        )

            except Exception as exc:

                print(
                    f"Weather retrieval error: {exc}"
                )

    # =========================================================
    # 5. CHECK WHETHER WE GOT EVIDENCE
    # =========================================================

    if not evidence:

        return {
            "success": True,
            "stage": "completed",
            "route": plan.route,
            "plan_reason": plan.reason,
            "evidence_relevant": False,
            "evidence_sufficient": False,
            "evidence_quality": "low",
            "evidence_reason": (
                "No usable evidence was returned "
                "by the selected research sources."
            ),
            "answer": (
                "I could not find enough reliable information "
                "in the available research sources to answer "
                "this confidently."
            ),
            "citations": [],
        }

    # =========================================================
    # 6. GROUNDED ANSWER GENERATION
    # =========================================================

    try:

        # -----------------------------------------------------
        # The answer generator receives only the evidence
        # retrieved by the authorized tools.
        # -----------------------------------------------------

        research_answer = generate_research_answer(
            question=question,
            source=plan.route,
            evidence=evidence,
        )

    except Exception as exc:

        return {
            "success": False,
            "stage": "answer_generation",
            "route": plan.route,
            "error": (
                f"Answer generation failed: {exc}"
            ),
        }

    # =========================================================
    # 7. FINAL RESULT
    # =========================================================

    return {
        "success": True,
        "stage": "completed",
        "route": plan.route,
        "plan_reason": plan.reason,
        "evidence_relevant": research_answer.relevant,
        "evidence_sufficient": research_answer.sufficient,
        "evidence_quality": research_answer.quality,
        "evidence_reason": research_answer.reason,
        "answer": research_answer.answer,
        "citations": research_answer.citations,
    }


# =============================================================
# COMMAND-LINE INTERFACE
# =============================================================

if __name__ == "__main__":

    question = input(
        "\nEnter your research question: "
    ).strip()

    result = run_research(question)

    print("\n" + "=" * 70)
    print("GROUNDED RESEARCH RESULT")
    print("=" * 70)

    print(
        f"\nRoute: {result.get('route')}"
    )

    print(
        f"Success: {result.get('success')}"
    )

    if result.get("error"):

        print(
            f"Error: {result['error']}"
        )

    if result.get("answer"):

        print(
            f"\nAnswer:\n{result['answer']}"
        )

    citations = result.get(
        "citations",
        []
    )

    if citations:

        print("\nSources:")

        for citation in citations:

            print(
                f"- {citation}"
            )

    print(
        "\n" + "=" * 70
    )
