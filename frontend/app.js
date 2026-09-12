"use strict";

const API_URL = "http://127.0.0.1:8000";

let lastQuestion = "";

document.addEventListener("DOMContentLoaded", () => {
    console.log("Grounded Research Agent frontend loaded.");

    const questionInput = document.querySelector("#question");
    const researchButton = document.querySelector("#research-btn");
    const buttonText = document.querySelector("#button-text");
    const buttonLoader = document.querySelector("#button-loader");
    const charCount = document.querySelector("#char-count");
    const retryButton = document.querySelector("#retry-btn");

    if (!questionInput) {
        console.error("Missing #question");
        return;
    }

    if (!researchButton) {
        console.error("Missing #research-btn");
        return;
    }

    // Character counter
    questionInput.addEventListener("input", () => {
        const length = questionInput.value.length;

        if (charCount) {
            charCount.textContent = `${length} / 2000`;
        }
    });

    // Main research button
    researchButton.addEventListener("click", () => {
        runResearch(questionInput.value);
    });

    // Ctrl + Enter
    questionInput.addEventListener("keydown", (event) => {
        if (event.ctrlKey && event.key === "Enter") {
            event.preventDefault();
            runResearch(questionInput.value);
        }
    });

    // Retry
    if (retryButton) {
        retryButton.addEventListener("click", () => {
            if (lastQuestion) {
                runResearch(lastQuestion);
            }
        });
    }

    console.log("Frontend event listeners attached.");
});


async function runResearch(rawQuestion) {
    const question = String(rawQuestion || "").trim();

    if (!question) {
        showError(
            "Please enter a research question.",
            "validation",
            "unknown"
        );
        return;
    }

    if (question.length > 2000) {
        showError(
            "Question is too long. Maximum 2000 characters.",
            "validation",
            "unknown"
        );
        return;
    }

    lastQuestion = question;

    clearError();
    hideResult();
    showPipeline();

    setButtonLoading(true);

    try {
        // STEP 1
        setPipelineStep("router", "active");
        setStatusMessage("AI router is deciding which source to use...");

        console.log("Sending research request:", question);

        // STEP 2
        setPipelineStep("retrieval", "active");

        const response = await fetch(`${API_URL}/research`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question
            })
        });

        console.log("Backend status:", response.status);

        let data;

        try {
            data = await response.json();
        } catch (jsonError) {
            throw new Error(
                "The research server returned an invalid response."
            );
        }

        console.log("Research response:", data);

        if (!response.ok) {
            throw new Error(
                data.detail ||
                data.error ||
                "Research request failed."
            );
        }

        // STEP 3
        setPipelineStep("retrieval", "complete");
        setPipelineStep("evidence", "active");
        setStatusMessage("Evidence retrieved. Validating evidence...");

        await shortDelay(150);

        // STEP 4
        setPipelineStep("evidence", "complete");
        setPipelineStep("answer", "active");
        setStatusMessage("Writing a grounded answer...");

        await shortDelay(150);

        setPipelineStep("answer", "complete");

        displayResult(data);

        setStatusMessage(
            "Research complete — this answer is backed by the evidence checked below."
        );

    } catch (error) {
        console.error("Research error:", error);

        showError(
            getSafeErrorMessage(error),
            "research",
            "unknown"
        );

        setStatusMessage("Research request failed.");

        resetPipeline();
    } finally {
        setButtonLoading(false);
    }
}


function displayResult(data) {
    console.log("Displaying result:", data);

    const resultSection =
        document.querySelector("#result-section");

    if (!resultSection) {
        console.error("Missing #result-section");
        return;
    }

    // Show result
    resultSection.classList.remove("hidden");
    resultSection.hidden = false;

    // Answer
    const answerElement =
        document.querySelector("#answer");

    if (answerElement) {
        answerElement.textContent =
            data.answer ||
            data.generated_answer ||
            "No grounded answer was returned.";
    }

    // Route
    const route =
        data.route || "unknown";

    const routeElement =
        document.querySelector("#result-route");

    if (routeElement) {
        routeElement.textContent =
            formatRoute(route);
    }

    // Evidence count
    const evidence =
        Array.isArray(data.evidence)
            ? data.evidence
            : [];

    const evidenceCount =
        evidence.length;

    const evidenceNumber =
        document.querySelector("#evidence-number");

    if (evidenceNumber) {
        evidenceNumber.textContent =
            String(evidenceCount);
    }

    const resultEvidence =
        document.querySelector("#result-evidence");

    if (resultEvidence) {
        resultEvidence.textContent =
            String(evidenceCount);
    }

    // Quality
    const quality =
        data.quality || "unavailable";

    const qualityStatus =
        document.querySelector("#quality-status");

    if (qualityStatus) {
        qualityStatus.textContent =
            `Quality ${formatQuality(quality)}`;
    }

    const detailQuality =
        document.querySelector("#detail-quality");

    if (detailQuality) {
        detailQuality.textContent =
            formatQuality(quality);
    }

    // Relevance
    const relevance =
        data.relevant;

    const relevanceText =
        formatBoolean(relevance);

    const relevanceStatus =
        document.querySelector("#relevance-status");

    if (relevanceStatus) {
        relevanceStatus.textContent =
            `• Relevance ${relevanceText}`;
    }

    const detailRelevance =
        document.querySelector("#detail-relevance");

    if (detailRelevance) {
        detailRelevance.textContent =
            relevanceText;
    }

    // Sufficiency
    const sufficient =
        data.sufficient;

    const sufficientText =
        formatBoolean(sufficient);

    const sufficiencyStatus =
        document.querySelector("#sufficiency-status");

    if (sufficiencyStatus) {
        sufficiencyStatus.textContent =
            `• Sufficiency ${sufficientText}`;
    }

    const detailSufficiency =
        document.querySelector("#detail-sufficiency");

    if (detailSufficiency) {
        detailSufficiency.textContent =
            sufficientText;
    }

    // Assessment / reason
    const assessment =
        document.querySelector("#assessment-text");

    if (assessment) {
        assessment.textContent =
            data.reason ||
            "The retrieved evidence was evaluated before answer generation.";
    }

    // Source preview
    const sourcePreview =
        document.querySelector("#evidence-source-preview");

    if (sourcePreview) {
        sourcePreview.textContent =
            `Route: ${formatRoute(route)} · ${evidenceCount} evidence item(s)`;
    }

    // Source count badge
    const sourceCount =
        document.querySelector("#source-count");

    const citations =
        Array.isArray(data.citations)
            ? data.citations
            : [];

    if (sourceCount) {
        sourceCount.textContent =
            `${citations.length} source${citations.length === 1 ? "" : "s"}`;
    }

    renderSources(citations);
    renderEvidence(evidence);

    // Result is now visible
    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


function renderSources(citations) {
    const container =
        document.querySelector("#sources");

    if (!container) {
        return;
    }

    container.innerHTML = "";

    if (
        !Array.isArray(citations) ||
        citations.length === 0
    ) {
        const empty =
            document.createElement("p");

        empty.textContent =
            "No verified sources were returned.";

        container.appendChild(empty);

        return;
    }

    citations.forEach((citation) => {
        if (
            typeof citation !== "string" ||
            !citation.startsWith("https://")
        ) {
            return;
        }

        const wrapper =
            document.createElement("div");

        wrapper.className =
            "source-item";

        const link =
            document.createElement("a");

        link.href = citation;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = citation;

        wrapper.appendChild(link);
        container.appendChild(wrapper);
    });
}


function renderEvidence(evidence) {
    const container =
        document.querySelector("#evidence");

    /*
     * Your current HTML does not contain an #evidence container.
     *
     * Evidence is still represented through:
     * - evidence-number
     * - evidence-source-preview
     * - assessment
     *
     * So we safely return if there is no detailed evidence container.
     */
    if (!container) {
        return;
    }

    container.innerHTML = "";

    if (!Array.isArray(evidence) || evidence.length === 0) {
        const empty =
            document.createElement("p");

        empty.textContent =
            "No evidence was available.";

        container.appendChild(empty);

        return;
    }

    evidence.forEach((item) => {
        if (!item || typeof item !== "object") {
            return;
        }

        const card =
            document.createElement("div");

        card.className =
            "evidence-item";

        const title =
            item.title ||
            item.question_title ||
            item.source ||
            item.source_type ||
            "Research evidence";

        const content =
            item.content ||
            item.body ||
            item.question_body ||
            item.answer ||
            item.answer_body ||
            "";

        const source =
            item.source ||
            item.source_type ||
            "";

        const titleElement =
            document.createElement("strong");

        titleElement.textContent =
            title;

        const sourceElement =
            document.createElement("div");

        sourceElement.textContent =
            source;

        const contentElement =
            document.createElement("div");

        contentElement.textContent =
            content;

        card.appendChild(titleElement);
        card.appendChild(sourceElement);
        card.appendChild(contentElement);

        container.appendChild(card);
    });
}


function showPipeline() {
    const pipeline =
        document.querySelector("#progress-section");

    if (!pipeline) {
        return;
    }

    pipeline.classList.remove("hidden");
    pipeline.hidden = false;
}


function hidePipeline() {
    const pipeline =
        document.querySelector("#progress-section");

    if (!pipeline) {
        return;
    }

    pipeline.classList.add("hidden");
    pipeline.hidden = true;
}


function resetPipeline() {
    const steps = [
        "router",
        "retrieval",
        "evidence",
        "answer"
    ];

    steps.forEach((step) => {
        setPipelineStep(step, "waiting");
    });
}


function setPipelineStep(name, state) {
    const element =
        document.querySelector(`#status-${name}`);

    if (!element) {
        return;
    }

    element.classList.remove(
        "active",
        "complete",
        "done",
        "current"
    );

    if (state === "active") {
        element.classList.add("active");
    }

    if (state === "complete") {
        element.classList.add("complete");
        element.classList.add("done");
    }

    const icon =
        element.querySelector(".step-icon");

    if (!icon) {
        return;
    }

    if (state === "active") {
        icon.textContent = "●";
    } else if (state === "complete") {
        icon.textContent = "✓";
    } else {
        icon.textContent = "○";
    }
}


function setStatusMessage(message) {
    console.log("STATUS:", message);
}


function showError(message, stage, route) {
    const errorSection =
        document.querySelector("#error-section");

    const errorMessage =
        document.querySelector("#error-message");

    const errorStage =
        document.querySelector("#error-stage");

    const errorRoute =
        document.querySelector("#error-route");

    if (errorMessage) {
        errorMessage.textContent =
            message;
    }

    if (errorStage) {
        errorStage.textContent =
            stage || "research";
    }

    if (errorRoute) {
        errorRoute.textContent =
            route || "unknown";
    }

    if (errorSection) {
        errorSection.classList.remove("hidden");
        errorSection.hidden = false;

        errorSection.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    }
}


function clearError() {
    const errorSection =
        document.querySelector("#error-section");

    if (!errorSection) {
        return;
    }

    errorSection.classList.add("hidden");
    errorSection.hidden = true;
}


function hideResult() {
    const resultSection =
        document.querySelector("#result-section");

    if (!resultSection) {
        return;
    }

    resultSection.classList.add("hidden");
    resultSection.hidden = true;
}


function setButtonLoading(isLoading) {
    const button =
        document.querySelector("#research-btn");

    const buttonText =
        document.querySelector("#button-text");

    const loader =
        document.querySelector("#button-loader");

    if (!button) {
        return;
    }

    button.disabled = isLoading;

    if (buttonText) {
        buttonText.textContent =
            isLoading
                ? "Researching..."
                : "Start research";
    }

    if (loader) {
        loader.classList.toggle(
            "hidden",
            !isLoading
        );
    }
}


function formatRoute(route) {
    const routes = {
        stackexchange: "Stack Exchange",
        weather: "Open-Meteo Weather",
        tavily: "Tavily Web Search",
        both: "Multiple Sources",
        unsupported: "Unsupported"
    };

    return routes[route] || route;
}


function formatQuality(value) {
    if (
        value === undefined ||
        value === null ||
        value === ""
    ) {
        return "Unavailable";
    }

    return String(value)
        .charAt(0)
        .toUpperCase() +
        String(value).slice(1);
}


function formatBoolean(value) {
    if (value === true) {
        return "Yes";
    }

    if (value === false) {
        return "No";
    }

    return "Unavailable";
}


function getSafeErrorMessage(error) {
    const message =
        error && error.message
            ? String(error.message)
            : "";

    if (!message) {
        return "Could not complete the research request.";
    }

    if (
        message.includes("Failed to fetch") ||
        message.includes("NetworkError") ||
        message.includes("fetch")
    ) {
        return (
            "Could not connect to the research server. " +
            "Make sure the backend is running on port 8000."
        );
    }

    return message;
}


function shortDelay(milliseconds) {
    return new Promise((resolve) => {
        setTimeout(resolve, milliseconds);
    });
}