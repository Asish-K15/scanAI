const API_URL = "/api/predict";

const form = document.getElementById("predictionForm");

const speciesSelect = document.getElementById("species");
const bodyAreaSelect = document.getElementById("bodyArea");
const imageInput = document.getElementById("image");

const uploadArea = document.getElementById("uploadArea");
const browseButton = document.getElementById("browseButton");

const uploadContent = document.getElementById("uploadContent");
const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");
const removeImageButton = document.getElementById("removeImage");

const analyzeButton = document.getElementById("analyzeButton");

const loading = document.getElementById("loading");
const errorMessage = document.getElementById("errorMessage");
const errorText = document.getElementById("errorText");

const resultCard = document.getElementById("resultCard");

const resultCondition = document.getElementById("resultCondition");
const confidenceBadge = document.getElementById("confidenceBadge");

const resultSpecies = document.getElementById("resultSpecies");
const resultBodyArea = document.getElementById("resultBodyArea");
const resultConfidence = document.getElementById("resultConfidence");
const resultConfidenceLevel = document.getElementById(
    "resultConfidenceLevel"
);

const uncertaintyText = document.getElementById("uncertaintyText");

const resultSeverity = document.getElementById("resultSeverity");
const resultUrgency = document.getElementById("resultUrgency");
const resultEvidenceStatus = document.getElementById(
    "resultEvidenceStatus"
);

const resultRecommendation = document.getElementById(
    "resultRecommendation"
);

const evidenceOutput = document.getElementById("evidenceOutput");

const newScreeningButton = document.getElementById("newScreening");

const gateResultCard = document.getElementById("gateResultCard");
const gateEyebrow = document.getElementById("gateEyebrow");
const gateTitle = document.getElementById("gateTitle");
const gateDecisionBadge = document.getElementById("gateDecisionBadge");
const gateNoticeBox = document.getElementById("gateNoticeBox");
const gateNoticeIcon = document.getElementById("gateNoticeIcon");
const gateNoticeHeadline = document.getElementById("gateNoticeHeadline");
const gateNoticeDescription = document.getElementById("gateNoticeDescription");
const gateDecision = document.getElementById("gateDecision");
const gateReasonCode = document.getElementById("gateReasonCode");
const gatePredictedSpecies = document.getElementById("gatePredictedSpecies");
const gateModelVersion = document.getElementById("gateModelVersion");
const gateReasonMessage = document.getElementById("gateReasonMessage");
const gateEvidenceOutput = document.getElementById("gateEvidenceOutput");
const gateNewScreeningButton = document.getElementById("gateNewScreening");

let selectedFile = null;


/* --------------------------------------------------
   Utility functions
-------------------------------------------------- */

function show(element) {
    element.classList.remove("hidden");
}


function hide(element) {
    element.classList.add("hidden");
}


function formatCondition(condition) {
    if (!condition) {
        return "Unknown";
    }

    return condition
        .replace(/^skin__/, "")
        .replace(/_/g, " ")
        .replace(/\b\w/g, (letter) => letter.toUpperCase());
}


function formatValue(value) {
    if (value === null || value === undefined || value === "") {
        return "Not available";
    }

    return String(value)
        .replace(/_/g, " ")
        .replace(/\b\w/g, (letter) => letter.toUpperCase());
}


function formatConfidence(value) {
    if (typeof value !== "number") {
        return "—";
    }

    return `${(value * 100).toFixed(2)}%`;
}


function getConfidenceClass(level) {
    if (level === "high") {
        return "confidence-high";
    }

    if (level === "moderate") {
        return "confidence-moderate";
    }

    if (level === "low") {
        return "confidence-low";
    }

    return "";
}


function formatReasonCode(code) {
    if (!code) {
        return "Not available";
    }

    return String(code).trim();
}


function getReasonExplanation(reasonCode, decision) {
    if (!reasonCode) {
        return decision === "REJECT"
            ? "The uploaded image did not pass the safety gate criteria. Disease analysis was halted."
            : "The safety gate could not confirm image validity with sufficient confidence. Disease analysis was halted.";
    }

    const code = String(reasonCode).trim();
    const normalized = code.toUpperCase();

    switch (normalized) {
        case "NON_ANIMAL_DETECTED":
            return "No domestic animal was detected in the uploaded image. ScanAI only processes animal images.";
        case "HUMAN_DETECTED":
            return "A human subject was detected in the image. ScanAI is strictly restricted to animal health screening.";
        case "SPECIES_MISMATCH":
            return "The animal species detected in the image does not match the species selected in the screening form.";
        case "LOW_CONFIDENCE":
            return "Safety gate verification confidence is below the approved threshold to proceed with disease analysis.";
        case "UNSUPPORTED_SPECIES":
            return "The selected or detected animal species is not supported for clinical screening.";
        case "INVALID_IMAGE":
            return "The uploaded image file is invalid, corrupted, or cannot be processed.";
        case "MODEL_ERROR":
            return "An internal error occurred during safety gate verification.";
        case "UNCERTAIN":
            return "The safety gate was unable to conclusively verify animalness or species.";
        default:
            return `Safety gate validation halted with reason code: ${code}.`;
    }
}


/* --------------------------------------------------
   Error handling
-------------------------------------------------- */

function showError(message) {
    errorText.textContent = message;

    show(errorMessage);
    hide(resultCard);
    hide(gateResultCard);
}


function clearError() {
    errorText.textContent = "";
    hide(errorMessage);
}


/* --------------------------------------------------
   Image selection
-------------------------------------------------- */

function setSelectedImage(file) {
    if (!file) {
        return;
    }

    if (!file.type.startsWith("image/")) {
        showError("Please select a valid image file.");
        return;
    }

    selectedFile = file;

    const objectUrl = URL.createObjectURL(file);

    imagePreview.src = objectUrl;
    fileName.textContent = file.name;

    hide(uploadContent);
    show(previewContainer);

    clearError();
}


function clearSelectedImage() {
    selectedFile = null;

    imageInput.value = "";

    imagePreview.removeAttribute("src");
    fileName.textContent = "";

    show(uploadContent);
    hide(previewContainer);
}


browseButton.addEventListener("click", (event) => {
    event.stopPropagation();
    imageInput.click();
});


uploadArea.addEventListener("click", () => {
    if (!selectedFile) {
        imageInput.click();
    }
});


imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];

    if (file) {
        setSelectedImage(file);
    }
});


removeImageButton.addEventListener("click", (event) => {
    event.stopPropagation();
    clearSelectedImage();
});


/* --------------------------------------------------
   Drag and drop
-------------------------------------------------- */

uploadArea.addEventListener("dragover", (event) => {
    event.preventDefault();

    uploadArea.classList.add("dragover");
});


uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
});


uploadArea.addEventListener("drop", (event) => {
    event.preventDefault();

    uploadArea.classList.remove("dragover");

    const file = event.dataTransfer.files[0];

    if (file) {
        setSelectedImage(file);
    }
});


/* --------------------------------------------------
   Display result
-------------------------------------------------- */

function displayResult(data) {

    resultCondition.textContent =
        formatCondition(data.condition);

    resultSpecies.textContent =
        formatValue(data.species);

    resultBodyArea.textContent =
        formatValue(data.body_area);

    resultConfidence.textContent =
        formatConfidence(data.confidence);

    resultConfidenceLevel.textContent =
        formatValue(data.confidence_level);


    /* ----------------------------------------------
       Confidence badge
    ---------------------------------------------- */

    const confidenceLevel =
        data.confidence_level || "unknown";

    confidenceBadge.textContent =
        formatValue(confidenceLevel);

    confidenceBadge.className =
        "confidence-badge " +
        getConfidenceClass(confidenceLevel);


    /* ----------------------------------------------
       Uncertainty
    ---------------------------------------------- */

    if (data.uncertain === true) {

        uncertaintyText.textContent =
            "The model is uncertain about this screening result. " +
            "The result should be treated as a screening indication " +
            "and not as a clinical diagnosis.";

    } else {

        uncertaintyText.textContent =
            "The model returned a screening prediction. " +
            "This result is not a clinical diagnosis.";
    }


    /* ----------------------------------------------
       Clinical fields
       IMPORTANT:
       We display the API values exactly as returned.
       We do NOT calculate severity or urgency here.
    ---------------------------------------------- */

    resultSeverity.textContent =
        data.severity === null
            ? "Not determined"
            : formatValue(data.severity);

    resultUrgency.textContent =
        data.urgency === null
            ? "Not determined"
            : formatValue(data.urgency);

    resultEvidenceStatus.textContent =
        formatValue(data.evidence_status);


    /* ----------------------------------------------
       Recommendation
    ---------------------------------------------- */

    resultRecommendation.textContent =
        data.recommendation ||
        "No recommendation available.";


    /* ----------------------------------------------
       Model evidence
    ---------------------------------------------- */

    if (data.evidence) {

        evidenceOutput.textContent =
            JSON.stringify(data.evidence, null, 2);

    } else {

        evidenceOutput.textContent =
            "No model evidence returned.";
    }


    hide(gateResultCard);
    show(resultCard);

    resultCard.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


/* --------------------------------------------------
   Display safety gate result
-------------------------------------------------- */

function displayGateResult(data) {
    hide(resultCard);
    clearError();

    const decisionUpper =
        typeof data.decision === "string"
            ? data.decision.toUpperCase()
            : "REJECT";

    const isReject = decisionUpper === "REJECT";

    if (isReject) {
        gateResultCard.classList.remove("gate-uncertain-card");
        gateResultCard.classList.add("gate-reject-card");

        gateEyebrow.textContent = "SAFETY GATE REJECTION";
        gateEyebrow.className = "eyebrow gate-eyebrow gate-eyebrow-reject";

        gateTitle.textContent = "Analysis Stopped — Image Rejected";

        gateDecisionBadge.textContent = "REJECT";
        gateDecisionBadge.className = "gate-badge gate-badge-reject";

        gateNoticeBox.className = "gate-notice-box gate-notice-reject";
        gateNoticeIcon.textContent = "✕";
        gateNoticeHeadline.textContent = "ScanAI stopped the analysis";
        gateNoticeDescription.textContent =
            "The image did not pass the animal/species safety gate. Disease screening was not performed.";
    } else {
        gateResultCard.classList.remove("gate-reject-card");
        gateResultCard.classList.add("gate-uncertain-card");

        gateEyebrow.textContent = "SAFETY GATE UNCERTAIN";
        gateEyebrow.className = "eyebrow gate-eyebrow gate-eyebrow-uncertain";

        gateTitle.textContent = "Analysis Stopped — Gate Uncertain";

        gateDecisionBadge.textContent = "UNCERTAIN";
        gateDecisionBadge.className = "gate-badge gate-badge-uncertain";

        gateNoticeBox.className = "gate-notice-box gate-notice-uncertain";
        gateNoticeIcon.textContent = "!";
        gateNoticeHeadline.textContent = "ScanAI stopped the analysis";
        gateNoticeDescription.textContent =
            "The safety gate could not confirm with sufficient confidence that the image contains a supported animal/species.";
    }

    gateDecision.textContent = formatValue(data.decision);
    gateReasonCode.textContent = formatReasonCode(data.reason_code);
    gatePredictedSpecies.textContent = formatValue(data.predicted_species);
    gateModelVersion.textContent =
        data.model_version || data.model_name || "SCANAI-ANIMALNESS-GATE-V1";

    gateReasonMessage.textContent = getReasonExplanation(
        data.reason_code,
        decisionUpper
    );

    if (gateEvidenceOutput) {
        gateEvidenceOutput.textContent = JSON.stringify(data, null, 2);
    }

    show(gateResultCard);

    gateResultCard.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


/* --------------------------------------------------
   API request
-------------------------------------------------- */

async function submitPrediction() {

    clearError();

    hide(resultCard);
    hide(gateResultCard);

    if (!selectedFile) {
        showError("Please upload an animal image.");
        return;
    }


    const species = speciesSelect.value;
    const bodyArea = bodyAreaSelect.value;


    if (!species) {
        showError("Please select the animal species.");
        speciesSelect.focus();
        return;
    }


    if (!bodyArea) {
        showError("Please select the body area.");
        bodyAreaSelect.focus();
        return;
    }


    const formData = new FormData();

    formData.append("image", selectedFile);
    formData.append("species", species);
    formData.append("body_area", bodyArea);


    analyzeButton.disabled = true;
    analyzeButton.textContent = "Analyzing...";

    show(loading);


    try {

        const response = await fetch(
            API_URL,
            {
                method: "POST",
                body: formData
            }
        );


        let data = null;

        try {
            data = await response.json();
        } catch (jsonError) {
            data = null;
        }


        if (!response.ok) {

            let message =
                `API request failed (${response.status}).`;

            if (data) {

                if (typeof data.detail === "string") {
                    message = data.detail;
                } else if (Array.isArray(data.detail)) {
                    message = data.detail
                        .map((item) => item.msg || String(item))
                        .join(", ");
                }
            }

            throw new Error(message);
        }


        if (!data || typeof data !== "object") {
            throw new Error(
                "The API returned an invalid response."
            );
        }


        const decision =
            typeof data.decision === "string"
                ? data.decision.toUpperCase()
                : null;

        if (decision === "REJECT" || decision === "UNCERTAIN") {
            displayGateResult(data);
        } else {
            displayResult(data);
        }

    } catch (error) {

        console.error("ScanAI API error:", error);

        showError(
            error.message ||
            "Unable to connect to the ScanAI API."
        );

    } finally {

        hide(loading);

        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analyze Image";
    }
}


/* --------------------------------------------------
   Form submit
-------------------------------------------------- */

form.addEventListener("submit", async (event) => {

    event.preventDefault();

    await submitPrediction();
});


/* --------------------------------------------------
   New screening
-------------------------------------------------- */

function resetScreening() {
    form.reset();

    clearSelectedImage();
    clearError();

    hide(resultCard);
    hide(gateResultCard);
    hide(loading);

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
}

newScreeningButton.addEventListener("click", resetScreening);

if (gateNewScreeningButton) {
    gateNewScreeningButton.addEventListener("click", resetScreening);
}


/* --------------------------------------------------
   Initial state
-------------------------------------------------- */

hide(loading);
hide(errorMessage);
hide(resultCard);
hide(gateResultCard);
