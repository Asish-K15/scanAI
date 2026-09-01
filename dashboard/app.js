const API_URL = "http://127.0.0.1:8000/api/predict";

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


/* --------------------------------------------------
   Error handling
-------------------------------------------------- */

function showError(message) {
    errorText.textContent = message;

    show(errorMessage);
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


    show(resultCard);

    resultCard.scrollIntoView({
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


        displayResult(data);

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

newScreeningButton.addEventListener("click", () => {

    form.reset();

    clearSelectedImage();
    clearError();

    hide(resultCard);
    hide(loading);

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });
});


/* --------------------------------------------------
   Initial state
-------------------------------------------------- */

hide(loading);
hide(errorMessage);
hide(resultCard);
