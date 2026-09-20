/* =========================================================
   SECURE ASKRAG — FRONTEND CONTROLLER
   ========================================================= */

const API_BASE = "http://127.0.0.1:8000";


// =========================================================
// STATE
// =========================================================

const state = {
    userId: "",
    password: "",
    role: "",
    selectedDocument: null,
};


// =========================================================
// DOM HELPERS
// =========================================================

function $(id) {
    return document.getElementById(id);
}


function show(element) {
    if (element) {
        element.classList.remove("hidden");
    }
}


function hide(element) {
    if (element) {
        element.classList.add("hidden");
    }
}


// =========================================================
// ELEMENTS
// =========================================================

const loginSection = $("login-section");
const askSection = $("ask-section");

const userIdInput = $("user-id");
const passwordInput = $("password");

const loginButton = $("login-btn");
const loginMessage = $("login-message");

const userBadge = $("user-badge");
const loggedUser = $("logged-user");
const loggedRole = $("logged-role");
const accessRole = $("access-role");

const questionInput = $("question");
const askButton = $("ask-btn");

const selectedDocumentLabel = $("selected-document");

const loading = $("loading");
const result = $("result");

const decision = $("decision");
const decisionIndicator = $("decision-indicator");

const risk = $("risk");
const riskScore = $("risk-score");
const evidenceScore = $("evidence-score");

const answer = $("answer");
const source = $("source");
const reason = $("reason");


// =========================================================
// INITIAL STATE
// =========================================================

hide(askSection);
hide(userBadge);
hide(loading);
hide(result);


// =========================================================
// LOGIN
// =========================================================

async function login() {

    const userId = userIdInput.value.trim();
    const password = passwordInput.value;

    loginMessage.textContent = "";

    if (!userId || !password) {

        loginMessage.textContent =
            "Please enter both User ID and password.";

        return;
    }


    loginButton.disabled = true;

    loginButton.querySelector("span:first-child").textContent =
        "Authenticating...";


    try {

        const response = await fetch(
            `${API_BASE}/login`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify({
                    user_id: userId,
                    password: password,
                }),
            }
        );


        let data;

        try {
            data = await response.json();
            console.log("========== API RESPONSE ==========");
            console.log(data);
            console.log("==================================");
        } catch {
            data = {};
        }


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Authentication failed."
            );
        }


        // ---------------------------------------------
        // SAVE SESSION STATE
        // ---------------------------------------------

        state.userId = userId;
        state.password = password;
        state.role = data.role || "UNKNOWN";


        // ---------------------------------------------
        // UPDATE USER UI
        // ---------------------------------------------

        loggedUser.textContent = state.userId;
        loggedRole.textContent = state.role;

        accessRole.textContent = state.role;


        show(userBadge);
        hide(loginSection);
        show(askSection);


        loginMessage.textContent = "";


    } catch (error) {

        loginMessage.textContent =
            error.message ||
            "Unable to connect to the authentication service.";

    } finally {

        loginButton.disabled = false;

        loginButton.querySelector("span:first-child").textContent =
            "Authenticate";
    }
}


// =========================================================
// DOCUMENT SELECTION
// =========================================================

function selectDocument(button) {

    const filename = button.dataset.document;

    if (!filename) {
        return;
    }


    // Remove active state from every document

    document
        .querySelectorAll(".document-item")
        .forEach(item => {
            item.classList.remove("active");
        });


    // Activate selected document

    button.classList.add("active");


    state.selectedDocument = filename;


    selectedDocumentLabel.textContent = filename;


    // Clear previous result

    hide(result);


    // Focus question field

    questionInput.focus();
}


document
    .querySelectorAll(".document-item")
    .forEach(button => {

        button.addEventListener(
            "click",
            () => selectDocument(button)
        );

    });


// =========================================================
// ASK QUESTION
// =========================================================

async function askQuestion() {

    const question = questionInput.value.trim();


    // ---------------------------------------------
    // VALIDATION
    // ---------------------------------------------

    if (!state.userId || !state.password) {

        alert(
            "Please authenticate before asking a question."
        );

        return;
    }


    if (!state.selectedDocument) {

        alert(
            "Please select a document first."
        );

        return;
    }


    if (!question) {

        alert(
            "Please enter a question."
        );

        questionInput.focus();

        return;
    }


    // ---------------------------------------------
    // UI — LOADING
    // ---------------------------------------------

    hide(result);
    show(loading);

    askButton.disabled = true;


    const originalButtonHTML =
        askButton.innerHTML;


    askButton.innerHTML =
        "<span>Processing...</span><span>↻</span>";


    try {

        const response = await fetch(
            `${API_BASE}/ask`,
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json",
                },

                body: JSON.stringify({

                    user_id: state.userId,

                    password: state.password,

                    question: question,

                    filename: state.selectedDocument,

                }),
            }
        );


        let data;

        try {
            data = await response.json();
        } catch {

            throw new Error(
                "The API returned an invalid response."
            );
        }


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "The secure query failed."
            );
        }


        // ---------------------------------------------
        // DISPLAY RESULT
        // ---------------------------------------------

        renderResult(data);


    } catch (error) {

        renderError(
            error.message ||
            "Unable to communicate with Secure AskRAG."
        );

    } finally {

        hide(loading);

        askButton.disabled = false;

        askButton.innerHTML =
            originalButtonHTML;
    }
}


// =========================================================
// RENDER RESULT
// =========================================================

function renderResult(data) {

    show(result);


    const decisionValue =
        String(data.decision || "UNKNOWN")
            .toUpperCase();


    const riskValue =
        data.risk ??
        data.risk_level ??
        "—";


    const riskScoreValue =
        data.risk_score ??
        "—";


    const evidenceScoreValue =
        data.evidence_score ??
        "—";


    // ---------------------------------------------
    // DECISION
    // ---------------------------------------------

    decision.textContent =
        decisionValue;


    decisionIndicator.textContent =
        decisionValue;


    decisionIndicator.classList.remove(
        "decision-allow",
        "decision-deny"
    );


    if (decisionValue === "ALLOW") {

        decisionIndicator.classList.add(
            "decision-allow"
        );

    } else {

        decisionIndicator.classList.add(
            "decision-deny"
        );
    }


    // ---------------------------------------------
    // METRICS
    // ---------------------------------------------

    risk.textContent =
        String(riskValue).toUpperCase();


    riskScore.textContent =
        formatNumber(riskScoreValue);


    evidenceScore.textContent =
        formatEvidenceScore(evidenceScoreValue);


    // ---------------------------------------------
    // ANSWER
    // ---------------------------------------------

    answer.textContent =
        data.answer ||
        "No answer was returned.";


    // ---------------------------------------------
    // SOURCE
source.textContent =
    formatSource(
        data.sources || data.source
    );


    // ---------------------------------------------
    reason.textContent =
    data.reason ||
    data.security_reason ||
    (
        data.decision === "ALLOW"
            ? "Request allowed based on permitted evidence."
            : "No security reason provided."
    );

    // ---------------------------------------------
    // SCROLL TO RESULT
    // ---------------------------------------------

    result.scrollIntoView({
        behavior: "smooth",
        block: "start",
    });
}


// =========================================================
// ERROR RESULT
// =========================================================

function renderError(message) {

    show(result);


    decision.textContent =
        "ERROR";


    decisionIndicator.textContent =
        "ERROR";


    decisionIndicator.classList.remove(
        "decision-allow"
    );


    decisionIndicator.classList.add(
        "decision-deny"
    );


    risk.textContent = "—";
    riskScore.textContent = "—";
    evidenceScore.textContent = "—";


    answer.textContent =
        message;


    source.textContent =
        state.selectedDocument ||
        "—";


    reason.textContent =
        "The request could not be completed.";
}


// =========================================================
// FORMAT HELPERS
// =========================================================

function formatNumber(value) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return "—";
    }


    if (typeof value === "number") {

        return Number.isInteger(value)
            ? String(value)
            : value.toFixed(2);
    }


    return String(value);
}


function formatEvidenceScore(value) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return "—";
    }


    const numeric =
        Number(value);


    if (Number.isNaN(numeric)) {
        return String(value);
    }


    // If backend returns 0–1,
    // display it as a percentage.

    if (numeric >= 0 && numeric <= 1) {

        return `${(numeric * 100).toFixed(0)}%`;
    }


    return String(value);
}


function formatSource(value) {

    if (!value) {
        return "No source returned.";
    }


    if (Array.isArray(value)) {

        return value
            .map(item => {

                if (
                    typeof item === "string"
                ) {
                    return `• ${item}`;
                }

                return `• ${
                    item.filename ||
                    item.source ||
                    JSON.stringify(item)
                }`;

            })
            .join("\n");
    }


    if (typeof value === "object") {

        return JSON.stringify(
            value,
            null,
            2
        );
    }


    return String(value);
}


// =========================================================
// KEYBOARD SHORTCUT
// =========================================================

questionInput.addEventListener(
    "keydown",
    event => {

        if (
            event.ctrlKey &&
            event.key === "Enter"
        ) {

            event.preventDefault();

            askQuestion();
        }

    }
);


// =========================================================
// ENTER KEY ON LOGIN
// =========================================================

passwordInput.addEventListener(
    "keydown",
    event => {

        if (event.key === "Enter") {

            event.preventDefault();

            login();
        }

    }
);


// =========================================================
// BUTTON EVENTS
// =========================================================

loginButton.addEventListener(
    "click",
    login
);


askButton.addEventListener(
    "click",
    askQuestion
);


// =========================================================
// FRONTEND STARTUP
// =========================================================

console.log(
    "Secure AskRAG frontend initialized."
);

console.log(
    `API endpoint: ${API_BASE}`
);