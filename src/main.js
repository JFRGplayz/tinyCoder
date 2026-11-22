
import { Transformer } from "./transformer.js";
import { Tokenizer } from "./tokenizer.js";
import { Adam, trainChatExample } from "./training.js";
import { generate } from "./generation.js";

import {
    openMemory,
    saveRecord,
    getRecords,
    saveSetting,
    getSetting
} from "./memory.js";

const $ = id => document.getElementById(id);

// ========================================
// MODEL SETUP
// ========================================

const tokenizer = new Tokenizer();

const model = new Transformer(
    tokenizer.size,
    8,
    32
);

const parameters = model.parameters();
const optimizer = new Adam(parameters, 0.001);

let trainingSteps = 0;
let exampleCount = 0;
let busy = false;

// Current conversation context
let chatHistory = [];

const welcome = $("welcome");
const messages = $("messages");
const input = $("message-input");
const sendButton = $("send-button");

// ========================================
// CHAT UI
// ========================================

function addMessage(role, content) {
    welcome.style.display = "none";

    const container = document.createElement("div");
    container.className = `message ${role}`;

    const author = document.createElement("div");
    author.className = "message-author";
    author.textContent = role === "user" ? "You" : "Tiny";

    const body = document.createElement("div");
    body.className = "message-content";
    body.textContent = content;

    container.append(author, body);
    messages.appendChild(container);

    $("chat-area").scrollTop =
        $("chat-area").scrollHeight;

    return container;
}

function updateStats() {
    $("example-count").textContent = exampleCount;
    $("training-count").textContent = trainingSteps;
}

// ========================================
// MODEL SAVE / LOAD
// ========================================

function saveModel() {
    const weights = parameters.map(parameter => parameter.data);

    return saveSetting("modelWeights", weights);
}

async function loadModel() {
    const weights = await getSetting("modelWeights");

    if (!weights || weights.length !== parameters.length) {
        return false;
    }

    weights.forEach((weight, index) => {
        parameters[index].data = weight;
    });

    return true;
}

// ========================================
// TEACHING
// ========================================

async function teach(question, answer) {
    const loss = trainChatExample(
        model,
        optimizer,
        tokenizer,
        question,
        answer
    );

    await saveRecord("training", {
        question,
        answer,
        created: Date.now()
    });

    await saveModel();

    exampleCount++;
    trainingSteps++;

    updateStats();

    return loss;
}

async function restoreTraining() {
    const examples = await getRecords("training");

    exampleCount = examples.length;

    updateStats();
}

// ========================================
// CONVERSATION
// ========================================

function buildPrompt(newMessage) {
    // Keep a few recent exchanges for context.
    const recentHistory = chatHistory.slice(-3);

    let prompt = "";

    for (const exchange of recentHistory) {
        prompt += `User: ${exchange.user}\n`;
        prompt += `Tiny: ${exchange.assistant}\n`;
    }

    prompt += `User: ${newMessage}\nTiny:`;

    return prompt;
}

async function sendMessage() {
    const text = input.value.trim();

    if (!text || busy) return;

    input.value = "";

    addMessage("user", text);

    busy = true;
    sendButton.disabled = true;

    try {
        // ====================================
        // TEACHING COMMAND
        // ====================================

        if (text.startsWith("!teach")) {
            const content = text.slice(6).trim();
            const parts = content.split("\n---\n");

            if (parts.length !== 2) {
                addMessage(
                    "assistant",
                    "Teaching format:\n\n!teach\nYour question\n---\nThe correct answer"
                );
            } else {
                const question = parts[0].trim();
                const answer = parts[1].trim();

                if (!question || !answer) {
                    addMessage(
                        "assistant",
                        "Both a question and an answer are required."
                    );

                    return;
                }

                addMessage(
                    "assistant",
                    "Learning this example... my tiny brain is working."
                );

                const loss = await teach(question, answer);

                addMessage(
                    "assistant",
                    `I've trained on that example!\nTraining loss: ${loss.toFixed(4)}\nExamples learned: ${exampleCount}`
                );
            }

            return;
        }

        // ====================================
        // NORMAL CHAT
        // ====================================

        const prompt = buildPrompt(text);

        const response = generate(
            model,
            tokenizer,
            prompt,
            120
        );

        const cleanResponse =
            response.trim() || "I'm not sure what to say yet.";

        addMessage("assistant", cleanResponse);

        // Keep conversation in memory for context.
        chatHistory.push({
            user: text,
            assistant: cleanResponse
        });

        // Save conversation to IndexedDB.
        await saveRecord("conversations", {
            user: text,
            assistant: cleanResponse,
            created: Date.now()
        });

    } catch (error) {
        console.error(error);

        addMessage(
            "assistant",
            `Something went wrong: ${error.message}`
        );
    } finally {
        busy = false;
        sendButton.disabled = false;
        input.focus();
    }
}

// ========================================
// INITIALISATION
// ========================================

async function initialise() {
    try {
        await openMemory();

        await loadModel();
        await restoreTraining();

        $("status-indicator").classList.add("ready");
        $("status-text").textContent = "Ready";
        $("header-status").textContent = "Running locally";

        addMessage(
            "assistant",
            "Hello! I'm Tiny. I'm still learning how to chat, but every example you teach me helps train my neural network. Try talking to me, or use !teach to teach me something."
        );

    } catch (error) {
        console.error(error);

        $("status-text").textContent = "Initialisation failed";

        addMessage(
            "assistant",
            "I couldn't initialise my local memory. Please check your browser console."
        );
    }
}

// ========================================
// EVENT LISTENERS
// ========================================

sendButton.addEventListener("click", sendMessage);

input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

$("new-chat").addEventListener("click", () => {
    messages.replaceChildren();

    chatHistory = [];

    welcome.style.display = "block";
});

document.querySelectorAll("[data-prompt]").forEach(button => {
    button.addEventListener("click", () => {
        input.value = button.dataset.prompt;
        input.focus();
    });
});

initialise();
