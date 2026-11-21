
import { Transformer } from "./transformer.js";
import { Tokenizer } from "./tokenizer.js";
import { Adam, trainExample } from "./training.js";
import { generate } from "./generation.js";

import {
    openMemory,
    saveRecord,
    getRecords,
    saveSetting,
    getSetting
} from "./memory.js";

const $ = id => document.getElementById(id);

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

const welcome = $("welcome");
const messages = $("messages");
const input = $("message-input");
const sendButton = $("send-button");

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

    $("chat-area").scrollTop = $("chat-area").scrollHeight;

    return container;
}

function updateStats() {
    $("example-count").textContent = exampleCount;
    $("training-count").textContent = trainingSteps;
}

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

async function teach(question, answer) {
    const text = `User: ${question}\nTiny: ${answer}\n`;

    const tokens = tokenizer.encode(text);

    if (tokens.length < 2) {
        throw new Error("Training example is too short.");
    }

    // Train using bounded chunks.
    // Each chunk predicts its following characters.
    const chunkSize = model.contextSize;

    for (let start = 0; start < tokens.length - 1; start += chunkSize) {
        const chunk = tokens.slice(start, start + chunkSize + 1);

        if (chunk.length < 2) continue;

        const inputs = chunk.slice(0, -1);
        const targets = chunk.slice(1);

        trainExample(model, optimizer, inputs, targets);

        trainingSteps++;
    }

    await saveRecord("training", {
        question,
        answer,
        created: Date.now()
    });

    await saveModel();

    exampleCount++;
    updateStats();
}

async function restoreTraining() {
    const examples = await getRecords("training");

    exampleCount = examples.length;

    updateStats();
}

async function sendMessage() {
    const text = input.value.trim();

    if (!text || busy) return;

    input.value = "";
    addMessage("user", text);

    busy = true;
    sendButton.disabled = true;

    try {
        // Teaching command
        if (text.startsWith("!teach")) {
            const content = text.slice(6).trim();
            const parts = content.split("\n---\n");

            if (parts.length !== 2) {
                addMessage(
                    "assistant",
                    "Teaching format:\n!teach\nYour question\n---\nThe correct answer"
                );
            } else {
                const [question, answer] = parts;

                await teach(question.trim(), answer.trim());

                addMessage(
                    "assistant",
                    "Thank you! I've added your example to my training data and updated my model weights."
                );
            }

            return;
        }

        const prompt = `User: ${text}\nTiny:`;

        const response = generate(
            model,
            tokenizer,
            prompt,
            120
        );

        addMessage(
            "assistant",
            response.trim() || "I don't have a response yet."
        );

        await saveRecord("conversations", {
            user: text,
            assistant: response,
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
            "Hello! I'm Tiny. My neural network is ready. I'm still learning, so my answers may not make sense yet. You can teach me using !teach."
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

sendButton.addEventListener("click", sendMessage);

input.addEventListener("keydown", event => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
});

$("new-chat").addEventListener("click", () => {
    messages.replaceChildren();
    welcome.style.display = "block";
});

document.querySelectorAll("[data-prompt]").forEach(button => {
    button.addEventListener("click", () => {
        input.value = button.dataset.prompt;
        input.focus();
    });
});

initialise();
