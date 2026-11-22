
import { Value, softmax } from "./engine.js";

export class Adam {
    constructor(parameters, learningRate = 0.001) {
        this.parameters = parameters;
        this.lr = learningRate;

        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.epsilon = 1e-8;

        this.m = new Float64Array(parameters.length);
        this.v = new Float64Array(parameters.length);

        this.steps = 0;
    }

    zeroGrad() {
        for (const parameter of this.parameters) {
            parameter.grad = 0;
        }
    }

    step() {
        this.steps++;

        for (let i = 0; i < this.parameters.length; i++) {
            const p = this.parameters[i];
            const g = Math.max(-1, Math.min(1, p.grad));

            this.m[i] =
                this.beta1 * this.m[i] +
                (1 - this.beta1) * g;

            this.v[i] =
                this.beta2 * this.v[i] +
                (1 - this.beta2) * g * g;

            const mHat =
                this.m[i] /
                (1 - Math.pow(this.beta1, this.steps));

            const vHat =
                this.v[i] /
                (1 - Math.pow(this.beta2, this.steps));

            p.data -= this.lr * mHat /
                (Math.sqrt(vHat) + this.epsilon);
        }
    }
}

export function calculateLoss(logits, targets) {
    const losses = [];

    for (let i = 0; i < targets.length; i++) {
        const probabilities = softmax(logits[i]);

        losses.push(
            probabilities[targets[i]].log().mul(-1)
        );
    }

    return losses.reduce(
        (a, b) => a.add(b),
        new Value(0)
    ).mul(1 / losses.length);
}

export function trainExample(model, optimizer, input, target) {
    if (!input.length || input.length !== target.length) {
        throw new Error("Invalid training sequence.");
    }

    optimizer.zeroGrad();

    const logits = model.forward(input);
    const loss = calculateLoss(logits, target);

    loss.backward();
    optimizer.step();

    return loss.data;
}

/**
 * Trains Tiny to respond to a specific prompt.
 *
 * The prompt is context, not something Tiny is asked
 * to reproduce. Training only happens on answer tokens.
 */
export function trainChatExample(
    model,
    optimizer,
    tokenizer,
    question,
    answer
) {
    const prompt = `User: ${question}\nTiny:`;
    const promptTokens = tokenizer.encode(prompt);
    const answerTokens = tokenizer.encode(answer + "\n");

    if (!promptTokens.length || !answerTokens.length) {
        throw new Error("Empty training example.");
    }

    let totalLoss = 0;

    for (let i = 0; i < answerTokens.length; i++) {
        // Context includes the question and previous answer tokens.
        const context = [
            ...promptTokens,
            ...answerTokens.slice(0, i)
        ];

        // Keep only what the model can actually see.
        const contextSize = model.contextSize ?? 32;
        const input = context.slice(-contextSize);

        // Predict the next answer token.
        const logits = model.forward(input);
        const lastLogits = logits[logits.length - 1];

        optimizer.zeroGrad();

        const loss = calculateLoss(
            [lastLogits],
            [answerTokens[i]]
        );

        loss.backward();
        optimizer.step();

        totalLoss += loss.data;
    }

    return totalLoss / answerTokens.length;
}
