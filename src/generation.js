
import { softmax } from "./engine.js";

export function generate(model, tokenizer, prompt, maxTokens = 120) {
    const tokens = tokenizer.encode(prompt);

    if (!tokens.length) {
        return "";
    }

    const generated = [];

    for (let i = 0; i < maxTokens; i++) {
        const context = tokens.slice(-model.contextSize);

        const logits = model.forward(context);
        const lastLogits = logits[logits.length - 1];

        const probabilities = softmax(lastLogits)
            .map(v => Math.max(0, v.data));

        // Temperature controls randomness.
        const temperature = 0.8;

        const adjusted = probabilities.map(p =>
            Math.pow(Math.max(p, 1e-12), 1 / temperature)
        );

        const total = adjusted.reduce((a, b) => a + b, 0);

        let random = Math.random() * total;
        let nextToken = 0;

        for (let j = 0; j < adjusted.length; j++) {
            random -= adjusted[j];

            if (random <= 0) {
                nextToken = j;
                break;
            }
        }

        tokens.push(nextToken);
        generated.push(nextToken);

        const character = tokenizer.decode([nextToken]);

        if (character === "\n") break;
    }

    return tokenizer.decode(generated);
}
