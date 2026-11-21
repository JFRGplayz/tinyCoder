
import {
    Value,
    Linear,
    sum,
    softmax
} from "./engine.js";

export class Transformer {
    constructor(vocabSize = 95, embeddingSize = 8, contextSize = 32) {
        this.vocabSize = vocabSize;
        this.embeddingSize = embeddingSize;
        this.contextSize = contextSize;

        const randomValue = () =>
            new Value((Math.random() - 0.5) * 0.2);

        this.tokens = Array.from(
            { length: vocabSize },
            () => Array.from(
                { length: embeddingSize },
                randomValue
            )
        );

        this.positions = Array.from(
            { length: contextSize },
            () => Array.from(
                { length: embeddingSize },
                randomValue
            )
        );

        this.query = new Linear(embeddingSize, embeddingSize);
        this.key = new Linear(embeddingSize, embeddingSize);
        this.value = new Linear(embeddingSize, embeddingSize);

        this.attentionOutput = new Linear(
            embeddingSize,
            embeddingSize
        );

        this.feedForward1 = new Linear(
            embeddingSize,
            embeddingSize * 2
        );

        this.feedForward2 = new Linear(
            embeddingSize * 2,
            embeddingSize
        );

        this.output = new Linear(
            embeddingSize,
            vocabSize
        );
    }

    forward(tokenIds) {
        const length = tokenIds.length;

        if (length > this.contextSize) {
            throw new Error("Input exceeds context size.");
        }

        // Token and positional embeddings
        const x = tokenIds.map((token, position) =>
            this.tokens[token].map((v, i) =>
                v.add(this.positions[position][i])
            )
        );

        const queries = x.map(v => this.query.forward(v));
        const keys = x.map(v => this.key.forward(v));
        const values = x.map(v => this.value.forward(v));

        // Causal self-attention
        const attended = [];

        for (let i = 0; i < length; i++) {
            const scores = [];

            for (let j = 0; j <= i; j++) {
                scores.push(
                    sum(
                        queries[i].map((q, k) =>
                            q.mul(keys[j][k])
                        )
                    ).mul(1 / Math.sqrt(this.embeddingSize))
                );
            }

            const weights = softmax(scores);

            const context = [];

            for (let k = 0; k < this.embeddingSize; k++) {
                context.push(
                    sum(
                        weights.map((weight, j) =>
                            weight.mul(values[j][k])
                        )
                    )
                );
            }

            const projected =
                this.attentionOutput.forward(context);

            attended.push(
                x[i].map((v, k) =>
                    v.add(projected[k])
                )
            );
        }

        // Feed-forward network
        const hidden = attended.map(vector => {
            const first = this.feedForward1.forward(vector)
                .map(v => v.relu());

            const second = this.feedForward2.forward(first);

            return vector.map((v, i) =>
                v.add(second[i])
            );
        });

        // Vocabulary logits
        return hidden.map(vector =>
            this.output.forward(vector)
        );
    }

    parameters() {
        const params = [
            ...this.tokens.flat(),
            ...this.positions.flat()
        ];

        for (const layer of [
            this.query,
            this.key,
            this.value,
            this.attentionOutput,
            this.feedForward1,
            this.feedForward2,
            this.output
        ]) {
            params.push(...layer.parameters());
        }

        return params;
    }
}
