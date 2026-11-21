
export class Tokenizer {
    constructor() {
        this.characters = Array.from(
            { length: 95 },
            (_, i) => String.fromCharCode(i + 32)
        );

        this.vocabulary = new Map(
            this.characters.map((char, index) => [char, index])
        );

        this.unknown = 0;
    }

    encode(text) {
        return Array.from(text, char =>
            this.vocabulary.get(char) ?? this.unknown
        );
    }

    decode(tokens) {
        return tokens
            .map(token => this.characters[token] ?? "?")
            .join("");
    }

    get size() {
        return this.characters.length;
    }
}
