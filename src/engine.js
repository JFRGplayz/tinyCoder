
export class Value {
    constructor(data, children = [], backward = () => {}) {
        this.data = data;
        this.grad = 0;
        this.children = children;
        this._backward = backward;
    }

    add(other) {
        other = val(other);

        const out = new Value(
            this.data + other.data,
            [this, other]
        );

        out._backward = () => {
            this.grad += out.grad;
            other.grad += out.grad;
        };

        return out;
    }

    mul(other) {
        other = val(other);

        const out = new Value(
            this.data * other.data,
            [this, other]
        );

        out._backward = () => {
            this.grad += other.data * out.grad;
            other.grad += this.data * out.grad;
        };

        return out;
    }

    pow(power) {
        const out = new Value(
            this.data ** power,
            [this]
        );

        out._backward = () => {
            this.grad +=
                power *
                this.data ** (power - 1) *
                out.grad;
        };

        return out;
    }

    exp() {
        const out = new Value(
            Math.exp(this.data),
            [this]
        );

        out._backward = () => {
            this.grad += out.data * out.grad;
        };

        return out;
    }

    log() {
        const out = new Value(
            Math.log(Math.max(this.data, 1e-12)),
            [this]
        );

        out._backward = () => {
            this.grad += out.grad / Math.max(this.data, 1e-12);
        };

        return out;
    }

    relu() {
        const out = new Value(
            Math.max(0, this.data),
            [this]
        );

        out._backward = () => {
            if (this.data > 0) {
                this.grad += out.grad;
            }
        };

        return out;
    }

    backward() {
        const order = [];
        const visited = new Set();

        const build = node => {
            if (visited.has(node)) return;

            visited.add(node);

            for (const child of node.children) {
                build(child);
            }

            order.push(node);
        };

        build(this);
        this.grad = 1;

        for (const node of order.reverse()) {
            node._backward();
        }
    }
}

function val(number) {
    return number instanceof Value
        ? number
        : new Value(number);
}

export function add(a, b) {
    return val(a).add(b);
}

export function mul(a, b) {
    return val(a).mul(b);
}

export function sum(values) {
    return values.reduce(
        (result, value) => result.add(value),
        new Value(0)
    );
}

export function softmax(values) {
    const max = Math.max(...values.map(v => v.data));

    const exps = values.map(v =>
        v.add(-max).exp()
    );

    const total = sum(exps);

    return exps.map(v =>
        v.mul(total.pow(-1))
    );
}

export class Linear {
    constructor(input, output) {
        this.weights = Array.from(
            { length: output },
            () => Array.from(
                { length: input },
                () => new Value(
                    (Math.random() - 0.5) * 0.2
                )
            )
        );

        this.biases = Array.from(
            { length: output },
            () => new Value(0)
        );
    }

    forward(input) {
        return this.weights.map((row, i) =>
            sum(
                row.map((weight, j) =>
                    weight.mul(input[j])
                )
            ).add(this.biases[i])
        );
    }

    parameters() {
        return [
            ...this.weights.flat(),
            ...this.biases
        ];
    }
}
