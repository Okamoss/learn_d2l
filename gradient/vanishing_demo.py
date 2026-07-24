from pathlib import Path
import math

import numpy as np


class DeepMLP:
    def __init__(self, input_dim=128, hidden_dim=128, hidden_layers=20, activation="sigmoid", seed=42):
        activation_map = {"sigmoid": "sigmoid", "relu": "relu"}
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.activation = activation_map[activation]
        self.rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        self._init_parameters()

    def _init_parameters(self):
        layer_dims = [self.input_dim] + [self.hidden_dim] * self.hidden_layers + [1]
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            limit = math.sqrt(6.0 / (in_dim + out_dim))
            weight = self.rng.uniform(-limit, limit, size=(in_dim, out_dim))
            bias = np.zeros(out_dim, dtype=np.float64)
            self.weights.append(weight)
            self.biases.append(bias)

    def _activate(self, z):
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-z))
        if self.activation == "relu":
            return np.maximum(0.0, z)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _activation_grad(self, z):
        if self.activation == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-z))
            return s * (1.0 - s)
        if self.activation == "relu":
            return (z > 0.0).astype(np.float64)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, x):
        activations = [x]
        pre_activations = []
        current = x
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            z = current @ weight + bias
            pre_activations.append(z)
            if index < len(self.weights) - 1:
                current = self._activate(z)
            else:
                current = z
            activations.append(current)
        return current, activations, pre_activations

    def backward(self, loss_grad, activations, pre_activations):
        gradient_norms = []
        grad = loss_grad
        for index in reversed(range(len(self.weights))):
            a_prev = activations[index]
            z = pre_activations[index]

            if index < len(self.weights) - 1:
                grad = grad * self._activation_grad(z)

            d_weight = a_prev.T @ grad
            d_bias = grad.sum(axis=0)
            gradient_norms.append(np.linalg.norm(d_weight))

            grad = grad @ self.weights[index].T
            # Keep the parameter update quantities available for debugging or extension.
            _ = d_bias

        gradient_norms.reverse()
        return gradient_norms


def collect_gradient_norms(model, inputs):
    outputs, activations, pre_activations = model.forward(inputs)
    loss = outputs.mean()
    loss_grad = np.ones_like(outputs) / outputs.size
    gradient_norms = model.backward(loss_grad, activations, pre_activations)
    return gradient_norms, float(loss)


def _svg_line(points, color, width=3):
    return '<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{points}" />'.format(
        color=color,
        width=width,
        points=" ".join(f"{x:.2f},{y:.2f}" for x, y in points),
    )


def render_svg(results, output_path):
    width = 1100
    height = 680
    margin_left = 90
    margin_right = 40
    margin_top = 50
    margin_bottom = 90

    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    all_logs = []
    for norms in results.values():
        all_logs.extend(math.log10(max(norm, 1e-30)) for norm in norms)

    min_log = math.floor(min(all_logs)) - 0.5
    max_log = math.ceil(max(all_logs)) + 0.5
    log_span = max(max_log - min_log, 1.0)

    colors = {
        "Sigmoid": "#d62728",
        "ReLU": "#1f77b4",
    }

    def x_pos(index, total):
        if total == 1:
            return margin_left + chart_width / 2
        return margin_left + index * (chart_width / (total - 1))

    def y_pos(value):
        log_value = math.log10(max(value, 1e-30))
        normalized = (log_value - min_log) / log_span
        return margin_top + chart_height - normalized * chart_height

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-size="24" font-family="Arial" fill="#111">Gradient flow in a deep network</text>',
    ]

    x0 = margin_left
    y0 = margin_top + chart_height
    svg_parts.append(f'<line x1="{x0}" y1="{margin_top}" x2="{x0}" y2="{y0}" stroke="#222" stroke-width="2" />')
    svg_parts.append(f'<line x1="{x0}" y1="{y0}" x2="{width - margin_right}" y2="{y0}" stroke="#222" stroke-width="2" />')

    tick_start = int(math.floor(min_log))
    tick_end = int(math.ceil(max_log))
    for tick_power in range(tick_start, tick_end + 1):
        tick_value = 10 ** tick_power
        tick_y = y_pos(tick_value)
        svg_parts.append(
            f'<line x1="{x0 - 6}" y1="{tick_y:.2f}" x2="{x0}" y2="{tick_y:.2f}" stroke="#222" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{x0 - 12}" y="{tick_y + 4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#444">10^{tick_power}</text>'
        )
        svg_parts.append(
            f'<line x1="{x0}" y1="{tick_y:.2f}" x2="{width - margin_right}" y2="{tick_y:.2f}" stroke="#ddd" stroke-width="1" stroke-dasharray="4 4" />'
        )

    total_layers = len(next(iter(results.values())))
    for layer_idx in range(total_layers):
        x = x_pos(layer_idx, total_layers)
        svg_parts.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y0 + 6}" stroke="#222" stroke-width="1" />')
        svg_parts.append(
            f'<text x="{x:.2f}" y="{y0 + 24}" text-anchor="middle" font-size="12" font-family="Arial" fill="#444">{layer_idx + 1}</text>'
        )

    svg_parts.append(
        f'<text x="{width / 2:.1f}" y="{height - 24}" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">Linear layer index</text>'
    )
    svg_parts.append(
        f'<text x="20" y="{height / 2:.1f}" transform="rotate(-90 20 {height / 2:.1f})" text-anchor="middle" font-size="14" font-family="Arial" fill="#333">Gradient norm (log scale)</text>'
    )

    for name, norms in results.items():
        points = []
        for idx, norm in enumerate(norms):
            points.append((x_pos(idx, total_layers), y_pos(norm)))
        svg_parts.append(_svg_line(points, colors[name]))
        for x, y in points:
            svg_parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{colors[name]}" />')

    legend_x = width - margin_right - 220
    legend_y = margin_top + 20
    svg_parts.append(f'<rect x="{legend_x}" y="{legend_y - 18}" width="200" height="70" rx="8" fill="#fafafa" stroke="#ddd" />')
    for offset, (name, color) in enumerate(colors.items()):
        y = legend_y + offset * 24
        svg_parts.append(f'<line x1="{legend_x + 14}" y1="{y}" x2="{legend_x + 44}" y2="{y}" stroke="{color}" stroke-width="3" />')
        svg_parts.append(f'<text x="{legend_x + 54}" y="{y + 4}" font-size="13" font-family="Arial" fill="#222">{name}</text>')

    svg_parts.append("</svg>")
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def run_demo():
    np.random.seed(42)

    batch_size = 64
    input_dim = 128
    hidden_dim = 128
    hidden_layers = 20
    inputs = np.random.randn(batch_size, input_dim)

    models = {
        "Sigmoid": DeepMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation="sigmoid",
            seed=42,
        ),
        "ReLU": DeepMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
            activation="relu",
            seed=42,
        ),
    }

    results = {}
    for name, model in models.items():
        norms, loss = collect_gradient_norms(model, inputs)
        results[name] = norms
        first_norm = norms[0]
        last_norm = norms[-1]
        print(
            f"{name}: loss={loss:.4f}, first-layer grad={first_norm:.3e}, "
            f"last-layer grad={last_norm:.3e}, ratio(last/first)={(last_norm / first_norm):.3e}"
        )

    layer_ids = list(range(1, len(next(iter(results.values()))) + 1))
    print("\nLayer-wise gradient norms:")
    header = "layer".ljust(8) + "".join(name.rjust(18) for name in results)
    print(header)
    for index, layer_id in enumerate(layer_ids):
        row = str(layer_id).ljust(8)
        for name in results:
            row += f"{results[name][index]:18.3e}"
        print(row)

    output_dir = Path("gradient_outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "vanishing_gradient_demo.svg"
    render_svg(results, output_path)
    print(f"\nSaved figure to: {output_path}")


if __name__ == "__main__":
    run_demo()
