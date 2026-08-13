import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from RNN.data_utils import load_data_time_machine
from RNN.model import RNN
from RNN.rnn_utils import grad_clipping, predict_rnn


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a character-level RNN on Time Machine.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=35)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--sampling", choices=["random", "consecutive"], default="random")
    parser.add_argument("--max-tokens", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    use_random_iter = args.sampling == "random"
    train_iter, vocab = load_data_time_machine(
        args.batch_size,
        args.num_steps,
        use_random_iter=use_random_iter,
        max_tokens=args.max_tokens,
    )
    print(f"词表大小: {len(vocab)}")

    net = RNN(len(vocab), num_hiddens=args.hidden_size, num_layers=1).to(device)
    net.apply(init_weights)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    print(f"Start training for {args.epochs} epochs...")
    best_ppl = float("inf")

    for epoch in range(args.epochs):
        net.train()
        total_loss = 0.0
        total_tokens = 0
        state = None

        for X, Y in train_iter:
            X = X.to(device)
            Y = Y.to(device)

            if use_random_iter or state is None:
                state = net.begin_state(batch_size=X.shape[0], device=device)
            else:
                if isinstance(state, tuple):
                    for hidden_state in state:
                        hidden_state.detach_()
                else:
                    state.detach_()

            y_hat, state = net(X, state)
            y = Y.T.reshape(-1)
            loss = loss_fn(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            grad_clipping(net, 1.0)
            optimizer.step()

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

        ppl = math.exp(total_loss / total_tokens)
        best_ppl = min(best_ppl, ppl)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            pred_str = predict_rnn("time traveller ", 20, net, vocab, device)
            print(f"Epoch {epoch + 1}/{args.epochs}, Perplexity: {ppl:.2f}")
            print(f"Prediction: {pred_str}")

    print(f"Training done! Best perplexity: {best_ppl:.2f}")


if __name__ == "__main__":
    main()
