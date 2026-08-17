import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.classifier import forward, backward, relu, softmax


def test_relu_zeroes_negatives():
    assert relu(np.array([-1.0, 2.0, -3.0, 4.0])).tolist() == [
        0.0,
        2.0,
        0.0,
        4.0,
    ]


def test_softmax_rows_sum_to_one():
    arr = np.random.randn(3, 3)
    probs = softmax(arr)

    assert np.allclose(probs.sum(axis=1), 1.0)


def test_forward_output_shapes():
    np.random.default_rng(0)
    x = np.random.randn(5, 4)
    w1 = np.random.randn(4, 3)
    b1 = np.zeros(3)
    w2 = np.random.randn(3, 2)
    b2 = np.zeros(2)

    probs, z1, a1, z2 = forward(x, w1, b1, w2, b2)

    assert probs.shape == (5, 2)
    assert z1.shape == (5, 3)
    assert a1.shape == (5, 3)
    assert z2.shape == (5, 2)


def test_backward_matches_autograd():
    import torch

    rng = np.random.default_rng(42)

    x = rng.standard_normal((5, 4))
    w1 = rng.standard_normal((4, 3))
    b1 = rng.standard_normal(3)
    w2 = rng.standard_normal((3, 2))
    b2 = rng.standard_normal(2)
    y = np.array([0, 1, 0, 1, 0])

    probs, z1, a1, z2 = forward(x, w1, b1, w2, b2)

    dw1, db1, dw2, db2 = backward(x, y, z1, a1, z2, probs, w2)

    xt = torch.tensor(x, dtype=torch.float64)
    w1t = torch.tensor(w1, dtype=torch.float64, requires_grad=True)
    b1t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)
    w2t = torch.tensor(w2, dtype=torch.float64, requires_grad=True)
    b2t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)
    yt = torch.tensor(y, dtype=torch.long)

    z1t = xt @ w1t + b1t
    a1t = torch.relu(z1t)
    z2t = a1t @ w2t + b2t
    probst = torch.softmax(z2t, dim=1)
    correct_probst = probst[torch.arange(len(yt)), yt]
    losst = -torch.log(correct_probst).mean()
    losst.backward()

    np.allclose(dw1, w1t.grad.numpy(), atol=1e-9)
