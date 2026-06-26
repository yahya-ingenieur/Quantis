"""
src/model.py — Neural forecasting models.

Defines two small, near-identical recurrent forecasters and the Monte Carlo
dropout inference utility:

    LSTMForecaster   primary model
    GRUForecaster    comparison model (same interface -> drop-in swap)

Both take a windowed sequence of features plus a ticker_id and output a single
predicted value (next-day log return). The ticker_id is passed through a learned
embedding (self.ticker_embedding) and concatenated with the RNN's final hidden
state before the output layer, instead of being fed as a raw integer into the
sequence.

Windowing of the raw table into (batch, window, n_features) tensors happens in
train.py, not here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Primary model: LSTM
# --------------------------------------------------------------------------- #

class LSTMForecaster(nn.Module):
    """LSTM forecaster for next-day log return.

    Reads a window of features with an LSTM, takes the final hidden state,
    concatenates a learned ticker embedding, and maps that to a single value.
    """

    def __init__(
        self,
        n_features: int,
        n_tickers: int,
        hidden_size: int = 64,
        embedding_dim: int = 8,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Learned vector per ticker (named attribute so the embeddings can be
        # pulled out and inspected later, e.g. for the report).
        self.ticker_embedding = nn.Embedding(n_tickers, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Head dropout — this is the layer MC dropout re-activates at inference.
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + embedding_dim, 1)

    def forward(self, x_seq: torch.Tensor, ticker_id: torch.Tensor) -> torch.Tensor:
        """Predict next-day log return.

        x_seq:     (batch, window, n_features) float
        ticker_id: (batch,)                    long
        returns:   (batch,)                    float
        """
        # LSTM returns output, (h_n, c_n); we only need the final hidden state.
        _, (h_n, _c_n) = self.lstm(x_seq)
        last_hidden = h_n[-1]                       # (batch, hidden_size)

        ticker_vec = self.ticker_embedding(ticker_id)  # (batch, embedding_dim)

        combined = torch.cat([last_hidden, ticker_vec], dim=1)
        combined = self.dropout(combined)
        out = self.head(combined)                   # (batch, 1)
        return out.squeeze(-1)                       # (batch,)


# --------------------------------------------------------------------------- #
# Comparison model: GRU (same interface as LSTMForecaster)
# --------------------------------------------------------------------------- #

class GRUForecaster(nn.Module):
    """GRU forecaster — identical interface to LSTMForecaster.

    The only real difference from the LSTM version is nn.GRU, whose forward
    returns just h_n (no cell state).
    """

    def __init__(
        self,
        n_features: int,
        n_tickers: int,
        hidden_size: int = 64,
        embedding_dim: int = 8,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.ticker_embedding = nn.Embedding(n_tickers, embedding_dim)

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size + embedding_dim, 1)

    def forward(self, x_seq: torch.Tensor, ticker_id: torch.Tensor) -> torch.Tensor:
        """Same signature and output shape as LSTMForecaster.forward."""
        # GRU returns output, h_n (no cell state).
        _, h_n = self.gru(x_seq)
        last_hidden = h_n[-1]                       # (batch, hidden_size)

        ticker_vec = self.ticker_embedding(ticker_id)  # (batch, embedding_dim)

        combined = torch.cat([last_hidden, ticker_vec], dim=1)
        combined = self.dropout(combined)
        out = self.head(combined)                   # (batch, 1)
        return out.squeeze(-1)                       # (batch,)


# --------------------------------------------------------------------------- #
# Monte Carlo dropout inference
# --------------------------------------------------------------------------- #

def enable_mc_dropout(model: nn.Module) -> None:
    """Put the model in eval mode but re-activate its Dropout layers.

    model.eval() normally turns dropout OFF (it becomes an identity). For Monte
    Carlo dropout we want everything else behaving as at eval, but the dropout
    layers still randomly masking — so we flip just the nn.Dropout modules back
    into training mode.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    x_seq: torch.Tensor,
    ticker_id: torch.Tensor,
    n_passes: int = 50,
    ci: float = 0.95,
    method: str = "std",
):
    """Monte Carlo dropout inference.

    Runs the model n_passes times on the SAME input with dropout left active,
    so each pass uses a different random dropout mask and therefore a slightly
    different sub-network. The spread across passes is the forecast uncertainty.

    Returns a plain tuple (mean, lower, upper, std), each shape (batch,):
        mean   average prediction across passes (the point forecast)
        lower  lower edge of the confidence band
        upper  upper edge of the confidence band
        std    per-sample standard deviation across passes

    method="std":        band = mean +/- z * std, z from the requested ci
    method="percentile": band = empirical (1-ci)/2 and (1+ci)/2 quantiles
    """
    enable_mc_dropout(model)

    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds.append(model(x_seq, ticker_id))
    preds = torch.stack(preds, dim=0)   # (n_passes, batch)

    mean = preds.mean(dim=0)
    std = preds.std(dim=0)

    if method == "std":
        # Two-sided z for the requested central interval (e.g. ci=0.95 -> ~1.96).
        z = _z_score(ci)
        lower = mean - z * std
        upper = mean + z * std
    elif method == "percentile":
        lower_q = (1.0 - ci) / 2.0
        upper_q = 1.0 - lower_q
        lower = torch.quantile(preds, lower_q, dim=0)
        upper = torch.quantile(preds, upper_q, dim=0)
    else:
        raise ValueError(f"Unknown method {method!r}; use 'std' or 'percentile'.")

    return mean, lower, upper, std


def _z_score(ci: float) -> float:
    """Two-sided z-multiplier for a central confidence level.

    Uses the inverse normal CDF. Looks up a couple of common values directly so
    we don't pull in SciPy just for this; falls back to the erfinv form otherwise.
    """
    common = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
    if ci in common:
        return common[ci]
    # z = sqrt(2) * erfinv(ci) for a central two-sided interval.
    return float(torch.sqrt(torch.tensor(2.0)) * torch.erfinv(torch.tensor(ci)))
