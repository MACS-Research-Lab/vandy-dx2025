import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.preprocessing import StandardScaler
import joblib


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        bottleneck_dim=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        seq_len=128,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer_encoder1 = nn.TransformerEncoder(encoder_layer, num_layers)

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.bottleneck_proj = nn.Linear(d_model, bottleneck_dim)

        self.expand_proj = nn.Linear(bottleneck_dim, d_model)
        self.transformer_encoder2 = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x_embed = self.input_proj(x)
        x_embed = self.pos_encoder(x_embed)

        # Encode
        memory1 = self.transformer_encoder1(x_embed)  # (batch_size, seq_len, d_model)

        # Pooling time series information
        pooled = self.pooling(memory1.transpose(1, 2)).squeeze(
            -1
        )  # (batch_size, d_model)

        # Autoencoder bottleneck
        compressed = self.bottleneck_proj(pooled)  # (batch_size, bottleneck_dim)

        # Decompress (and expand)
        expanded = (
            self.expand_proj(compressed).unsqueeze(1).repeat(1, self.seq_len, 1)
        )  # (batch_size, seq_len, d_model)

        # Decode
        memory2 = self.transformer_encoder2(expanded)  # (batch_size, seq_len, d_model)

        last_element = memory2[:, -1, :]  # (batch_size, d_model)
        out = self.output_proj(last_element)  # (batch_size, input_dim)

        return out


class MLP(nn.Module):
    def __init__(self, input_dim=5, hidden_size=64, output_size=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)

class MLPClassifierFromErrors(nn.Module):
    def __init__(self, model_paths, error_mode="mse", normalizer_path="error_normalizer.pkl"):
        super().__init__()

        # Load 4 pretrained MLP models
        self.models = nn.ModuleList()
        input_dims = [5,5,4,4]
        for path, in_dim in zip(model_paths, input_dims):
            model = MLP(in_dim)
            model.load_state_dict(torch.load(path))
            model.eval()
            self.models.append(model)

        self.error_mode = error_mode
        self.scaler = StandardScaler()
        self.normalizer_path = normalizer_path

        # 2-layer classifier
        self.classifier = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Assuming 4 fault types
        )

        # Try loading existing normalizer
        self._load_normalizer()

    def compute_errors(self, inputs, targets):
        errors = []
        for model, inp, tgt in zip(self.models, inputs, targets):
            with torch.no_grad():
                pred = model(inp)
            if self.error_mode == "mse":
                err = F.mse_loss(pred, tgt, reduction='none').mean(dim=1)
            elif self.error_mode == "abs":
                err = (pred - tgt).abs().mean(dim=1)
            else:
                raise ValueError("Unsupported error_mode")
            errors.append(err.unsqueeze(1))  # shape: [batch, 1]
        return torch.cat(errors, dim=1)  # shape: [batch, 4]

    def fit_normalizer(self, dataloader):
        """Fit sklearn StandardScaler on error vectors and save it."""
        all_errors = []
        for batch in dataloader:
            *inputs_targets, _ = batch
            inputs = inputs_targets[::2]
            targets = inputs_targets[1::2]
            errors = self.compute_errors(inputs, targets)
            all_errors.append(errors.cpu())

        all_errors_np = torch.cat(all_errors, dim=0).numpy()
        self.scaler.fit(all_errors_np)
        joblib.dump(self.scaler, self.normalizer_path)
        print(f"Saved error normalizer to: {self.normalizer_path}")

    def _load_normalizer(self):
        """Try loading normalizer from disk."""
        try:
            self.scaler = joblib.load(self.normalizer_path)
            # print(f"Loaded normalizer from {self.normalizer_path}")
        except FileNotFoundError:
            print(f"No existing normalizer found at {self.normalizer_path}.")

    def normalize_errors(self, error_tensor):
        return torch.tensor(
            self.scaler.transform(error_tensor.cpu().numpy()),
            dtype=torch.float32,
            device=error_tensor.device
        )

    def forward(self, input1, output1, input2, output2, input3, output3, input4, output4):
        inputs = [input1, input2, input3, input4]
        targets = [output1, output2, output3, output4]
        errors = self.compute_errors(inputs, targets)
        norm_errors = self.normalize_errors(errors)
        logits = self.classifier(norm_errors)
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(logits, dim=1)  # Softmax over the 4 classes
        return probabilities