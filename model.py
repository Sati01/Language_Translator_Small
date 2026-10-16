"""Inference model matching translator_notebook(shared_embed_transfomer).ipynb.

Architecture: character-level Transformer with shared character vocabulary
and tied input/output embedding weights. Training is NOT required here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn


class Embedding(nn.Module):
    """Token embedding plus the notebook's fixed sinusoidal positions."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 122,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.input_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=0
        )
        pe = self.pos_encoding(max_len, d_model)
        self.pos_embedding = nn.Embedding.from_pretrained(pe, freeze=True)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def pos_encoding(max_len: int, d_model: int) -> torch.Tensor:
        # Exactly the formula and shape used in the training notebook.
        dividend = torch.arange(max_len).unsqueeze(0).T
        divisor = torch.pow(
            10000.0, torch.arange(0, d_model, 2) / d_model
        )
        angles = dividend / divisor
        pe = torch.zeros((max_len, d_model))
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(dim=-1), device=x.device)
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_embedding(positions)
        return self.dropout(x)


class Translator(nn.Module):
    """Same module names and weight sharing as the original notebook."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        vocab_size: int = 30000,
        max_len: int = 122,
    ) -> None:
        super().__init__()
        self.embedding = Embedding(
            vocab_size, d_model, max_len=max_len
        )
        self.transformer = nn.Transformer(
            d_model,
            nhead,
            num_encoder_layers,
            num_decoder_layers,
            dim_feedforward,
            dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(d_model, vocab_size)
        # The notebook deliberately shares the embedding and output weights.
        self.fc.weight = self.embedding.input_embedding.weight


class TranslationEngine:
    """Load an existing checkpoint and perform Swedish -> English inference."""

    def __init__(
        self,
        checkpoint_path: str | Path = "pico_model.pth",
        vocab_path: str | Path = "pico_translator.vocab",
        device: str | torch.device | None = None,
    ) -> None:
        checkpoint_path = Path(checkpoint_path)
        vocab_path = Path(vocab_path)

        for path in (checkpoint_path, vocab_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {path.name}. Put your trained weights AND "
                    "pico_translator.vocab next to app.py."
                )

        with vocab_path.open("r", encoding="utf-8") as f:
            self.token2idx = json.load(f)
        if not isinstance(self.token2idx, dict):
            raise ValueError("Expected a JSON token-to-ID dictionary in the .vocab file")
        self.idx2token = {value: key for key, value in self.token2idx.items()}

        # Verify the notebook's special symbol indices before inference.
        expected_special = {"<pad>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        for symbol, index in expected_special.items():
            if self.token2idx.get(symbol) != index:
                raise ValueError(
                    f"Vocabulary must map {symbol!r} to {index}. "
                    "Use the vocabulary saved with this exact checkpoint."
                )

        # weights_only=True avoids unpickling arbitrary objects in a checkpoint.
        state = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        if not isinstance(state, dict) or "embedding.input_embedding.weight" not in state:
            raise ValueError(
                "Expected a state_dict from torch.save(translator.state_dict(), ...)."
            )

        vocab_size, d_model = state["embedding.input_embedding.weight"].shape
        max_len = state["embedding.pos_embedding.weight"].shape[0]
        ff_size = state["transformer.encoder.layers.0.linear1.weight"].shape[0]
        encoder_layers = len({
            int(key.split(".")[3])
            for key in state
            if key.startswith("transformer.encoder.layers.")
            and key.endswith(".self_attn.in_proj_weight")
        })
        decoder_layers = len({
            int(key.split(".")[3])
            for key in state
            if key.startswith("transformer.decoder.layers.")
            and key.endswith(".self_attn.in_proj_weight")
        })

        if len(self.token2idx) != vocab_size:
            raise ValueError(
                "Checkpoint and vocabulary disagree on vocabulary size: "
                f"{vocab_size} versus {len(self.token2idx)}."
            )
        if not encoder_layers or not decoder_layers:
            raise ValueError("Could not identify encoder/decoder layers in checkpoint")

        # nhead cannot be inferred reliably from the tensor sizes; notebook = 8.
        self.model = Translator(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=ff_size,
            vocab_size=vocab_size,
            max_len=max_len,
        )
        self.model.load_state_dict(state, strict=True)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device).eval()
        self.max_characters = max_len - 2  # account for <s> and </s>
        self.max_len = max_len

    @torch.inference_mode()
    def translate(self, sentence: str) -> str:
        sentence = sentence.strip()
        if not sentence:
            return ""
        if len(sentence) > self.max_characters:
            raise ValueError(
                f"Input is too long ({len(sentence)} characters). "
                f"Maximum is {self.max_characters}."
            )

        # Mirror notebook's seqs2tensors(): one token per Unicode character.
        symbols = ["<s>", *sentence, "</s>"]
        ids = [self.token2idx.get(symbol, 1) for symbol in symbols]
        src = torch.tensor([ids], dtype=torch.long, device=self.device)

        memory = self.model.transformer.encoder(self.model.embedding(src))
        tgt = torch.tensor(
            [[self.token2idx["<s>"]]], dtype=torch.long, device=self.device
        )

        # Same decoding limit as the notebook: source length + 20, capped by MAX_LEN.
        steps = min(src.size(1) + 20, self.max_len) - 1
        for _ in range(steps):
            target_embeddings = self.model.embedding(tgt)
            target_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.size(1), device=self.device
            )
            decoder_output = self.model.transformer.decoder(
                target_embeddings, memory, tgt_mask=target_mask
            )
            logits = self.model.fc(decoder_output[:, -1, :])
            next_id = logits.argmax(dim=-1, keepdim=True)
            tgt = torch.cat((tgt, next_id), dim=1)
            if next_id.item() == self.token2idx["</s>"]:
                break

        decoded = [self.idx2token.get(idx, "<unk>") for idx in tgt[0, 1:].tolist()]
        if decoded and decoded[-1] == "</s>":
            decoded.pop()
        return "".join(decoded)
