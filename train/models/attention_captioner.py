"""Captioner with spatial attention over CNN feature maps.

The pooled-vector decoder could only see a single 2048-dim summary of the whole
image, which is why it produced generic high-frequency text. This model attends
over the ``7x7`` spatial grid at every decoding step, so it can select the
region that names the object it is about to describe.

Architecture: frozen ResNet-101 spatial map ``[B, 2048, 7, 7]`` -> Bahdanau
(additive) attention over the 49 locations -> LSTM decoder with the context
vector concatenated to each token embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive attention over a set of spatial locations."""

    def __init__(self, feature_dim: int, hidden_dim: int, attn_dim: int = 512) -> None:
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, attn_dim, bias=False)
        self.hidden_proj = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self, features: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (context, attention weights).

        Args:
            features: ``[B, L, C]`` flattened spatial locations.
            hidden: ``[B, H]`` decoder hidden state used as the query.
        """
        query = self.hidden_proj(hidden).unsqueeze(1)          # [B, 1, A]
        keys = self.feature_proj(features)                     # [B, L, A]
        energies = self.score(torch.tanh(query + keys)).squeeze(-1)  # [B, L]
        weights = F.softmax(energies, dim=-1)                  # [B, L]
        context = torch.bmm(weights.unsqueeze(1), features).squeeze(1)  # [B, C]
        return context, weights


@dataclass
class BeamCandidate:
    tokens: list[int]
    logprob: float


class AttentionCaptioner(nn.Module):
    """Attention-based caption decoder over cached spatial feature maps."""

    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = 2048,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        attn_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        self.feature_dim = feature_dim

        self.attention = BahdanauAttention(feature_dim, hidden_dim, attn_dim)
        # Input is the token embedding plus the attention context.
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.context_proj = nn.Linear(feature_dim, embed_dim, bias=False)
        # Seeds the LSTM's initial state directly from the mean image feature,
        # so the first attention query is image-aware.
        self.image_seed = nn.Linear(feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim + feature_dim, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embed.weight[self.pad_idx].fill_(0.0)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def flatten_features(features: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` -> ``[B, H*W, C]``."""
        if features.dim() == 4:
            return features.flatten(2).transpose(1, 2)
        return features

    def _init_hidden(
        self, features: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Seed the decoder state from the mean of the spatial features.

        Seeding from the image means the very first attention query already
        carries visual signal. When the decoder starts from zeros instead, the
        first queries are pure language prior and attention selects regions
        based on what words are expected rather than what is in the picture.
        """
        mean_feature = features.mean(dim=1)                     # [B, C]
        seed = torch.tanh(self.image_seed(mean_feature))        # [B, H]
        hidden = seed.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        cell = seed.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        hidden[-1] = seed
        return hidden.contiguous(), cell.contiguous()

    def step(
        self,
        token: torch.Tensor,
        features: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One decoding step.

        Returns (logits, hidden, cell, attention_weights).
        """
        query = hidden[-1]                                      # [B, H]
        context, weights = self.attention(features, query)      # [B, C], [B, L]
        embedded = self.dropout(self.embed(token))              # [B, E]
        context_embedded = self.context_proj(context)           # [B, E]
        lstm_input = torch.cat([embedded, context_embedded], dim=-1).unsqueeze(1)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        logits = self.output(torch.cat([output[:, -1, :], context], dim=-1))
        return logits, hidden, cell, weights

    def forward(
        self, features: torch.Tensor, captions: torch.Tensor
    ) -> torch.Tensor:
        """Teacher-forced pass, batched over time with image-aware attention.

        The decoder is image-conditioned from the very first step:

        * the LSTM state is seeded from the mean image feature, and
        * step 0 consumes that same mean feature instead of a token, so the
          hidden state used as the first attention query already encodes the
          image.

        Attention context does not feed back into the LSTM, which keeps the
        whole sequence computable in a single LSTM call rather than one call per
        token (a per-token loop measured ~45 minutes per epoch).
        """
        features = self.flatten_features(features)              # [B, L, C]
        batch_size, seq_len = captions.shape

        hidden, cell = self._init_hidden(features, batch_size)

        mean_feature = features.mean(dim=1)                      # [B, C]
        seed = self.context_proj(mean_feature).unsqueeze(1)      # [B, 1, E]
        embedded = self.dropout(self.embed(captions))            # [B, T, E]
        # Step 0 takes the image seed; step t takes the token at t-1.
        tokens_in = torch.cat([seed, embedded[:, :-1, :]], dim=1)  # [B, T, E]

        # Single causal pass. The context half is zero here; attention context
        # is added afterwards and does not feed back, so this is exact.
        zeros = torch.zeros_like(tokens_in)
        outputs, _ = self.lstm(
            torch.cat([tokens_in, zeros], dim=-1), (hidden, cell)
        )                                                        # [B, T, H]

        # Query for step t is the hidden state from step t-1, which for t=0 is
        # the image-seeded initial state.
        queries = torch.cat(
            [hidden[-1].unsqueeze(1), outputs[:, :-1, :]], dim=1
        )                                                        # [B, T, H]

        # Batched additive attention: [B, T, H] x [B, L, C] -> [B, T, L]
        keys = self.attention.feature_proj(features)             # [B, L, A]
        query = self.attention.hidden_proj(queries)              # [B, T, A]
        energies = self.attention.score(
            torch.tanh(query.unsqueeze(2) + keys.unsqueeze(1))
        ).squeeze(-1)                                            # [B, T, L]
        weights = F.softmax(energies, dim=-1)
        context = torch.bmm(weights, features)                   # [B, T, C]

        logits = self.output(
            torch.cat([outputs, context], dim=-1)
        )                                                        # [B, T, V]
        # The LSTM consumes one more position than there are targets (the image
        # seed precedes the first predicted token), so trim to align.
        return logits[:, : seq_len - 1, :]

    def loss(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        logits = self.forward(features, captions)
        targets = captions[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=self.pad_idx,
        )

    def scheduled_sampling_loss(
        self,
        features: torch.Tensor,
        captions: torch.Tensor,
        sampling_ratio: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Cross-entropy while feeding the model its own predictions.

        Implemented as one batched pass, matching the teacher-forced forward:
        the teacher-forced pass supplies a prediction for every position, and a
        random ``sampling_ratio`` fraction of positions is then overwritten with
        that prediction before a second pass. A per-step loop would be exact but
        costs a forward pass per token, which made an epoch ~45 minutes.
        """
        features = self.flatten_features(features)
        batch_size, seq_len = captions.shape

        if sampling_ratio <= 0.0:
            return self.loss(features, captions)

        if generator is None:
            generator = torch.Generator(device=features.device).manual_seed(0)

        teacher_logits = self.forward(features, captions)
        # Never substitute <pad>: feeding the pad token back as if it were a
        # prediction teaches the model to predict pad, which collapses it to a
        # constant output within a few hundred steps. Mask pad and take the best
        # legal token instead.
        safe_logits = teacher_logits.clone()
        safe_logits[..., self.pad_idx] = float("-inf")
        prediction = safe_logits.argmax(dim=-1)

        # forward returns T-1 positions; pad one slot so it aligns with the
        # full caption length, then shift by one when substituting.
        predictions = torch.cat(
            [
                torch.full(
                    (teacher_logits.shape[0], 1), self.pad_idx,
                    dtype=torch.long, device=teacher_logits.device,
                ),
                prediction,
            ],
            dim=1,
        )                                                        # [B, T]

        sampled = captions.clone()
        replace = (
            torch.rand(batch_size, seq_len, device=features.device, generator=generator)
            < sampling_ratio
        )
        replace[:, 0] = False          # never overwrite <start>
        replace &= captions != self.pad_idx
        # A substituted token must itself be a real token.
        replace &= predictions != self.pad_idx
        # Position t is replaced with the model's prediction from step t-1.
        sampled = torch.where(replace, predictions, sampled)

        logits = self.forward(features, sampled)
        targets = sampled[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=self.pad_idx,
        )

    @torch.no_grad()
    def beam_decode(
        self,
        features: torch.Tensor,
        start_idx: int,
        end_idx: int,
        beam_size: int = 5,
        max_len: int = 30,
        min_len: int = 2,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
        length_alpha: float = 0.7,
    ) -> list[list[int]]:
        """Batched beam search with per-beam LSTM state and attention context.

        ``length_alpha`` controls GNMT-style length normalization (0 disables
        it). It is applied while ranking live beams as well as finished ones.
        """
        from train.models.captioner import apply_decoding_constraints

        self.eval()
        features = self.flatten_features(features)
        batch_size = features.shape[0]
        if beam_size <= 1:
            return self.greedy_decode(
                features, start_idx, end_idx, max_len=max_len, min_len=min_len,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
            )

        device = features.device

        def length_penalty(length: int) -> float:
            """GNMT-style length normalization.

            Without this, accumulated log-probabilities are monotonically
            decreasing in length, so the search systematically prefers short
            captions — which is exactly what depresses BLEU-4. Normalizing only
            the finished beams (as this used to) leaves the search itself
            length-biased.
            """
            return ((5.0 + length) / 6.0) ** length_alpha

        hidden, cell = self._init_hidden(features, batch_size)
        # Expand to one row per beam.
        hidden = hidden.repeat_interleave(beam_size, dim=1).contiguous()
        cell = cell.repeat_interleave(beam_size, dim=1).contiguous()
        expanded_features = features.repeat_interleave(beam_size, dim=0)

        tokens = torch.full(
            (batch_size, beam_size, 0), 0, dtype=torch.long, device=device
        )
        scores = torch.zeros(batch_size, beam_size, device=device)
        finished: list[list[tuple[list[int], float]]] = [[] for _ in range(batch_size)]

        step_token = torch.full(
            (batch_size * beam_size,), start_idx, dtype=torch.long, device=device
        )

        for _ in range(max_len):
            logits, hidden, cell, _ = self.step(
                step_token, expanded_features, hidden, cell
            )
            log_probs = F.log_softmax(logits, dim=-1)

            if tokens.shape[-1] < min_len:
                log_probs = log_probs.clone()
                log_probs[:, end_idx] = float("-inf")

            if (
                (repetition_penalty != 1.0 or no_repeat_ngram > 0)
                and tokens.shape[-1] > 0
            ):
                flat_sequences = [
                    [t for t in row if t != end_idx]
                    for row in tokens.reshape(-1, tokens.shape[-1]).tolist()
                ]
                log_probs = apply_decoding_constraints(
                    log_probs, flat_sequences,
                    repetition_penalty, no_repeat_ngram,
                )

            vocab_size = log_probs.shape[-1]
            # Rank by length-normalized score so the search does not drift
            # toward short captions, but keep the raw cumulative log-probability
            # in `scores` so the penalty is never applied twice.
            current_len = tokens.shape[-1]
            normalization = length_penalty(current_len + 1)
            normalized = scores / normalization
            flat = (normalized.reshape(-1, 1) + log_probs).view(batch_size, -1)
            top_scores, top_indices = torch.topk(
                flat, min(beam_size, flat.shape[1]), dim=-1
            )
            beam_ids = top_indices // vocab_size
            token_ids = top_indices % vocab_size
            new_beams = beam_ids.shape[1]

            prev_beams = scores.shape[1]
            batch_offsets = (
                torch.arange(batch_size, device=device) * prev_beams
            ).unsqueeze(1)
            flat_ids = (beam_ids + batch_offsets).reshape(-1)
            hidden = torch.index_select(hidden, 1, flat_ids).contiguous()
            cell = torch.index_select(cell, 1, flat_ids).contiguous()
            expanded_features = torch.index_select(
                expanded_features, 0, flat_ids
            ).contiguous()

            tokens = torch.gather(
                tokens, 1, beam_ids.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            )
            tokens = torch.cat([tokens, token_ids.unsqueeze(-1)], dim=-1)
            # Recover the raw cumulative score from the normalized ranking.
            scores = (top_scores - log_probs.gather(
                1, token_ids.reshape(-1, 1)
            ).reshape(batch_size, -1)) * normalization

            for b in range(batch_size):
                for k in range(tokens.shape[1]):
                    if (
                        int(token_ids[b, k]) == end_idx
                        and tokens.shape[-1] - 1 >= min_len
                    ):
                        sequence = tokens[b, k, :-1].tolist()
                        finished[b].append(
                            (
                                sequence,
                                float(scores[b, k])
                                / length_penalty(len(sequence)),
                            )
                        )

            retired = token_ids == end_idx
            if bool(retired.any()):
                if bool(retired.all()):
                    break
                scores = scores.masked_fill(retired, float("-inf"))

            step_token = token_ids.reshape(-1)

        results: list[list[int]] = []
        for b in range(batch_size):
            if finished[b]:
                best = max(finished[b], key=lambda item: item[1])
                results.append(best[0])
                continue
            best_index = int(torch.argmax(scores[b]).item())
            sequence = tokens[b, best_index].tolist()
            results.append([t for t in sequence if t != end_idx])
        return results

    @torch.no_grad()
    def greedy_decode(
        self,
        features: torch.Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int = 30,
        min_len: int = 0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
    ) -> list[list[int]]:
        """Greedy decoding."""
        from train.models.captioner import apply_decoding_constraints

        self.eval()
        features = self.flatten_features(features)
        batch_size = features.shape[0]
        hidden, cell = self._init_hidden(features, batch_size)

        step_token = torch.full(
            (batch_size,), start_idx, dtype=torch.long, device=features.device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=features.device)
        results: list[list[int]] = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            logits, hidden, cell, _ = self.step(step_token, features, hidden, cell)
            if min_len > 0:
                shortest = min(len(r) for r in results)
                if shortest < min_len:
                    logits = logits.clone()
                    logits[:, end_idx] = float("-inf")
            logits = apply_decoding_constraints(
                logits, results, repetition_penalty, no_repeat_ngram
            )
            next_token = logits.argmax(dim=-1)

            for i in range(batch_size):
                if finished[i]:
                    continue
                token = int(next_token[i].item())
                if token == end_idx and len(results[i]) >= min_len:
                    finished[i] = True
                elif token != end_idx:
                    results[i].append(token)

            if bool(finished.all()):
                break
            step_token = next_token.clone()
            if bool(finished.any()):
                step_token[finished] = end_idx

        return results
