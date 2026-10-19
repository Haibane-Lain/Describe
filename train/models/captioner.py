"""Captioner: frozen CNN features + trained LSTM decoder.

The encoder runs once during feature extraction, so this module only ever sees
a ``[batch, feature_dim]`` tensor. That is what makes training cheap enough to
run comfortably on a single GPU - or on CPU if the GPU is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BeamCandidate:
    tokens: list[int]
    logprob: float

    @property
    def normalized_score(self) -> float:
        return self.logprob


class Captioner(nn.Module):
    """Feature projection -> embedding -> LSTM -> vocabulary logits."""

    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = 2048,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        self.feature_proj = nn.Linear(feature_dim, embed_dim)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.feature_proj.weight)
        nn.init.zeros_(self.feature_proj.bias)
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embed.weight[self.pad_idx].fill_(0.0)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    # ------------------------------------------------------------------ forward
    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
            features: ``[batch, feature_dim]`` cached encoder output.
            captions: ``[batch, seq_len]`` token ids including ``<start>``.

        Returns:
            ``[batch, seq_len, vocab_size]`` logits.
        """
        batch_size, seq_len = captions.shape
        projected = torch.tanh(self.feature_proj(features))  # [B, E]
        embedded = self.dropout(self.embed(captions))        # [B, T, E]

        # Prepend the image vector as the first timestep.
        inputs = torch.cat([projected.unsqueeze(1), embedded[:, :-1, :]], dim=1)

        hidden = self._init_hidden(projected, batch_size)
        outputs, _ = self.lstm(inputs, hidden)
        return self.output(self.dropout(outputs))

    def forward_inputs(
        self, projected: torch.Tensor, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Run a pre-built input sequence through the decoder.

        ``projected`` is the tanh-projected image vector ``[B, E]`` and
        ``inputs`` is ``[B, T, E]``. Shared by teacher forcing and scheduled
        sampling so both use identical decoder mechanics.
        """
        hidden = self._init_hidden(projected, projected.shape[0])
        outputs, _ = self.lstm(inputs, hidden)
        return self.output(self.dropout(outputs))

    def scheduled_sampling_loss(
        self,
        features: torch.Tensor,
        captions: torch.Tensor,
        sampling_ratio: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Cross-entropy while feeding the model its own predictions.

        Teacher forcing scores the model on ground-truth prefixes it never sees
        at inference, so it never learns to recover from its own mistakes. That
        exposure bias is what makes an undertrained decoder loop on a single
        token. Here, at each step, a random ``sampling_ratio`` fraction of the
        batch is fed its own previous prediction instead of the ground truth.

        The teacher-forced prefix is still built in parallel; the model's
        predictions overwrite it at the sampled positions. Because the whole
        sequence is then run in a single LSTM call, the chosen tokens are what
        the LSTM actually consumes, so the gradient sees the true input.

        Args:
            sampling_ratio: 0.0 is pure teacher forcing, 1.0 is fully
                free-running. Use 0.0 for the first epoch, then anneal upward.
        """
        batch_size, seq_len = captions.shape
        projected = torch.tanh(self.feature_proj(features))

        if sampling_ratio <= 0.0:
            return self.loss(features, captions)

        embedded = self.dropout(self.embed(captions))
        # Position 0 is the image vector; positions 1.. are <start> then tokens.
        inputs = torch.cat([projected.unsqueeze(1), embedded[:, :-1, :]], dim=1)

        if generator is None:
            generator = torch.Generator(device=features.device).manual_seed(0)

        # One teacher-forced pass supplies every position's prediction. A
        # per-step loop would be more exact, but it costs a forward pass per
        # token (roughly 8x slower) for a second-order correction, so the
        # single-pass approximation is used deliberately.
        teacher_logits = self.forward_inputs(projected, inputs)
        predictions = teacher_logits.argmax(dim=-1)  # [B, T]

        # Decide which positions to replace with the model's own prediction.
        replace = (
            torch.rand(batch_size, seq_len, device=features.device, generator=generator)
            < sampling_ratio
        )
        # Never replace padding, the image slot, or the <start> position.
        replace[:, 0] = False
        if seq_len > 1:
            replace[:, 1] = False
        replace &= captions != self.pad_idx

        if bool(replace.any()):
            replacement = self.embed(predictions)
            inputs = torch.where(replace.unsqueeze(-1), replacement, inputs)

        logits = self.forward_inputs(projected, inputs)
        targets = captions[:, 1:]
        # `inputs` has one timestep more than `targets` (the image vector
        # precedes the first predicted token), so trim to align them.
        logits = logits[:, : targets.shape[1], :]
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=self.pad_idx,
        )

    def _init_hidden(
        self, projected: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialise (h, c) from the image vector, zero-padded to num_layers.

        Seeding the top layer from the image gives the decoder immediate access
        to visual context instead of relying on the first timestep alone.
        """
        hidden = projected.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        cell = projected.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        hidden[-1] = projected
        return hidden.contiguous(), cell.contiguous()

    # -------------------------------------------------------------------- loss
    def loss(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over the target tokens, ignoring padding.

        ``captions`` is the full ``[<start>, ..., <end>]`` sequence; the model
        predicts token ``t`` from tokens ``< t``.
        """
        logits = self.forward(features, captions[:, :-1])
        targets = captions[:, 1:]
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=self.pad_idx,
        )

    # ------------------------------------------------------------------ decode
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
        """Greedy decoding. Returns token ids without ``<start>``/``<end>``."""
        self.eval()
        batch_size = features.shape[0]
        projected = torch.tanh(self.feature_proj(features))
        hidden = self._init_hidden(projected, batch_size)

        step_input = projected.unsqueeze(1)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=features.device)
        results: list[list[int]] = [[] for _ in range(batch_size)]

        for _ in range(max_len):
            output, hidden = self.lstm(step_input, hidden)
            logits = self.output(output[:, -1, :])
            # Mask <end> until the minimum length is reached, matching beam
            # search. Without this, a min_len above the natural caption length
            # makes the decoder loop on <end>.
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

            # Feed each beam the token it actually produced. A finished
            # sequence must be fed <end>, not its last emitted token, or the
            # LSTM is conditioned on a token the model never chose to emit.
            step_tokens = next_token.clone()
            if bool(finished.any()):
                step_tokens[finished] = end_idx
            step_input = self.embed(step_tokens).unsqueeze(1)

        return results

    @torch.no_grad()
    def beam_decode(
        self,
        features: torch.Tensor,
        start_idx: int,
        end_idx: int,
        beam_size: int = 5,
        max_len: int = 30,
        min_len: int = 5,
        repetition_penalty: float = 1.0,
        no_repeat_ngram: int = 0,
    ) -> list[list[int]]:
        """Batched beam search decoding.

        Every image keeps ``beam_size`` independent hypotheses, each with its own
        LSTM state. Hypotheses are reordered in step with the hidden state when
        beams are pruned, so each beam continues from its own history.

        Scores are length-normalised, which stops the search from preferring
        very short captions.

        Args:
            repetition_penalty: values > 1 suppress tokens already in the
                hypothesis. A small penalty (1.1-1.3) is the standard remedy for
                the degenerate loops an undertrained decoder falls into.
            no_repeat_ngram: forbid repeating an n-gram of this size (0 = off).
                ``3`` blocks "a a a" and similar.
        """
        self.eval()
        batch_size = features.shape[0]
        if beam_size <= 1:
            return self.greedy_decode(
                features, start_idx, end_idx, max_len=max_len, min_len=min_len,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
            )

        device = features.device
        projected = torch.tanh(self.feature_proj(features))          # [B, E]
        hidden, cell = self._init_hidden(projected, batch_size)
        # Expand to one LSTM row per beam up front, so every step's top-k indices
        # refer to a fixed [L, B*beam_size, H] layout. Rows are ordered
        # (batch * beam_size + beam); all beams of an image start from the same
        # image-derived state.
        hidden = hidden.reshape(
            self.num_layers, batch_size, self.hidden_dim
        ).repeat_interleave(beam_size, dim=1).contiguous()
        cell = cell.reshape(
            self.num_layers, batch_size, self.hidden_dim
        ).repeat_interleave(beam_size, dim=1).contiguous()

        # Each image starts with beam_size identical live beams.
        tokens = torch.full(
            (batch_size, beam_size, 0), 0, dtype=torch.long, device=device
        )
        scores = torch.zeros(batch_size, beam_size, device=device)
        finished: list[list[tuple[list[int], float]]] = [[] for _ in range(batch_size)]

        step_input = projected.repeat_interleave(beam_size, dim=0).unsqueeze(1)

        for _ in range(max_len):
            output, _ = self.lstm(step_input, (hidden, cell))
            log_probs = F.log_softmax(self.output(output[:, -1, :]), dim=-1)

            # Mask <end> until the minimum length is reached. Without this the
            # search is forced to emit <end> repeatedly to pad itself up to
            # min_len, producing degenerate all-<end> captions.
            if tokens.shape[-1] < min_len:
                log_probs = log_probs.clone()
                log_probs[:, end_idx] = float("-inf")

            # Break repetition loops. `tokens` is [B, beams, T] and `log_probs`
            # is [(B*beams), V], so both are flattened to the same row order.
            # T is 0 on the first step, where there is nothing to constrain.
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
            # log_probs is [B*beams, V] and scores is [B, beams]; flatten scores
            # to the same row layout so the beam-token scores add elementwise.
            flat = (scores.reshape(-1, 1) + log_probs).view(batch_size, -1)
            top_scores, top_indices = torch.topk(
                flat, min(beam_size, flat.shape[1]), dim=-1
            )
            # Indices are into the flattened [beams * V] row of one image.
            beam_ids = top_indices // vocab_size
            token_ids = top_indices % vocab_size
            new_beams = beam_ids.shape[1]

            # Reorder state so each surviving beam continues from its own history.
            # The previous layout is (batch * beam_size + beam), and prev_beams
            # equals the current beam count, so the flat index is
            # prev_beam_id + batch * prev_beams.
            prev_beams = scores.shape[1]
            batch_offsets = (
                torch.arange(batch_size, device=device) * prev_beams
            ).unsqueeze(1)
            flat_ids = (beam_ids + batch_offsets).reshape(-1)
            hidden = torch.index_select(hidden, 1, flat_ids).contiguous()
            cell = torch.index_select(cell, 1, flat_ids).contiguous()

            tokens = torch.gather(
                tokens, 1, beam_ids.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            )
            tokens = torch.cat([tokens, token_ids.unsqueeze(-1)], dim=-1)
            scores = top_scores

            # Retire beams that emitted <end> after the minimum length.
            for b in range(batch_size):
                for k in range(tokens.shape[1]):
                    if (
                        int(token_ids[b, k]) == end_idx
                        and tokens.shape[-1] - 1 >= min_len
                    ):
                        sequence = tokens[b, k, :-1].tolist()
                        finished[b].append(
                            (sequence, float(scores[b, k]) / max(len(sequence), 1))
                        )

            # Retire finished beams so they cannot be selected again. Without
            # this a completed beam keeps generating <end> and pads its own
            # sequence out to max_len, which leaks <end> into the output.
            retired = token_ids == end_idx
            if bool(retired.any()):
                if bool(retired.all()):
                    break
                scores = scores.masked_fill(retired, float("-inf"))

            step_input = self.embed(token_ids).view(
                batch_size * new_beams, 1, self.embed.embedding_dim
            )

        # Pick the best finished hypothesis, falling back to the live beams.
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


def apply_decoding_constraints(
    logits: torch.Tensor,
    sequences: list[list[int]],
    repetition_penalty: float = 1.0,
    no_repeat_ngram: int = 0,
) -> torch.Tensor:
    """Apply repetition and n-gram constraints to a batch of logits.

    An undertrained captioner falls into loops such as "a a a a ..." because the
    highest-probability continuation of a repeated token is that same token.
    These two constraints break the loop without touching the model:

    * ``repetition_penalty`` divides the logit of every already-emitted token
      (positive logits) or multiplies it (negative logits), which is the
      convention used by HuggingFace's ``repetition_penalty``.
    * ``no_repeat_ngram`` sets the logit to ``-inf`` for any token that would
      complete an n-gram already present in that sequence.

    Args:
        logits: ``[batch, vocab]`` next-token scores.
        sequences: per-row tokens emitted so far.
        repetition_penalty: 1.0 disables the penalty.
        no_repeat_ngram: 0 disables the n-gram block.
    """
    if repetition_penalty == 1.0 and no_repeat_ngram <= 0:
        return logits

    logits = logits.clone()

    if repetition_penalty != 1.0:
        for row, tokens in enumerate(sequences):
            if not tokens:
                continue
            unique = torch.tensor(
                sorted(set(tokens)), device=logits.device, dtype=torch.long
            )
            values = logits[row, unique]
            logits[row, unique] = torch.where(
                values > 0, values / repetition_penalty, values * repetition_penalty
            )

    if no_repeat_ngram > 0:
        n = no_repeat_ngram
        for row, tokens in enumerate(sequences):
            if len(tokens) < n:
                continue
            # Ban any token whose n-gram already occurred anywhere in this
            # sequence. Scanning every position (not just one prefix) is what
            # actually stops "a a a a"; checking a single prefix leaves shorter
            # cycles such as "a a the a a two" free to repeat.
            seen: set[tuple[int, ...]] = set()
            for start in range(len(tokens) - n + 1):
                seen.add(tuple(tokens[start : start + n]))
            prefix = tuple(tokens[-(n - 1) :])
            banned = {gram[-1] for gram in seen if gram[:-1] == prefix}
            if banned:
                logits[row, sorted(banned)] = float("-inf")

    return logits


def collate_captions(
    sequences: list[list[int]], pad_idx: int
) -> torch.Tensor:
    """Right-pad a batch of token id sequences to a rectangular tensor."""
    max_len = max(len(s) for s in sequences)
    out = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
    for row, seq in enumerate(sequences):
        out[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out
