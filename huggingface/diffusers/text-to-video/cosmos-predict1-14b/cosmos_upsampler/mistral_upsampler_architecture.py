# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from optimum.rbln.transformers.models.decoderonly.decoderonly_architecture import (
    DecoderOnlyAttention,
    DecoderOnlyForCausalLM,
    DecoderOnlyLayer,
    DecoderOnlyModel,
    DecoderOnlyWrapper,
)

from .mistral_upsampler_config import RBLNMistralNeMoForTextUpsamplerConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dimensions of the input tensor."""
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x_reshaped[..., 0]
    x2 = x_reshaped[..., 1]
    output = torch.stack((-x2, x1), dim=-1).reshape(*x.shape)
    return output


def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MistralNeMoAttention(DecoderOnlyAttention):
    def __init__(
        self,
        self_attn,
        rbln_config: "RBLNMistralNeMoForTextUpsamplerConfig",
        is_sliding: bool = False,
    ):
        nn.Module.__init__(self)
        self._original_mod = self_attn
        self.layer_idx = self._original_mod.layer_id
        self.num_key_value_heads = self._original_mod.n_kv_heads
        self.num_heads = self._original_mod.config.n_heads
        self.head_dim = self._original_mod.head_dim
        self.qk_norm = self._original_mod.use_qk_normalization
        self.scale = torch.tensor(self.get_attn_scale())
        self._phase = "prefill"
        self.quantization = rbln_config.quantization
        self.use_attention_mask = rbln_config.use_attention_mask
        self.use_position_ids = rbln_config.use_position_ids
        self.is_sliding = is_sliding
        self.attn_impl = rbln_config.attn_impl

        if self.is_sliding and self.attn_impl != "eager":
            raise NotImplementedError(
                "Sliding window attention is only supported with eager attention."
            )

        self.kvcache_partition_len = rbln_config.kvcache_partition_len

        setattr(self, self.get_attention_name(), self.create_attention_op())
        self.kvcache_block_size = rbln_config.kvcache_block_size
        self.__post_init__()

    def __post_init__(self):
        self.q_proj = self._original_mod.wq
        self.k_proj = self._original_mod.wk
        self.v_proj = self._original_mod.wv
        self.o_proj = self._original_mod.wo

    def projection(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.qk_norm:
            query_states = self._original_mod.q_norm(query_states)
            key_states = self._original_mod.k_norm(key_states)

        return query_states, key_states, value_states

    def apply_rotary_pos_embed(self, query_states, key_states, cos, sin):
        return apply_rotary_pos_emb(query_states, key_states, cos, sin)


class MistralNeMoLayer(DecoderOnlyLayer):
    def get_pre_attention_layernorm(self):
        return self._original_mod.attention_norm

    def get_post_attention_layernorm(self):
        return self._original_mod.ffn_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_positions: torch.LongTensor,
        past_key_values: Tuple[Tuple[torch.Tensor]],
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states
        hidden_states = self.get_pre_attention_layernorm()(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            seq_positions=seq_positions,
            past_key_values=past_key_values,
            cos=cos,
            sin=sin,
            block_tables=block_tables,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.get_post_attention_layernorm()(hidden_states)
        hidden_states = self._original_mod.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MistralNeMoModel(DecoderOnlyModel):
    def get_embedding(self) -> nn.Embedding:
        return self._original_mod.tok_embeddings


class MistralNeMoForTextUpsampler(DecoderOnlyForCausalLM):
    def __init__(self, causal_lm, model):
        nn.Module.__init__(self)
        self.config = causal_lm.config
        self._original_mod = causal_lm
        self.model = model
        self._phase = "prefill"

    def forward(
        self,
        input_ids: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        query_position: torch.Tensor = None,
        past_key_values: Tuple[Tuple[torch.Tensor]] = None,
        rotary_emb: nn.Module = None,
        global_block_tables: Optional[torch.Tensor] = None,
        local_block_tables: Optional[torch.Tensor] = None,
    ):
        # outputs
        hidden_states = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
            past_key_values=past_key_values,
            rotary_emb=rotary_emb,
            global_block_tables=global_block_tables,
            local_block_tables=local_block_tables,
        )

        if self.phase == "prefill":
            hidden_states = hidden_states[:, query_position.to(torch.int).unsqueeze(0)]

        logits = self._original_mod.lm_head(hidden_states)
        return logits


class RotaryEmbedding1DV1(nn.Module):
    def __init__(
        self,
        config,
        max_seq_len_cached: int,
    ):
        super().__init__()
        self.mscale = 1.0
        self.config = config
        self.dim = config.head_dim if config.head_dim else self.config.dim // self.config.n_heads

        self.inv_freq = 1.0 / (
            self.config.rope_theta ** (torch.arange(0, self.dim, 2, dtype=torch.int64) / self.dim)
        )
        cache_position = torch.arange(0, max_seq_len_cached, dtype=torch.float32)
        cache_position_expanded = cache_position[:, None]

        inv_freq_expanded = self.inv_freq[None, :]
        self.freqs = cache_position_expanded.float() @ inv_freq_expanded.float()

        emb = torch.stack((self.freqs, self.freqs), dim=-1).reshape(*self.freqs.shape[:-1], -1)
        self.register_buffer("_cos_cached", (emb.cos() * self.mscale), persistent=False)
        self.register_buffer("_sin_cached", (emb.sin() * self.mscale), persistent=False)

    def forward(self, x, seq_len):
        return (
            self._cos_cached[:seq_len].to(dtype=x.dtype),
            self._sin_cached[:seq_len].to(dtype=x.dtype),
        )


class MistralNeMoForTextUpsamplerWrapper(DecoderOnlyWrapper):
    def get_attn_layer(self, layer):
        return layer.attention

    def get_rbln_attn_class(self):
        return MistralNeMoAttention

    def get_rbln_layer_class(self):
        return MistralNeMoLayer

    def get_rbln_model_class(self):
        return MistralNeMoModel

    def get_rbln_causal_lm_class(self):
        return MistralNeMoForTextUpsampler

    def get_rotary_emb(self, max_seq_len):
        return RotaryEmbedding1DV1(config=self.config, max_seq_len_cached=max_seq_len)
