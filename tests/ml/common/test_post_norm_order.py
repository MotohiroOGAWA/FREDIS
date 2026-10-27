"""All attention residual blocks run their sublayer before Add & Norm."""

import torch

from fredis.ml.common.layers.transformer_encoder import TransformerEncoderLayer
from fredis.ml.common.layers.transformer_decoder import TransformerDecoderLayer
from fredis.ml.common.layers.graphormer import GraphormerBlock
from fredis.ml.common.layers.set_transformer import MAB
from fredis.ml.common.layers.gat import GATBlock
from fredis.ml.specgen.components.reaction.reactant_atom_attention import ReactionCrossAttentionBlock


def capture(module, names):
    events, inputs, outputs = [], {}, {}
    for name in names:
        child = getattr(module, name)
        def before(_module, args, name=name):
            events.append(name)
            inputs[name] = tuple(value.detach().clone() for value in args if isinstance(value, torch.Tensor))
        def after(_module, _args, value, name=name):
            outputs[name] = (value[0] if isinstance(value, tuple) else value).detach().clone()
        child.register_forward_pre_hook(before)
        child.register_forward_hook(after)
    return events, inputs, outputs


def test_encoder_attention_and_ffn_are_followed_by_residual_normalization():
    model = TransformerEncoderLayer(8, 2).eval()
    names = ['self_attn', 'self_norm', 'ffn', 'ffn_norm']
    events, inputs, outputs = capture(model, names)
    x = torch.randn(2, 3, 8)
    model(x)
    assert events == names
    torch.testing.assert_close(inputs['self_attn'][0], x)
    torch.testing.assert_close(inputs['self_norm'][0], x + outputs['self_attn'])
    torch.testing.assert_close(inputs['ffn'][0], outputs['self_norm'])
    torch.testing.assert_close(inputs['ffn_norm'][0], outputs['self_norm'] + outputs['ffn'])


def test_decoder_uses_raw_memory_and_post_norm_for_every_residual():
    model = TransformerDecoderLayer(8, 2).eval()
    names = ['self_attn', 'self_norm', 'cross_attn', 'cross_norm', 'ffn', 'ffn_norm']
    events, inputs, outputs = capture(model, names)
    x, memory = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    model(x, memory)
    assert events == names
    torch.testing.assert_close(inputs['self_attn'][0], x)
    torch.testing.assert_close(inputs['self_norm'][0], x + outputs['self_attn'])
    torch.testing.assert_close(inputs['cross_attn'][0], outputs['self_norm'])
    torch.testing.assert_close(inputs['cross_attn'][1], memory)
    torch.testing.assert_close(inputs['cross_norm'][0], outputs['self_norm'] + outputs['cross_attn'])
    torch.testing.assert_close(inputs['ffn'][0], outputs['cross_norm'])
    torch.testing.assert_close(inputs['ffn_norm'][0], outputs['cross_norm'] + outputs['ffn'])


def test_graphormer_self_cross_attention_and_ffn_use_post_norm():
    model = GraphormerBlock(8, 2, use_context=True).eval()
    names = ['mha', 'norm1', 'cross_attention', 'cross_norm', 'ffn', 'norm2']
    events, inputs, outputs = capture(model, names)
    x, memory = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    model(x, node_mask=torch.ones(2, 3, dtype=torch.bool), context_tokens=memory)
    assert events == names
    torch.testing.assert_close(inputs['mha'][0], x)
    torch.testing.assert_close(inputs['norm1'][0], x + outputs['mha'])
    torch.testing.assert_close(inputs['cross_attention'][0], outputs['norm1'])
    torch.testing.assert_close(inputs['cross_attention'][1], memory)
    torch.testing.assert_close(inputs['cross_norm'][0], outputs['norm1'] + outputs['cross_attention'])
    torch.testing.assert_close(inputs['ffn'][0], outputs['cross_norm'])
    torch.testing.assert_close(inputs['norm2'][0], outputs['cross_norm'] + outputs['ffn'])


def test_reaction_cross_attention_uses_post_norm():
    model = ReactionCrossAttentionBlock(8, 2, 4, 0.0).eval()
    names = ['out_proj', 'attention_norm', 'ffn', 'ffn_norm']
    events, inputs, outputs = capture(model, names)
    query = torch.randn(2, 2, 8)
    model(query, torch.randn(1, 3, 8), torch.zeros(2, 2, 2, 3),
          torch.zeros(2, dtype=torch.long),
          torch.ones(1, 3, dtype=torch.bool))
    assert events == names
    torch.testing.assert_close(inputs['attention_norm'][0], query + outputs['out_proj'])
    torch.testing.assert_close(inputs['ffn'][0], outputs['attention_norm'])
    torch.testing.assert_close(inputs['ffn_norm'][0], outputs['attention_norm'] + outputs['ffn'])


def test_set_transformer_uses_post_norm():
    model = MAB(8, 8, 8, num_heads=2).eval()
    names = ['q_proj', 'attn', 'ln0', 'ffn', 'ln1']
    events, inputs, outputs = capture(model, names)
    model(torch.randn(2, 3, 8), torch.randn(2, 5, 8))
    assert events == names
    torch.testing.assert_close(inputs['ln0'][0], outputs['q_proj'] + outputs['attn'])
    torch.testing.assert_close(inputs['ffn'][0], outputs['ln0'])
    torch.testing.assert_close(inputs['ln1'][0], outputs['ln0'] + outputs['ffn'])


def test_gat_uses_post_norm():
    model = GATBlock(8, 2).eval()
    names = ['gat', 'norm1', 'feed_forward', 'norm2']
    events, inputs, outputs = capture(model, names)
    x = torch.randn(3, 8)
    model(x, torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), torch.randn(4, 2))
    assert events == names
    torch.testing.assert_close(inputs['gat'][0], x)
    torch.testing.assert_close(inputs['norm1'][0], x + outputs['gat'])
    torch.testing.assert_close(inputs['feed_forward'][0], outputs['norm1'])
    torch.testing.assert_close(inputs['norm2'][0], outputs['norm1'] + outputs['feed_forward'])
