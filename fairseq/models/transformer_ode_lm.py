# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import math

from fairseq import options, utils
from fairseq.models import (
    FairseqLanguageModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    Embedding,
)

from fairseq.models.fairseq_encoder import EncoderOut

from fairseq.modules import (
    AdaptiveSoftmax,
    FairseqDropout,
    AdaptiveInput,
    CharacterTokenEmbedder,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    MultiheadAttention,
)

from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.layer_history import CreateLayerHistory
from torch import Tensor

from fairseq.modules.drop_path import DropPath
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('transformer_ode_lm')
class TransformerODELanguageModel(FairseqLanguageModel):

    @classmethod
    def hub_models(cls):

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer_lm.gbw.adaptive_huge': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_gbw_huge.tar.bz2',
            'transformer_lm.wiki103.adaptive': 'https://dl.fbaipublicfiles.com/fairseq/models/lm/adaptive_lm_wiki103.v2.tar.bz2',
            'transformer_lm.wmt19.en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.bz2'),
            'transformer_lm.wmt19.de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.de.tar.bz2'),
            'transformer_lm.wmt19.ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.ru.tar.bz2'),
        }

    def __init__(self, decoder):
        super().__init__(decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension')
        parser.add_argument('--decoder-input-dim', type=int, metavar='N',
                            help='decoder input dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--no-decoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last decoder block')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--adaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--no-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--character-embeddings', action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--character-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--character-embedding-dim', default=4, type=int, metavar='N',
                            help='size of character embeddings')
        parser.add_argument('--char-embedder-highway-layers', default=2, type=int, metavar='N',
                            help='number of highway layers for character token embeddder')
        parser.add_argument('--adaptive-input', action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--adaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--adaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')
        parser.add_argument('--tie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--tie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D',
                            help='LayerDrop probability for decoder')
        parser.add_argument('--decoder-layers-to-keep',
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D',
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D',
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D',
                            help='scalar quantization noise and scalar quantization at training time')
        # fmt: on

        parser.add_argument('--max-relative-length', type=int, default=-1,
                            help='the max relative length')
        parser.add_argument('--k-only', default=False, action='store_true',
                            help='select the relative mode to map relative position information')

        # for Dynamic Linear Combinations of Layers
        parser.add_argument('--decoder-history-type',
                            help='decoder layer history type')
        parser.add_argument('--decoder-integration-type', choices=['avg', 'sum'],
                            help='decoder layer integration type')

        # the order of RK-method
        parser.add_argument('--dec-calculate-num', type=int, default=1,
                            help='Number of calculations per decoder layer')

        parser.add_argument('--dec-learnable-type', choices=['gated','ema','RK'])

        parser.add_argument('--alpha-type', choices=['scalar', 'vector'])

        parser.add_argument('--layer-wise', action='store_true',
                            help='the learnable coefficients are whether layer-wise or not')
        parser.add_argument('--rk-norm', action='store_true',
                            help='the RK intermediate representations are normed or not')

        parser.add_argument('--drop-path', type=float, default=0.0,
                            help='to alleviate the overfitting')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_lm_architecture(args)

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = getattr(args, 'tokens_per_sample', DEFAULT_MAX_TARGET_POSITIONS)

        if args.character_embeddings:
            embed_tokens = CharacterTokenEmbedder(
                task.source_dictionary, eval(args.character_filters),
                args.character_embedding_dim, args.decoder_embed_dim,
                args.char_embedder_highway_layers,
            )
        elif args.adaptive_input:
            embed_tokens = AdaptiveInput(
                len(task.source_dictionary), task.source_dictionary.pad(), args.decoder_input_dim,
                args.adaptive_input_factor, args.decoder_embed_dim,
                options.eval_str_list(args.adaptive_input_cutoff, type=int),
                args.quant_noise_pq, args.quant_noise_pq_block_size,
            )
        else:
            embed_tokens = cls.build_embedding(args, task.source_dictionary, args.decoder_input_dim)

        if args.tie_adaptive_weights:
            assert args.adaptive_input
            assert args.adaptive_input_factor == args.adaptive_softmax_factor
            assert args.adaptive_softmax_cutoff == args.adaptive_input_cutoff, '{} != {}'.format(
                args.adaptive_softmax_cutoff, args.adaptive_input_cutoff)
            assert args.decoder_input_dim == args.decoder_output_dim

        decoder = TransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True,
        )
        return cls(decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return embed_tokens



class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = None
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )

        self.no_mask_counter = 0
        self.to_see = self.args.max_tokens

        # create decoder layer history
        self.history = CreateLayerHistory(args, is_encoder=False)

        self.calculate_num = args.dec_calculate_num
        self.dec_learnable_type = args.dec_learnable_type
        self.alpha_type = args.alpha_type
        self.layer_wise = args.layer_wise

        # create the layer norm for the intermediate approxiamtions of high-order ODE computation
        # to ensure that each of the representation has been normed
        # we provide a shared version among different layers
        self.rk_norm = getattr(args, 'rk_norm', False)
        self.RK_norm = nn.ModuleList(LayerNorm(self.embed_dim) for _ in range(self.calculate_num)) if self.rk_norm else None
        self.residual_norm = nn.ModuleList(LayerNorm(embed_dim) for _ in range(args.decoder_layers)) if self.rk_norm else None
        if self.calculate_num == 2:
            if self.dec_learnable_type == 'gated':
                self.gate_linear = Linear(2 * self.embed_dim, 1)
            elif self.dec_learnable_type == 'ema':
                if self.alpha_type == 'scalar':
                    if self.layer_wise:
                        self.alpha = torch.nn.Parameter(torch.Tensor(args.decoder_layers, 1))
                        self.alpha.data.fill_(0.5)
                    else:
                        self.alpha = torch.nn.Parameter(torch.Tensor(1))
                        self.alpha.data.fill_(0.5)
                elif self.alpha_type == 'vector':
                    self.alpha = torch.nn.Parameter(torch.Tensor(self.embed_dim))
                    self.alpha.data.fill_(0.5)
        elif self.calculate_num == 4: 
            if self.dec_learnable_type == 'ema':
                if self.alpha_type == 'scalar':
                    if self.layer_wise:
                        self.alpha = torch.nn.Parameter(torch.Tensor(args.decoder_layers, 1))
                        self.alpha.data.fill_(0.5)
                    else:
                        self.alpha = torch.nn.Parameter(torch.Tensor(1))
                        self.alpha.data.fill_(0.5)
                elif self.alpha_type == 'vector':
                    self.alpha = torch.nn.Parameter(torch.Tensor(self.embed_dim))
                    self.alpha.data.fill_(0.5)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return TransformerDecoderLayer(args, no_encoder_attn)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. Aa copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        if self.history is not None:
            self.history.clean()

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        # if positions is not None:
        #     x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # add emb into history
        self.history.add(x)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)
        else:
            # Even when using sliding inference, we don't apply it to the first args.max_tokens tokens
            # (max_tokens is what we call subsequence length (L) in the paper). This is because it probably
            # wouldn't save much time. In order to detect if we are still generating the first max_tokens tokens
            # we check if there are any padding symbols in the input. If yes, it means we are still generating
            # the first max_tokens tokens. If not, we start incrementing the no_mask_counter (which makes the model
            # save its first cache entry. When no_mask_counter > 1, it means we have data in our cache and can start
            # using it.
            # to_see is the number of tokens that our model can attend to right now. At first its max_tokens and
            # it grows by 1 every iteration (until resetting at max_tokens*2). This is because in all our models
            # the number of entries in the cache (L' in the paper) is equal to the number of tokens in each subsequence (L).

            if self.args.sliding_inf > -1: # Only get beyond this line if we are doing token-by-token inference.
                self.no_mask_counter += 1
                if self.no_mask_counter > 1:
                    self.to_see +=1
                    if self.to_see > self.args.max_tokens*2  :
                        self.to_see = self.args.max_tokens + 1

        if self.no_mask_counter > 1:
            # We only get here during token-by-token inference.
            # When we do token-by-token inference, fairseq gives us the previous max_tokens at every iteration.
            # But if we're here, it means we've started using the cache, which means we don't need to recompute
            # all previous representations (we're going to reuse the cached ones), so we just need the *last* input token
            # and that is what we do here.

            x_shape = x.shape
            x = x[-1:,:,:]



        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x = self.history.pop()
            
            runge_kutta_list = []
            if self.rk_norm:
                residual = self.residual_norm[idx](x)
            else:
                residual = x


            # we use the RK2 or RK4 methods as the predictor to generate a rouge prediction
            for j in range(self.calculate_num):

                x, layer_attn, _ = layer(
                x,
                positions.transpose(0, 1),
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                to_see = self.to_see if self.no_mask_counter > 1 else 0,
                is_cache = j,
            )
                if self.rk_norm:
                    x = self.RK_norm[j](x)
                    runge_kutta_list.append(x)
                else:
                    runge_kutta_list.append(x)

                # to construct the order-input for the next step computation
                if self.calculate_num == 4:
                    if j == 0 or j == 1:
                        x = residual + 1 / 2 * x
                    elif j == 2:
                        x = residual + x
                elif self.calculate_num == 2:
                    x = residual + x
            if self.calculate_num == 4:
                if self.dec_learnable_type == 'ema':
                    x = residual + self.alpha * torch.pow(1-self.alpha,3) * runge_kutta_list[0] + self.alpha * torch.pow(1-self.alpha,2) * runge_kutta_list[1] + self.alpha * (1-self.alpha) * runge_kutta_list[2] + self.alpha * runge_kutta_list[3]
                else:
                    x = residual + 1 / 6 * (runge_kutta_list[0] + 2 * runge_kutta_list[1] + 2 * runge_kutta_list[2] + runge_kutta_list[3])
            elif self.calculate_num == 2:
                if self.dec_learnable_type == 'gated':
                    alpha = torch.sigmoid(self.gate_linear(torch.cat((runge_kutta_list[0], runge_kutta_list[1]), dim=-1)))
                    x = residual + alpha * runge_kutta_list[0] + (1 - alpha) * runge_kutta_list[1]
                elif self.dec_learnable_type == 'ema':
                    x = residual + self.alpha*(1-self.alpha) * runge_kutta_list[0] + self.alpha*runge_kutta_list[1]
                else:
                    x = residual + 1/2 * (runge_kutta_list[0] + runge_kutta_list[1])
            else:
                raise ValueError("invalid caculate num！")
            
            self.history.add(x) 

            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.history is not None:
            x = self.history.pop()

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        if self.no_mask_counter > 1:
            # We only get here during token-by-token inference.
            # We're making a prediction about just the next *single* output, but
            # because fairseq gave us max_tokens tokens it wants us to give it max_tokens predictions
            # (although in this mode it'll only look at the last prediction), so we just pad
            # our output with zeros (that will be ignored).
            x = torch.cat((x.new_zeros(x_shape[0]-1, x_shape[1], x_shape[2]), x), dim=0)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.args = args
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn) if getattr(args, "activation_fn", None) is not None else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__)
        self.normalize_before = args.decoder_normalize_before

        # drop path for further regularization
        self.drop_path = DropPath(args.drop_path) if args.drop_path > 0. else nn.Identity()

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim, args.decoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

        self.tfp = self.args.tokens_from_prev
        if self.tfp > 0:
            self.prev_out = None # This is the cache

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            #self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        positions,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        to_see: int = 0,
        is_cache: int = 10,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x
        if self.tfp > 0 and is_cache == 0:
            if self.args.sliding_inf == -1:
                # If we're here it means we're training. The code below just attaches the
                # existing cache to the input we just received.
                if self.prev_out is not None:
                    if y.shape[1] != self.prev_out.shape[1]: # An edge case that only happens sometimes in the last batch of the epoch
                        self.prev_out = None
                    else:
                        from_prev = self.prev_out[-self.tfp:,:,:]
                        y = torch.cat((from_prev, y))
                        # If we use a cache, it means we have to update our mask- all the tokens in the cache are
                        # historical, so all timesteps can look at them.
                        self_attn_mask = torch.cat((self_attn_mask.new_zeros(x.shape[0], self.tfp), self_attn_mask), dim=1)
                    self.prev_out = y.detach().clone()
                else:
                    self.prev_out = y.detach().clone() # This is the initial subsequence

            else:
                # This block is only used during token-by-token inference.
                if y.shape[0] == self.args.max_tokens:
                    # This is the first batch of inputs we get which we save in the cache.
                    # After this, all inputs will be of length 1 so we will not reach this line.
                    self.prev_out = y.detach().clone()
                elif y.shape[0] == 1:
                    if y.shape[1] != self.prev_out.shape[1]:
                        # Weird case that can only happen if batch size is not 1.
                        print('skipping', y.shape[1], self.prev_out.shape[1])
                        self.prev_out = y.detach().clone()
                    else:
                        from_prev = self.prev_out[-(to_see-1):, :, :]
                        y = torch.cat((from_prev, y))
                        self_attn_mask = None # We are doing token by token inference so we don't need an attention mask.
                        self.prev_out = y.detach().clone()


        pos_key = positions[:y.shape[0]]

        if to_see > 0:
            pos_query = positions[to_see - 1].unsqueeze(0) # This is what we do during (token-by-token) inference.
        else:
            pos_query = positions[y.shape[0]-x.shape[0]:y.shape[0]] # This is what we do during training.
                                                                    # If our subsequence length is 3, that means
                                                                    # that the cache will always get positional embeddings [1,2,3] and so
                                                                    # the current subsequence gets positional embeddings [4,5,6], both
                                                                    # in the query and in the key.
                                                                    # They keys/values are composed of both the current subsequence and the previous one
                                                                    # while the query is composed just of the vectors of the current subsequence.
                                                                    # In the next iteration, words 4,5,6 get moved into the cache and get positional embeddings 1,2,3
                                                                    # and the new words get positional embeddings 4,5,6, and then the process is repeated.

        x, attn = self.self_attn(
            query=x + pos_query,
            key=y + pos_key,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x= self.drop_path(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x= self.drop_path(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


@register_model_architecture('transformer_ode_lm', 'transformer_ode_lm')
def base_lm_architecture(args):
    # backward compatibility for older model checkpoints
    if hasattr(args, 'no_tie_adaptive_proj'):
        # previous models defined --no-tie-adaptive-proj, so use the existence of
        # that option to determine if this is an "old" model checkpoint
        args.no_decoder_final_norm = True  # old models always set this to True
        if args.no_tie_adaptive_proj is False:
            args.tie_adaptive_proj = True
    if hasattr(args, 'decoder_final_norm'):
        args.no_decoder_final_norm = not args.decoder_final_norm

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.0)

    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', 4)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')

    args.decoder_layerdrop = getattr(args, 'decoder_layerdrop', 0)
    args.decoder_layers_to_keep = getattr(args, 'decoder_layers_to_keep', None)
    args.quant_noise_pq = getattr(args, 'quant_noise_pq', 0)
    args.quant_noise_pq_block_size = getattr(args, 'quant_noise_pq_block_size', 8)
    args.quant_noise_scalar = getattr(args, 'quant_noise_scalar', 0)

    args.add_bos_token = getattr(args, 'add_bos_token', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.character_embeddings = getattr(args, 'character_embeddings', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    # Model training is not stable without this
    args.decoder_normalize_before = True
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', False)

    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.adaptive_input_factor = getattr(args, 'adaptive_input_factor', 4)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', None)

    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)

    args.max_relative_length = getattr(args, 'max_relative_length', args.max_relative_length)
    args.k_only = getattr(args, 'k_only', args.k_only)

    args.decoder_history_type = getattr(args, 'decoder_history_type', 'rk_predicotor_multistep_corrector')
    args.decoder_integration_type = getattr(args, 'decoder_integration_type', 'avg')

    args.dec_calculate_num = getattr(args, 'dec_calculate_num', 2)
    args.dec_learnable_type = getattr(args, 'dec_learnable_type', 'ema')
    args.alpha_type = getattr(args, 'alpha_type', 'scalar')
    args.layer_wise = getattr(args, 'layer_wise', False)
    args.rk_norm = getattr(args, 'rk_norm', False)

    args.drop_path = getattr(args, 'drop_path', 0)


@register_model_architecture('transformer_ode_lm', 'transformer_ode_lm_big')
def transformer_ode_lm_big(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    base_lm_architecture(args)


@register_model_architecture('transformer_ode_lm', 'transformer_ode_lm_wiki103')
@register_model_architecture('transformer_ode_lm', 'transformer_ode_lm_baevski_wiki103')
def transformer_ode_lm_baevski_wiki103(args):
    args.decoder_layers = getattr(args, 'decoder_layers', 16)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.dropout = getattr(args, 'dropout', 0.3)
    args.adaptive_input = getattr(args, 'adaptive_input', True)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', True)
    args.adaptive_input_cutoff = getattr(args, 'adaptive_input_cutoff', '20000,60000')
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', '20000,60000')
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0.2)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', True)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', True)
    args.decoder_history_type = getattr(args, 'decoder_history_type', 'learnable_dense')
    transformer_ode_lm_big(args)


@register_model_architecture('transformer_ode_lm', 'transformer_ode_lm_gbw')
@register_model_architecture('transformer_ode_lm', 'transformer_ode_lm_baevski_gbw')
def transformer_ode_lm_baevski_gbw(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.no_decoder_final_norm = getattr(args, 'no_decoder_final_norm', True)
    transformer_ode_lm_big(args)


@register_model_architecture('transformer_ode_lm', 'transformer_ode_lm_gpt')
def transformer_ode_lm_gpt(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 768)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 3072)
    args.decoder_layers = getattr(args, 'decoder_layers', 12)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 12)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    base_lm_architecture(args)


@register_model_architecture('transformer_ode_lm', 'transformer_ode_lm_gpt2_small')
def transformer_ode_lm_gpt2_small(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_layers = getattr(args, 'decoder_layers', 24)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    base_lm_architecture(args)


@register_model_architecture('transformer_ode_lm', 'transformer_ode_lm_gpt2_medium')
def transformer_ode_lm_gpt2_medium(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1280)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 5120)
    args.decoder_layers = getattr(args, 'decoder_layers', 36)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 20)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    base_lm_architecture(args)


@register_model_architecture('transformer_ode_lm', 'transformer_ode_lm_gpt2_big')
def transformer_ode_lm_gpt2_big(args):
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1600)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 6400)
    args.decoder_layers = getattr(args, 'decoder_layers', 48)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 25)
    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_fn = getattr(args, 'activation_fn', 'gelu')
    base_lm_architecture(args)
