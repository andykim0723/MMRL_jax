import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

import math 
from operator import mul
from functools import reduce
from typing import Callable, List, Tuple, Optional

from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from transformers import AutoTokenizer,FlaxCLIPModel,AutoProcessor
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers.models.clip.modeling_flax_clip import FlaxCLIPVisionEmbeddings,FlaxCLIPEncoder
from transformers.modeling_flax_outputs import FlaxBaseModelOutputWithPooling
from transformers.modeling_flax_utils import ACT2FN,FlaxPreTrainedModel


class FlaxCLIPPromptVisionTransformer(nn.Module):
    config: CLIPVisionConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.embeddings = FlaxCLIPVisionEmbeddings(self.config, dtype=self.dtype)
        self.pre_layrnorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.encoder = FlaxCLIPEncoder(self.config, dtype=self.dtype)
        self.post_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.num_token = 8
        self.prompt_dim = self.config.hidden_size
        # prompt
        self.prompt_dropout = nn.Dropout(rate=0.1)
        self.prompt_proj = nn.Dense(self.prompt_dim,kernel_init=nn.initializers.kaiming_normal())

        patch_size = (self.config.patch_size, self.config.patch_size)
        val =math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) +  self.prompt_dim))
        self.prompt_embeddings = self.param('prompt_embeddings',
                              nn.initializers.uniform(val),(1,self.num_token, self.prompt_dim))

    def __call__(
        self,
        pixel_values=None,
        deterministic: bool = True,
        output_attentions=None,
        output_hidden_states=None,
        return_dict: bool = True,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = self.embeddings(pixel_values)
        ### add prompt ###
        B = hidden_states.shape[0]
        prompts = self.prompt_dropout(self.prompt_proj(self.prompt_embeddings),deterministic=True).transpose(2,1,0)
        prompts = jnp.tile(prompts,B).transpose(2,1,0)
        hidden_states = jnp.concatenate([
            hidden_states[:, :1, :],
            prompts,
            hidden_states[:, 1:, :]
            ],axis=1)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class FlaxCLIPVisionModule(nn.Module):
    config: CLIPConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        vision_config = self.config.vision_config
        
        self.projection_dim = self.config.projection_dim
        self.vision_embed_dim = vision_config.hidden_size

        self.vision_model = FlaxCLIPPromptVisionTransformer(vision_config, dtype=self.dtype)

        self.visual_projection = nn.Dense(
            self.projection_dim,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            use_bias=False,
        )
    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        return image_embeds

class FlaxCLIPVisionPreTrainedModel(FlaxPreTrainedModel):
    config_class = CLIPConfig
    main_input_name = "pixel_values"
    module_class: nn.Module = FlaxCLIPVisionModule

    def __init__(
        self,
        config: CLIPConfig,
        input_shape: Optional[Tuple] = None,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        **kwargs,
    ):
        if input_shape is None:
            input_shape = (1, config.vision_config.image_size, config.vision_config.image_size, 3)
        module = self.module_class(config=config, dtype=dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensor
        pixel_values = jax.random.normal(rng, input_shape)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, pixel_values)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def __call__(
        self,
        pixel_values,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        return self.module.apply(
            {"params": params or self.params},
            jnp.array(pixel_values, dtype=jnp.float32),
            not train,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
        )
    
class CLModel(nn.Module):
    model : nn.Module #FlaxCLIPVisionPreTrainedModel
    cfg : dict

    def setup(self):
        self.online_predictor = nn.Sequential([
            nn.Dense(features=4096),
            nn.BatchNorm(use_running_average=not self.cfg['training']),
            nn.relu,
            nn.Dense(features=16)
        ])

    def __call__(self,pixel_values=None):
        im_emb = self.model(pixel_values)
        return self.online_predictor(im_emb)
        
