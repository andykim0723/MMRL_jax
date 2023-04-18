import wandb
from tqdm import tqdm
from typing import Dict

import jax
import flax
import optax

import jax.numpy as jnp

from flax.training import train_state
from flax.core import freeze, unfreeze

from prompt_jax.VPT_jax import CLModel,FlaxCLIPVisionPreTrainedModel
from prompt_jax.loss import mse

class TrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict
    
@jax.jit
def train_step(
    state: TrainState,
    batch: jnp.ndarray,
):
    def loss_fn(params):
        proj_features,updates = state.apply_fn({'params': params,'batch_stats': state.batch_stats},
                                               batch,mutable=['batch_stats'])
        f_query, f_key = jnp.split(proj_features,2, axis=0)
        loss = mse(f_query,f_key)

        return loss.mean(), updates
        
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss,updates), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    metrics =  {'loss': loss}

    return state, metrics

class Trainer:
    def __init__(
        self,
        cfg: Dict,
        model: CLModel
    ):
        self.config = cfg
        self.model = model

    def init_train_state(
        self,random_key, shape
    ) -> TrainState:
        
        # Initialize the Model
        variables = self.model.init(random_key, jnp.ones(shape))
        variables = unfreeze(variables)

        pretrained_clip = FlaxCLIPVisionPreTrainedModel.from_pretrained("openai/clip-vit-base-patch32")
        variables['params']['model'] = pretrained_clip.params
        # with open("state_loaded.txt", "a") as f:
        #     print(variables['params'], file=f)
        # print(variables['params']['online_predictor'])
        # print(variables['params']['model']['vision_model']['prompt_proj'])
        # print(variables['params']['model']['vision_model']['prompt_embeddings'])

        # Freezes all prompts and online_predictor.
        label_fn = flattened_traversal(
            lambda path, _: 'adam' if 'prompt_embeddings' in path or 'prompt_proj' in path or path[0] == 'online_predictor' 
            else 'none')
        tx = optax.multi_transform({'adam': optax.adam(self.config['lr']), 'none': optax.set_to_zero()},label_fn)
        # tx = optax.adam(self.config['lr'])
        
        ## check whether target parameters are frozen
        # fake_grads = jax.tree_map(jnp.ones_like, variables['params'].unfreeze())
        # opt_state = tx.init(variables['params'].unfreeze())
        # updates, opt_state = tx.update(fake_grads, opt_state)

        return TrainState.create(
            apply_fn = self.model.apply,
            tx=tx,
            params=unfreeze(variables['params']),
            batch_stats = variables['batch_stats']
        )

    def run(self, sampler, state):
        epochs = self.config['epochs']
        iters = self.config['iterations']

        losses = []
        for epoch in tqdm(range(1, epochs + 1)):
            # ============== Training ============== #
            train_batch_metrics = []
            for batch_idx in range(iters):
                batch = sampler.sample_positive_pairs()
                state, metrics = train_step(state,batch)
                train_batch_metrics.append(metrics)
            
            loss = [metric['loss'] for metric in train_batch_metrics]
            avg_loss = sum(loss)/len(loss)
            losses.append(avg_loss)
            print(f"epoch: {epoch}, loss: {avg_loss}")
            wandb.log({
                "Train Loss": avg_loss
            }, step=epoch)
        return state,losses

def flattened_traversal(fn):
    def mask(tree):
        flat = flax.traverse_util.flatten_dict(tree)
        return flax.traverse_util.unflatten_dict(
            {k: fn(k, v) for k, v in flat.items()})
    return mask

