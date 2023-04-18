import os 
import yaml
import wandb
import argparse

import jax
from flax.training import checkpoints

import orbax
from flax.training import orbax_utils

from prompt_jax.VPT_jax import FlaxCLIPVisionPreTrainedModel,CLModel
from prompt_jax.trainer import Trainer
from prompt_jax.sampler import Sampler

def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--domain-factor', help='domain factor', type=str)   
    parser.add_argument('--cfg', help='path of the config file', type=str)   

    args = parser.parse_args()

    return args 

def init_wandb(config_path,domain_factor):
    config_path 
    with open(config_path,'r') as f:
        cfg = yaml.load(f,Loader=yaml.FullLoader)

    wandb.init(
        project="prompt_learning",
        entity="andykim0723",
        job_type=domain_factor
    )        
    config = wandb.config
    config.seed = cfg['train_info']['seed']
    config.lr = cfg['train_info']['lr']
    config.epochs = cfg['train_info']['epoch']
    config.iterations = cfg['train_info']['iters_per_epoch']
    config.training = cfg['train_info']['training']
    config.batch_size = cfg['train_info']['batch_size']
    
    data_root = cfg['dataset']['data_root'] 
    save_root = cfg['train_info']['save_root']
    difficulty = cfg['dataset']['difficulty']
    
    config.root = os.path.join(data_root,difficulty,domain_factor,'train_dataset.pkl')    
    config.save_path = os.path.join(save_root,domain_factor)

    return config

if __name__ == "__main__":
    
    # wandb & configs
    args = parse_args()
    domain_factor = args.domain_factor
    cfg_path = args.cfg
    cfg = init_wandb(cfg_path,domain_factor)

    # initialize training modules
    pretrained_clip = FlaxCLIPVisionPreTrainedModel.from_pretrained("openai/clip-vit-base-patch32")
    # # with open("state_pretrained.txt", "a") as f:
    #     print(pretrained_clip.params, file=f)
    # exit()
    sampler = Sampler(cfg=cfg)
    model = CLModel(model=pretrained_clip.module,cfg=cfg)
    trainer = Trainer(cfg=cfg,model=model)
    
    rng = jax.random.PRNGKey(cfg['seed'])  # dropout is implemented internally(FlaxCLIPVisionPreTrainedModel)
    
    # train
    state = trainer.init_train_state(
        random_key=rng,
        shape=(1,224,224,3)
    )
    # with open("state1.txt", "a") as f:
    #     print(state.params, file=f)

    state,metrics = trainer.run(sampler=sampler,state=state)
    # with open("state2.txt", "a") as f:
    #     print(state.params, file=f)
    # TODO add model.save

    ckpt = {'model': state, 'config': dict(cfg)}

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(cfg['save_path'], ckpt, save_args=save_args)
