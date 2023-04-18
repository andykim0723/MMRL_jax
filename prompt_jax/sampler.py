import torch
import torchvision

import jax
import jax.numpy as jnp

import copy
import pickle

import pprint
import random

from transformers import AutoProcessor


class Sampler:
    def __init__(self, cfg):
        
        with open(cfg['root'], 'rb') as f:
            data = pickle.load(f)
        self.traj = data
        self.cfg = cfg
        self.classes = data["classes"]
        self.actions = data["actions"]
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.augment_img = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.ColorJitter(
                brightness=(0.2, 2),
                contrast=(0.9, 1.5),
                saturation=(1.5, 2), 
                hue=(-0.4, 0.4)
            ),
            # torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self._preprocess_dataset()
    
    def sample_positive_pairs(self):
        # query
        jax_key = jax.random.PRNGKey(0)
        
        self._tempo_index = jax.random.randint(key=jax_key,minval=0, maxval=len(self.traj_temp_action_timestpes), shape=(self.cfg['batch_size'],))
        self._actions_index = jax.random.randint(key=jax_key,minval=0, maxval=jnp.array([len(self.traj_temp_action_timestpes[i]) for i in self._tempo_index]), shape=(self.cfg['batch_size'],))
        self._timestep_index = jax.random.randint(key=jax_key,minval=0, maxval=jnp.array([self.traj_temp_action_timestpes[i][j] for i, j in zip(self._tempo_index, self._actions_index)]), shape=(self.cfg['batch_size'],))
        x = []
        tempo_keys = list(self.traj_temp_action.keys())
        for i in range(self.cfg['batch_size']):
            tp = tempo_keys[self._tempo_index[i]]
            tempo_action_keys = list(self.traj_temp_action[tp].keys())
            a = tempo_action_keys[self._actions_index[i]]
            s = self._timestep_index[i]

            x.append(self.traj_temp_action[tp][a][s])

        query = jnp.stack(x)
        query = self.processor(images=query, return_tensors="np")['pixel_values']

        # key
        self._timestep_index = jax.random.randint(key=jax_key,minval=0, maxval=jnp.array([self.traj_temp_action_timestpes[i][j] for i, j in zip(self._tempo_index, self._actions_index)]), shape=(self.cfg['batch_size'],))
        x = []
        tempo_keys = list(self.traj_temp_action.keys())
        for i in range(self.cfg['batch_size']):
            tp = tempo_keys[self._tempo_index[i]]
            tempo_action_keys = list(self.traj_temp_action[tp].keys())
            a = tempo_action_keys[self._actions_index[i]]
            s = self._timestep_index[i]

            input = self.processor(images=self.traj_temp_action[tp][a][s], return_tensors="np")['pixel_values'][0]
            input = jnp.reshape(input,(224,224,3))
            input = self.augment_img(input).numpy()
            x.append(input)

        key = jnp.stack(x)
        
        return jnp.concatenate([query,key]).reshape(-1,224,224,3)

    def _preprocess_dataset(self,):
        self.mdps = list(self.traj.keys())
        self.mdps.remove("classes")
        self.mdps.remove("actions")
        self.timesteps = copy.deepcopy(self.mdps)
        for i, mdp in enumerate(self.mdps):
            self.timesteps[i] = len(self.traj[mdp]["frame"])
        # print(self.mdps)
        # print(self.timesteps)

        # to tensor
        for mdp in self.mdps:
            # print(mdp, len(self.traj[mdp]["frame"]))
            for i in range(len(self.traj[mdp]["frame"])):
                self.traj[mdp]["frame"][i] = self.traj[mdp]["frame"][i]
                self.traj[mdp]["action"][i] = torch.tensor(self.traj[mdp]["action"][i][0])
                self.traj[mdp]["reward"][i] = torch.tensor(self.traj[mdp]["reward"][i])
        
        # temporal and behaviour based
        base_tempo = 15
        self.traj_temp_action = {}
        self.traj_temp_action_timestpes = []
        # dictionary init
        for timestep in self.timesteps:
            for tempo in range(base_tempo+1):
                self.traj_temp_action[tempo] = {}
                for action in self.actions:
                    self.traj_temp_action[tempo][action] = []
        # pprint.pprint(self.traj_temp_action)
        
        for mdp in self.mdps:
            for i in range(len(self.traj[mdp]["frame"])):
                tempo_group = list(range(timestep//base_tempo, timestep+timestep//base_tempo, timestep//base_tempo))
                #for tempo in self.traj_temp_action.keys():
                for j, tempo in enumerate(tempo_group):
                    if i < tempo:
                        break
                action = self.actions[self.traj[mdp]["action"][i]]
                self.traj_temp_action[j][action].append(self.traj[mdp]["frame"][i])
        
        vis_mdps = random.sample(self.mdps, 2)
        self.traj_vis = {}
        
        for mdp in vis_mdps:
            self.traj_vis[mdp]=[]
            for i in range(len(self.traj[mdp]["frame"])):
                # self.traj_vis[mdp].append(self._to_tensor(self.traj[mdp]["frame"][i], False))
                self.traj_vis[mdp].append(self.traj[mdp]["frame"][i])
        
        # clearing
        actions = self.actions.copy()
        actions_check = self.actions.copy()
        tempo_keys = self.traj_temp_action.keys()
        for tempo in tempo_keys:
            self.traj_temp_action_timestpes.append([])
            for action in self.actions:
                if len(self.traj_temp_action[tempo][action]) == 0:
                    del self.traj_temp_action[tempo][action]
                    continue
                else:
                    self.traj_temp_action_timestpes[-1].append(len(self.traj_temp_action[tempo][action]))
                    try:
                        actions_check.remove(action)
                    except:
                        pass
        
        remove = []
        for i, content in enumerate(self.traj_temp_action_timestpes):
            if not len(content):
                remove.append(i)
        tempo_keys = list(self.traj_temp_action.keys())
        for r in remove:
            del self.traj_temp_action[tempo_keys[r]]
            del self.traj_temp_action_timestpes[r]

        if len(actions_check):
            # print(actions_check)
            for action in actions_check:
                self.actions.remove(action)
        
        del self.traj
        # pprint.pprint(self.traj_temp_action)
        # pprint.pprint(self.traj_temp_action_timestpes)
    
