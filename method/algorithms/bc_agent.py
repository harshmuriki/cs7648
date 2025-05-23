from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR

from .base_agent import BaseAgent
from .expert_dataset import ExpertDataset
from ..networks import Actor
from ..utils.info_dict import Info
from ..utils.logger import logger
from ..utils.mpi import mpi_average
from ..utils.pytorch import (
    optimizer_cuda,
    count_parameters,
    compute_gradient_norm,
    compute_weight_norm,
    sync_networks,
    sync_grads,
    to_tensor,
)


class BCAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space, env_ob_space):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        self._epoch = 0

        self._actor = Actor(config, ob_space, ac_space, config.tanh_policy)
        self._network_cuda(config.device)
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.bc_lr)
        self._actor_lr_scheduler = StepLR(
            self._actor_optim, step_size=self._config.max_global_step // 5, gamma=0.5,
        )

        if config.is_train:
            self._dataset = ExpertDataset(
                config.demo_path,
                config.demo_subsample_interval,
                ac_space,
                use_low_level=config.demo_low_level,
            )

            if self._config.val_split != 0:
                dataset_size = len(self._dataset)
                indices = list(range(dataset_size))
                split = int(np.floor((1 - self._config.val_split) * dataset_size))
                train_indices, val_indices = indices[split:], indices[:split]
                train_sampler = SubsetRandomSampler(train_indices)
                val_sampler = SubsetRandomSampler(val_indices)
                self._train_loader = torch.utils.data.DataLoader(
                    self._dataset,
                    batch_size=self._config.batch_size,
                    sampler=train_sampler,
                )
                self._val_loader = torch.utils.data.DataLoader(
                    self._dataset,
                    batch_size=self._config.batch_size,
                    sampler=val_sampler,
                )
            else:
                self._train_loader = torch.utils.data.DataLoader(
                    self._dataset, batch_size=self._config.batch_size, shuffle=True
                )

        self._log_creation()

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a BC agent")
            logger.info("The actor has %d parameters", count_parameters(self._actor))

    def state_dict(self):
        return {
            "actor_state_dict": self._actor.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
            "ob_norm_state_dict": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
        self._network_cuda(self._config.device)

        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        optimizer_cuda(self._actor_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)

    def sync_networks(self):
        sync_networks(self._actor)

    def train(self):
        train_info = Info()
        for transitions in self._train_loader:
            _train_info = self._update_network(transitions, train=True)
            train_info.add(_train_info)
        self._epoch += 1
        self._actor_lr_scheduler.step()

        train_info.add(
            {
                "actor_grad_norm": compute_gradient_norm(self._actor),
                "actor_weight_norm": compute_weight_norm(self._actor),
            }
        )
        train_info = train_info.get_dict(only_scalar=True)
        logger.info("BC loss %f", train_info["actor_loss"])
        return train_info

    def evaluate(self):
        if self._val_loader:
            eval_info = Info()
            for transitions in self._val_loader:
                _eval_info = self._update_network(transitions, train=False)
                eval_info.add(_eval_info)
            self._epoch += 1
            return eval_info.get_dict(only_scalar=True)
        logger.warning("No validation set available, make sure '--val_split' is set")
        return None

    def _update_network(self, transitions, train=True):
        info = Info()

        # pre-process observations
        o = transitions["ob"]
        o = self.normalize(o)

        # convert double tensor to float32 tensor
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        print("Processed observation shape:")
        # for key, value in o.items():
        #     if hasattr(value, "shape"):
        #         print(f"{key}: {value.shape}")
        #     elif isinstance(value, list):
        #         print(f"{key}: {len(value)} (list with length)")
        #     else:
        #         print(f"{key}: {type(value)} (no shape)")
        ac = _to_tensor(transitions["ac"])
        for key, value in ac.items():
            if hasattr(value, "shape"):
                print(f"{key}: {value.shape}")
            elif isinstance(value, list):
                print(f"{key}: {len(value)} (list with length)")
            else:
                print(f"{key}: {type(value)} (no shape)")
        if isinstance(ac, OrderedDict):
            ac = list(ac.values())
            if len(ac[0].shape) == 1:
                ac = [x.unsqueeze(0) for x in ac]
            ac = torch.cat(ac, dim=-1)
            
        # the actor loss
        pred_ac, _ = self._actor(o)
        if isinstance(pred_ac, OrderedDict):
            pred_ac = list(pred_ac.values())
            if len(pred_ac[0].shape) == 1:
                pred_ac = [x.unsqueeze(0) for x in pred_ac]
            pred_ac = torch.cat(pred_ac, dim=-1)
        print("Predicted action shape:", pred_ac.shape)

        diff = ac - pred_ac
        actor_loss = diff.pow(2).mean()
        info["actor_loss"] = actor_loss.cpu().item()
        info["pred_ac"] = pred_ac.cpu().detach()
        info["GT_ac"] = ac.cpu()
        
        print("pred_ac", pred_ac[0].cpu().detach())
        print("GT_ac", ac[0].cpu().detach())
        print("actor loss", info["actor_loss"])
        
        diff = torch.sum(torch.abs(diff), axis=0).cpu()
        for i in range(diff.shape[0]):
            info["action" + str(i) + "_L1loss"] = diff[i].mean().item()

        if train:
            # update the actor
            self._actor_optim.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._config.max_grad_norm)
            sync_grads(self._actor)
            self._actor_optim.step()

        return mpi_average(info.get_dict(only_scalar=True))
