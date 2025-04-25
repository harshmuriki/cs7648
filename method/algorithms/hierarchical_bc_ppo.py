# algorithms/hierarchical_bc_ppo_agent.py
import torch
import torch.optim as optim
import numpy as np
from gym.spaces import Discrete
from .base_agent import BaseAgent
from ..networks import Actor, Critic
from ..utils.pytorch import to_tensor, obs2tensor, sync_networks, sync_grads
from ..utils.info_dict import Info
from ..utils.mpi import mpi_average
from .dataset import ReplayBuffer, RandomSampler
from gym.spaces import Discrete, Dict as SpaceDict
from torch.distributions import Categorical
Categorical.rsample = Categorical.sample

from ..utils.pytorch import (
    optimizer_cuda,
    count_parameters,
    compute_gradient_norm,
    compute_weight_norm,
    sync_networks,
    sync_grads,
    to_tensor,
    scale_dict_tensor,
)


class HierarchicalBCPPOAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space, env_ob_space):
        super().__init__(config, ob_space)
        opts = config.options
        self.device = "cuda"
        low_paths = config.low_ckpt_paths
        high_ac_space = SpaceDict({"option": Discrete(len(opts))})
        self.high_actor = Actor(
            config,
            ob_space,
            high_ac_space,               # ← now .spaces is a dict
            config.tanh_policy
        )
        high_ac_space = SpaceDict({"option": Discrete(len(opts))})
        self.high_actor = Actor(config, ob_space, high_ac_space, config.tanh_policy)
        self.high_old   = Actor(config, ob_space, high_ac_space, config.tanh_policy)
        # self.high_actor  = Actor(config, ob_space, Discrete(len(opts)), config.tanh_policy)
        # self.high_old    = Actor(config, ob_space, Discrete(len(opts)), config.tanh_policy)
        self.high_critic = Critic(config, ob_space)
        self.low_actors  = {o: Actor(config, ob_space, ac_space, config.tanh_policy) for o in opts}
        for o, p in zip(opts, low_paths):
            ck = torch.load(p, map_location=config.device)
            # if Trainer.save_ckpt wrapped your agent weights under ck["agent"]
            if "actor_state_dict" in ck:
                sd = ck
            elif "agent" in ck and isinstance(ck["agent"], dict) and "actor_state_dict" in ck["agent"]:
                sd = ck["agent"]
            else:
                raise KeyError(
                    f"No 'actor_state_dict' found in checkpoint {p}. "
                    f"Found keys: {list(ck.keys())}"
                )
            self.low_actors[o].load_state_dict(sd["actor_state_dict"])
            # ensure evaluation mode so no gradients or dropout
            self.low_actors[o].eval()

        self._to_cuda(config.device)
        self.hopt = optim.Adam(self.high_actor.parameters(),  lr=config.actor_lr)
        self.hcrt = optim.Adam(self.high_critic.parameters(), lr=config.critic_lr)
        self.hsch = optim.lr_scheduler.StepLR(
            self.hopt,
            step_size=config.max_global_step//config.rollout_length//5,
            gamma=0.5,
        )
        self.csch = optim.lr_scheduler.StepLR(
            self.hcrt,
            step_size=config.max_global_step//config.rollout_length//5,
            gamma=0.5,
        )
        self.buf = ReplayBuffer(
            ["ob","opt","rew","done","ret","adv"],
            config.rollout_length,
            RandomSampler().sample_func,
        )
        self.opts = opts

    def _to_cuda(self, d):
        self.high_actor.to(d)
        self.high_old.to(d)
        self.high_critic.to(d)
        for m in self.low_actors.values():
            m.to(d)

    def sync_networks(self):
        sync_networks(self.high_actor)
        sync_networks(self.high_critic)

    def act(self, ob, is_train=True):
        # 1. Clean & batch your observations
        clean_ob = {}
        for k, v in ob.items():
            t = torch.as_tensor(v, dtype=torch.float32, device=self._config.device)
            if t.dim() == 0:
                t = t.unsqueeze(0)
            elif t.dim() == 1:
                t = t.unsqueeze(0)
            clean_ob[k] = t

        # 2. Sample high-level option (returns an OrderedDict like {'option': tensor([option_index])})
        opt_dict, _, _, _ = self.high_actor.act(
            clean_ob,
            deterministic=not is_train
        )

        # 3. Get the plain Python int option index from the tensor
        opt_tensor = next(iter(opt_dict.values()))
        opt_idx = int(opt_tensor.item())

        # 4. Dispatch to the chosen low-level actor using the option index
        low_actor = self.low_actors[self.opts[opt_idx]]
        # low_act_dict contains the low-level action dictionary, e.g., {'action': tensor([...])}
        low_act_dict, _, _, _ = low_actor.act(clean_ob)

        # 5. Squeeze away the batch dimension and move low-level action to NumPy
        action = {
            key: tensor.detach().cpu().numpy().squeeze(0)
            for key, tensor in low_act_dict.items()
        }

        # 6. **Return** the action dictionary and an info dictionary including the option index under the key 'opt'
        # The RolloutRunner should handle merging this info into the rollout dictionary.
        info = {'opt': opt_idx} # Package the option index in an info dictionary with the key 'opt'

        return action, info # Return the action dictionary and the info dictionary

    def store_episode(self, rollouts):
        self._compute_gae(rollouts)
        self.buf.store_episode(rollouts)

    def _compute_gae(self, r):
        T = len(r["done"])
        ob  = obs2tensor(self.normalize(r["ob"]), self._config.device)
        last= obs2tensor(self.normalize(r["ob_next"][-1:]), self._config.device)
        v    = self.high_critic(ob).detach().cpu().numpy()[:,0]
        vl   = self.high_critic(last).detach().cpu().numpy()[:,0]
        vpred=np.append(v,vl)
        adv = np.zeros(T,dtype="float32")
        lg = 0
        for t in reversed(range(T)):
            nt = 1-r["done"][t]
            d  = r["rew"][t] + self._config.rl_discount_factor*vpred[t+1]*nt - vpred[t]
            adv[t] = lg = d + self._config.rl_discount_factor*self._config.gae_lambda*nt*lg
        r["adv"] = adv.tolist()
        r["ret"] = (adv + vpred[:-1]).tolist()

    def train(self):
        info = Info()
        self.hsch.step()
        self.csch.step()
        self._copy_target_network(self.high_old, self.high_actor)
        nb = self._config.ppo_epoch*self._config.rollout_length//self._config.batch_size
        for _ in range(nb):
            b = self.buf.sample(self._config.batch_size)
            info.add(self._update(b))
        self.buf.clear()
        info["high_grad_norm"] = compute_gradient_norm(self.high_actor)
        return info.get_dict(only_scalar=True)

    def _update(self, tr):
        info = Info()
        ob  = to_tensor(self.normalize(tr["ob"]),  self._config.device)
        opt = to_tensor(tr["opt"],          self._config.device).long()
        ret = to_tensor(tr["ret"],          self._config.device).unsqueeze(1)
        adv = to_tensor(tr["adv"],          self._config.device).unsqueeze(1)
        _,_,ol,_ = self.high_old.act(ob, activations=opt, return_log_prob=True)
        _,_,nl,ent = self.high_actor.act(ob, activations=opt, return_log_prob=True)
        r = torch.exp(nl-ol)
        l1= torch.min(r*adv, torch.clamp(r,1-self._config.ppo_clip,1+self._config.ppo_clip)*adv).mean()
        hap = -l1 - self._config.entropy_loss_coeff*ent.mean()
        val = self.high_critic(ob)
        hv  = self._config.value_loss_coeff*(ret-val).pow(2).mean()
        self.hopt.zero_grad(); hap.backward(); sync_grads(self.high_actor); self.hopt.step()
        self.hcrt.zero_grad(); hv.backward(); sync_grads(self.high_critic); self.hcrt.step()
        info["high_actor_loss"]=hap.item()
        info["high_value_loss"]=hv.item()
        return mpi_average(info.get_dict(only_scalar=True))
    
    
if __name__ == "__main__":
    print("Running Module directly.")
