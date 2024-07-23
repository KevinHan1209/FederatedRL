import numpy as np
import torch as th
import gymnasium
from policies import GaussianMLPPolicy
from algorithms.fedsvrpg_m import FEDSVRPG_M
from typing import List, Any
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.env_util import make_vec_env
from utils import get_ActorCritic_delta

class Federated_RL():
    def __init__(
            self,
            policy: th.nn.Module,
            envs: List[gymnasium.Env],
            env_kwargs: List[dict],
            num_agents: int,
            global_iterations: int,
            value_net_aggregation = False,
            callbacks: List,
            state_size: int,
            action_sizeLint,
            discount_factors: Any = .99,
            local_iterations: Any = 10,
            global_step_size: float = 1e-2,
            local_step_size: float = 1e-2,
            max_episode_length: int = 2000,
            agent_names: List = None,
            DR: List = None,
            algorithms: List = None,

    ):
        # Preprocess input policy module
        paramList = [param for param in policy.parameters()]
        paramList[0] = paramList[0].unsqueeze(dim=0)
        paramList[1] = paramList[1].unsqueeze(dim=0)
        self.policy_params = [name for name, param in policy.named_parameters()][2:]
        self.value_params = []
        self.envs = envs
        self.num_agents = num_agents
        self.global_iterations = global_iterations
        self.discount_factors = discount_factors if type(discount_factors) == list else [discount_factors for _ in range(self.num_agents)]
        self.K = local_iterations
        self.global_step_size = global_step_size
        self.nu = local_step_size
        self.max_episode_length = max_episode_length
        self.DR = DR if DR is not None else [False for i in range(self.num_agents)]
        self.agent_names = agent_names if agent_names is not None else [str(i) for i in range(1, self.num_agents + 1)]
        self.algorithms = algorithms
        self.callbacks = callbacks
        self.value_net_aggregation = value_net_aggregation
        self.state_size = state_size
        self.action_size = action_size

        self.iter_num = 0
        self.models = []

    def train(self):
        self.policy_deltas = []
        self.value_deltas = []
        self.action_weights = []

        # Train local policies
        for model, classification, callback, algo in zip(self.models, self.classifiers, self.callbacks, self.algorithms):
            if classification == "NonSB3":
                model.learn()
                self.policy_deltas.append(model.delta)
                self.value_deltas.append([])
                self.action_weights.append([])
            elif classification == "SB3":
                model.learn(total_timesteps=int(1e7), 
                callback=callback,
                log_interval=100)
                if algo == "PPO":
                    params = model.get_parameters()['policy']
                    PPO_policy_params = [params[param] for param in params if 'mlp' and 'policy' in param or 'action' in param]
                    PPO_value_params = [params[param] for param in params if 'value' in param]
                    PPO_dp, PPO_dv = get_ActorCritic_delta(self.policy_params, self.value_params, PPO_policy_params, PPO_value_params)
                    self.policy_deltas.append(PPO_dp)
                    self.value_deltas.append(PPO_dv)
                    self.action_weights.append([])
                if algo == "SAC":
                    params = model.get_parameters()['policy']
                    SAC_policy_params = [param for param in orig_params if 'actor.mu' in param]
                    SAC_critic_params1 = [param for param in orig_params if 'critic.qf0' in param]
                    SAC_critic_params2 = [param for param in orig_params if 'critic.qf1' in param]
                    SAC_critic_params =[(params[par1] + params[par2]) / 2 for par1, par2 in zip(p1, p2)]
                    # SAC uses state-action value estimation, so we must reduce action dimensions from the end
                    first_tensor = SAC_critic_params[0]
                    state_tensor = first_tensor[:, :-self.action_size]  # Shape: [32, 27]
                    action_tensor = first_tensor[:, -self.action_size:]  # Shape: [32, action_dim]
                    SAC_critic_params[0] = 
                    SAC_dp, SAC_dv = get_ActorCritic_delta(self.policy_params, self.value_params, SAC_policy_params, )



                    

        # Aggregate local policies and update global policy
        self.update_global()


    def update_global(self):

        # Ensure each model's delta is a list of tensors
        policy_deltas =self.policy_deltas
        
        # Transpose the list of lists to get a list of tensors at each parameter index
        transposed_deltas = list(zip(*model_deltas))
        
        # Sum the deltas for each parameter
        total_deltas = []
        for deltas in transposed_deltas:
            total_delta = np.sum([delta.detach().numpy() for delta in deltas], axis=0)
            total_deltas.append(total_delta)

        # Calculate global gradient and convert back to tensors
        global_grads = [th.tensor(total_delta / (self.nu * self.num_agents * self.K), dtype=delta.dtype) 
                        for total_delta, delta in zip(total_deltas, self.models[0].delta)]
        
        self.u_r = global_grads

        # Perform the gradient update
        with th.no_grad():
            for param, grad in zip(self.policy.parameters(), global_grads):
                param.add_(self.global_step_size * grad)
        
        if self.value_net_aggregation:
            value_deltas = self.value_deltas
                # Transpose the list of lists to get a list of tensors at each parameter index
            transposed_deltas = list(zip(*model_deltas))
            
            # Sum the deltas for each parameter
            total_deltas = []
            for deltas in transposed_deltas:
                total_delta = np.sum([delta.detach().numpy() for delta in deltas], axis=0)
                total_deltas.append(total_delta)

            # Calculate global gradient and convert back to tensors
            global_grads = [th.tensor(total_delta / (self.nu * self.num_agents * self.K), dtype=delta.dtype) 
                            for total_delta, delta in zip(total_deltas, self.models[0].delta)]
            
            self.u_r = global_grads

            # Perform the gradient update
            with th.no_grad():
                for param, grad in zip(self.policy.parameters(), global_grads):
                    param.add_(self.global_step_size * grad)

        # Refresh policy differences
        self.policy_deltas = []
        self.value_deltas = []


    def generate_models(self):
        self.classifiers = []
        for agent, (name, env, env_kwargs, gamma, algo) in enumerate(zip(self.agent_names, self.envs, self.env_kwargs, self.discount_factors, self.algorithms)):
            if algo == 'FedSVRPG-M':
                model_instance = FEDSVRPG_M(
                    policy=self.policy,
                    env=env,
                    u_r=self.u_r,
                    global_iter_num = self.iter_num,
                    name = name,
                    beta=0.2,
                    K = self.K,
                    nu=self.nu,
                    max_episode_length=self.max_episode_length,
                    gamma=gamma,
                    DR= self.DR,
                )
                self.classifiers.append('NonSB3')
            elif algo == 'PPO':
                model_instance = PPO(
                    'MlpPolicy',
                    env=make_vec_env(env, env_kwargs=env_kwargs, n_envs=1),
                    # tensorboard_log=filename+'/tb/',
                    verbose=1
                )
                self.classifiers.append('SB3')
            elif algo == 'SAC':
                model_instance = SAC(
                    'MlpPolicy',
                    env=make_vec_env(env, env_kwargs=env_kwargs, n_envs=1),
                    verbose=1
                )
                self.classifiers.append('NonSB3')
            elif algo == 'TD3':
                model_instance = TD3(
                    'MlpPolicy',
                    env=make_vec_env(env, env_kwargs=env_kwargs, n_envs=1),
                    verbose=1
                )
                self.classifiers.append('NonSB3')
            elif algo == 'A2C':
                model_instance = A2C(
                    'MlpPolicy',
                    env = make_vec_env(env, env_kwargs=env_kwargs, n_envs=1),
                    verbose=1
                )
                self.classifiers.append('NonSB3')
            else:
                raise ValueError("Invalid Algorithm:", algo)
            
            self.models.append(model_instance)


    def update_models(self):
        for i in self.models:

            
    def learn(self):
        for r in range(self.global_iterations):
            if r == 0:
                # Initialize gradient estimate
                self.u_r = [th.zeros_like(param) for param in self.policy.parameters()]

            self.generate_models()
            self.train()
            self.update_models()
            self.iter_num += 1

        return self
    

