import numpy as np
import torch as th
import gymnasium
from policies import GaussianMLPPolicy
from algorithms.fedsvrpg_m import FEDSVRPG_M
from typing import List, Any

class Federated_RL():
    def __init__(
            self,
            policy: th.nn.Module,
            envs: List[gymnasium.Env],
            num_agents: int,
            global_iterations: int,
            discount_factors: Any = .99,
            local_iterations: Any = 10,
            global_step_size: float = 1e-2,
            local_step_size: float = 1e-2,
            max_episode_length: int = 2000,
            agent_names: List = None,
            DR: List = None
    ):
        self.policy = policy
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

        self.iter_num = 0
        self.models = []

    def train(self):

        # Train local policies
        for model in self.models:
            model.learn()

        # Aggregate local policies and update global policy
        self.update_global()


    def update_global(self):

        # Ensure each model's delta is a list of tensors
        model_deltas = [model.delta for model in self.models]
        
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


    def generate_models(self):
        for agent, (name, env, gamma) in enumerate(zip(self.agent_names, self.envs, self.discount_factors)):
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
            self.models.append(model_instance)

    def refresh_models(self):
        self.models = []
            
    def learn(self):
        for r in range(self.global_iterations):
            if r == 0:
                # Initialize gradient estimate
                self.u_r = [th.zeros_like(param) for param in self.policy.parameters()]

            self.generate_models()
            self.train()
            self.refresh_models()
            self.iter_num += 1

        return self

