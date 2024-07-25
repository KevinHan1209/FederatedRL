import numpy as np
import torch as th
import gymnasium
from policies import GaussianMLPPolicy, ValueFunctionNetwork
from algorithms.fedsvrpg_m import FEDSVRPG_M
from typing import List, Any
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import VecCheckNan
from utils import (get_ActorCritic_delta, add_lists_of_tensors, multiply_tensors_in_place, replace_torch_parameters,
                   PPO_policy_update, SAC_policy_update, TD3_policy_update)
from evaluations import evaluate_local_policy
from gym_pybullet_drones.envs import HoverAviary
import copy

th.autograd.set_detect_anomaly(True)

class Federated_RL():
    def __init__(
            self,
            policy: th.nn.Module,
            envs: List[gymnasium.Env],
            env_kwargs: List[dict],
            num_agents: int,
            global_iterations: int,
            state_size: int,
            action_size: int,
            policy_kwargs: dict,
            critic_net_aggregation = False,
            critic_net: List = [32,32],
            discount_factors: Any = .99,
            local_iterations: Any = 10,
            global_step_size: float = 1e-3,
            local_step_size: float = 1e-2,
            max_episode_length: int = 2000,
            eval_episodes = 25,
            agent_names: List = None,
            DR: List = None,
            DR_episode_th: List = None,
            DR_step_th: List = None,
            algorithms: List = None,
            mass_ranges: List = None,
            wind_ranges: List = None

    ):
        self.policy = policy
        # Preprocess input policy module
        self.policy_params = [param for name, param in policy.named_parameters()][2:]
        # Preprocess value net
        if critic_net_aggregation:
            init_value_net = ValueFunctionNetwork(input_size=state_size, output_size=1, layer_sizes=critic_net)
        self.value_params = [param for name, param in init_value_net.named_parameters()]

        self.envs = envs
        self.env_kwargs = env_kwargs
        self.policy_kwargs = policy_kwargs # For sb3 policies
        self.critic_layers = critic_net
        self.num_agents = num_agents
        self.global_iterations = global_iterations
        self.discount_factors = discount_factors if type(discount_factors) == list else [discount_factors for _ in range(self.num_agents)]
        self.K = local_iterations
        self.global_step_size = global_step_size
        self.nu = local_step_size
        self.max_episode_length = max_episode_length
        self.DR = DR if DR is not None else [False for i in range(self.num_agents)]
        self.DR_episode_th = DR_episode_th
        self.DR_step_th = DR_step_th
        self.agent_names = agent_names if agent_names is not None else [str(i) for i in range(1, self.num_agents + 1)]
        self.algorithms = algorithms
        self.critic_net_aggregation = critic_net_aggregation
        self.state_size = state_size
        self.action_size = action_size
        self.eval_episodes = eval_episodes
        # Specific to drones environment
        self.mass_ranges = mass_ranges
        self.wind_ranges = wind_ranges

        self.iter_num = 0
        self.models = []
        self.total_rewards = []
        

    def train(self):
        self.policy_deltas = []
        self.value_deltas = []
        self.action_weights = []
        self.callbacks = []
        # Create callbacks for sb3 algorithms
        for classification, env, env_kwargs, algo, dr, dre_th, drs_th, mr, wr in zip(self.classifiers, self.envs, self.env_kwargs,self.algorithms, self.DR,
                                                                             self.DR_episode_th, self.DR_step_th, self.mass_ranges, self.wind_ranges):
            if classification == "SB3":
                # Training stop based on iteration not target reward
                callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=1e7,
                                        verbose=1)
                
                eval_callback = EvalCallback(env(**env_kwargs),
                                        callback_on_new_best=callback_on_best,
                                        verbose=1,
                                        best_model_save_path=algo+'/',
                                        log_path=algo+'/',
                                        eval_freq=self.max_episode_length,
                                        deterministic=True,
                                        DR=dr,
                                        DR_episode_th=dre_th,
                                        DR_step_th=drs_th,
                                        mass_range = mr,
                                        wind_range = wr,
                                        render=False)
                self.callbacks.append(eval_callback)
            else:
                self.callbacks.append(0) # Other algorithms don't use callbacks

        # Train local policies
        for model, classification, callback, algo, agent in zip(self.models, self.classifiers, self.callbacks, self.algorithms, self.agent_names):
            print()
            print("Training agent", agent, end = '\n\n')
            if classification == "NonSB3":
                model.learn()
                episode_mean_rewards = model.get_rewards()
                self.total_rewards.append(episode_mean_rewards)
                self.policy_deltas.append(model.delta)
                self.value_deltas.append(0)
                self.action_weights.append(0)
            elif classification == "SB3":
                model.learn(total_timesteps=int(1e7), 
                callback=callback,
                log_interval=self.max_episode_length)
                episode_mean_rewards = callback.get_mean_rewards()
                callback.refresh_rewards()
                self.total_rewards.append(episode_mean_rewards)
                if algo == "PPO":
                    params = model.get_parameters()['policy']
                    PPO_policy_params = [params[param] for param in params if 'mlp' and 'policy' in param or 'action' in param]
                    PPO_value_params = [params[param] for param in params if 'value' in param]
                    PPO_dp, PPO_dv = get_ActorCritic_delta(self.policy_params, self.value_params, PPO_policy_params, PPO_value_params)
                    self.policy_deltas.append(PPO_dp)
                    self.value_deltas.append(PPO_dv)
                    self.action_weights.append(0)
                if algo == "SAC":
                    params = model.get_parameters()['policy']
                    SAC_policy_params = [params[param] for param in params if 'actor.mu' in param or 'actor.latent' in param]
                    SAC_critic_params1 = [param for param in params if 'critic.qf0' in param]
                    SAC_critic_params2 = [param for param in params if 'critic.qf1' in param]
                    SAC_critic_params = [(params[par1] + params[par2]) / 2 for par1, par2 in zip(SAC_critic_params1, SAC_critic_params2)] # Consider average of 2 Q networks
                    # SAC uses state-action value estimation, so we must reduce action dimensions from the end
                    first_tensor = SAC_critic_params[0]
                    state_tensor, action_tensor = first_tensor[:, :-self.action_size], first_tensor[:, -self.action_size:] 
                    SAC_critic_params[0] = state_tensor
                    SAC_dp, SAC_dv = get_ActorCritic_delta(self.policy_params, self.value_params, SAC_policy_params, SAC_critic_params)
                    self.policy_deltas.append(SAC_dp)
                    self.value_deltas.append(SAC_dv)
                    self.action_weights.append(action_tensor)
                if algo == "TD3":
                    params = model.get_parameters()['policy']
                    TD3_policy_params = [params[param] for param in params if 'actor.mu' in param]
                    TD3_critic_params1 = [param for param in params if 'critic.qf0' in param]
                    TD3_critic_params2 = [param for param in params if 'critic.qf1' in param]
                    TD3_critic_params = [(params[par1] + params[par2]) / 2 for par1, par2 in zip(TD3_critic_params1, TD3_critic_params2)] # Consider average of 2 Q networks
                    # TD3 uses state-action value estimation, so we must reduce action dimensions from the end
                    first_tensor = TD3_critic_params[0]
                    state_tensor, action_tensor = first_tensor[:, :-self.action_size], first_tensor[:, -self.action_size:]
                    TD3_critic_params[0] = state_tensor
                    TD3_dp, TD3_dv = get_ActorCritic_delta(self.policy_params, self.value_params, TD3_policy_params, TD3_critic_params)
                    self.policy_deltas.append(TD3_dp)
                    self.value_deltas.append(TD3_dv)
                    self.action_weights.append(action_tensor)
                

        # Evaluate algorithms for weighted ensemble
        self.evaluate_policies()

        # Aggregate local policies and update global policy
        self.update_global()

    def evaluate_policies(self):
        print()
        print('Evaluating agents...', end='\n\n')
        self.policy_rewards = []
        for model, classification, env, env_kwargs, gamma, agent in zip(self.models, self.classifiers, self.envs, self.env_kwargs, self.discount_factors, self.agent_names):
            if classification == "SB3":
                mean_reward, std_reward = evaluate_policy(model,
                                              env=make_vec_env(env, env_kwargs=env_kwargs, n_envs=1),
                                              n_eval_episodes=self.eval_episodes
                                              )
                self.policy_rewards.append(mean_reward)
            if classification == "NonSB3":
                mean_reward = evaluate_local_policy(env=env, env_kwargs=env_kwargs, model=model, eval_episodes=self.eval_episodes,
                                                    max_episode_length=self.eval_episodes, gamma=gamma)
                self.policy_rewards.append(mean_reward)
        print("Mean rewards for the following agents:")
        for agent, mean_reward in zip(self.agent_names, self.policy_rewards):
            print(agent + ": " + str(mean_reward))
        

                



    def update_global(self):

        def zeros_like(tensor_list):
            return [th.zeros_like(tensor) for tensor in tensor_list]
        '''
        print("Policy delta")
        print([param for param in self.policy_deltas])
        print("Value delta")
        print([param for param in self.value_deltas])
        print("Action weight")
        print(self.action_weights)
        '''
        # Initialize gradients
        policy_grad, value_grad, self.action_update = zeros_like(self.policy_params), zeros_like(self.value_params), th.zeros_like(th.randn(self.critic_layers[0], self.action_size))
        # Initialize total rewards for each respective parameters
        policy_total_rewards, value_total_rewards, action_total_rewards = 0, 0, 0
        # Get total amount of agents which use each respective parameters
        self.total_policy_agents = sum(1 for x in self.policy_deltas if x != 0)
        self.total_critic_agents = sum(1 for x in self.value_deltas if x != 0)
        # Calculate gradients
        for dp, dv, da, weight in zip(self.policy_deltas, self.value_deltas, self.action_weights, self.policy_rewards):
            if dp != 0:
                policy_grad = add_lists_of_tensors(policy_grad, multiply_tensors_in_place(dp, weight))
                policy_total_rewards += weight
            if dv != 0:
                value_grad = add_lists_of_tensors(value_grad, multiply_tensors_in_place(dv, weight))
                value_total_rewards += weight
            if type(da) != int:
                self.action_update += da * weight
                action_total_rewards += weight

        self.policy_grad = [param/self.total_policy_agents for param in policy_grad]
        if self.total_critic_agents != 0:
            value_grad = [param/self.total_critic_agents for param in value_grad]
        else:
            value_grad = [param for param in value_grad]
        self.action_update /= action_total_rewards

        # Perform the gradient update
        self.policy_params = [param + self.global_step_size * grad for param, grad in zip(self.policy_params, self.policy_grad)]
        self.value_params = [param + self.global_step_size * grad for param, grad in zip(self.value_params, value_grad)]

        self.update_models()

        # Refresh policy differences
        self.policy_deltas = []
        self.value_deltas = []


    def generate_models(self):
        self.classifiers = []

        for agent, (name, env, env_kwargs, gamma, algo, dr, dre_th, drs_th, mr, wr) in enumerate(
            zip(self.agent_names, self.envs, self.env_kwargs, self.discount_factors, self.algorithms, self.DR,
                self.DR_episode_th, self.DR_step_th, self.mass_ranges, self.wind_ranges)):

            if algo == 'FedSVRPG-M':
                model_instance = FEDSVRPG_M(
                    policy=self.policy,
                    env=env(**env_kwargs),
                    u_r=[th.zeros_like(param) for param in self.policy.parameters()],
                    global_iter_num = self.iter_num,
                    name = name,
                    DR=dr,
                    beta=0.2,
                    K = self.K,
                    nu=self.nu,
                    max_episode_length=self.max_episode_length,
                    gamma=gamma,
                    DR_episode_th = dre_th,
                    DR_step_th = drs_th,
                    mass_range = mr,
                    wind_range = wr
                )
                self.classifiers.append('NonSB3')
            elif algo == 'PPO':
                if 'use_sde' in self.policy_kwargs:
                    del self.policy_kwargs['use_sde']
                # PPO has different keyword argument due to being an on-policy algorithm
                on_policy_kwargs = copy.deepcopy(self.policy_kwargs)
                # Change 'qf' to 'vf' in the copy
                if 'net_arch' in on_policy_kwargs and 'qf' in on_policy_kwargs['net_arch']:
                    on_policy_kwargs['net_arch']['vf'] = on_policy_kwargs['net_arch'].pop('qf')
                model_instance = PPO(
                    'MlpPolicy',
                    env=make_vec_env(env, env_kwargs=env_kwargs, n_envs=1),
                    policy_kwargs=on_policy_kwargs,
                    learning_rate = self.nu,
                    n_steps=self.max_episode_length,
                    local_iterations = self.K,
                    verbose=0
                )
                self.classifiers.append('SB3')
            elif algo == 'SAC':
                if 'use_sde' in self.policy_kwargs:
                    del self.policy_kwargs['use_sde']
                model_instance = SAC(
                    'MlpPolicy',
                    env=VecCheckNan(make_vec_env(env, env_kwargs=env_kwargs, n_envs=1), raise_exception=True),
                    policy_kwargs=self.policy_kwargs,
                    learning_rate = self.nu,
                    local_iterations = self.K,
                    train_freq=self.max_episode_length,
                    ent_coef=1,
                    verbose=0
                )
                self.classifiers.append('SB3')
            elif algo == 'TD3':
                if 'use_sde' in self.policy_kwargs:
                    del self.policy_kwargs['use_sde']
                model_instance = TD3(
                    'MlpPolicy',
                    env=make_vec_env(env, env_kwargs=env_kwargs, n_envs=1),
                    policy_kwargs=self.policy_kwargs,
                    learning_rate = self.nu,
                    local_iterations = self.K,
                    train_freq=self.max_episode_length,
                    verbose=0
                )
                self.classifiers.append('SB3')
            elif algo == 'A2C':
                model_instance = A2C(
                    'MlpPolicy',
                    env = make_vec_env(env, env_kwargs=env_kwargs, n_envs=1),
                    policy_kwargs=self.policy_kwargs,
                    learning_rate = self.nu,
                    local_iterations = self.K,
                    train_freq=self.max_episode_length,
                    verbose=0
                )
                self.classifiers.append('SB3')
            else:
                raise ValueError("Invalid Algorithm:", algo)
            
            self.models.append(model_instance)


    def update_models(self):
        # Post-process new parameters back into respective models
        for model, algo, classification in zip(self.models, self.algorithms, self.classifiers):
            if classification == "NonSB3":
                # Convert back to torch module
                new_policy = replace_torch_parameters(model, self.policy_params, start_index=2)
                model.update_attributes(self.policy_grad, self.iter_num, new_policy, new_policy)
            elif classification == "SB3":
                if algo == "PPO":
                    new_params = PPO_policy_update(model, self.policy_params, self.value_params)
                    model.set_parameters(new_params)
                if algo == "SAC":
                    # Re-attach action tensor to first critic parameter
                    reconstructed_first_tensor = th.cat((self.value_params[0], self.action_update), dim=1)
                    new_critic_params = [param if i != 0 else reconstructed_first_tensor for i, param in enumerate(self.value_params)]
                    new_params = SAC_policy_update(model, self.policy_params, new_critic_params)
                    model.set_parameters(new_params)
                if algo == "TD3":
                    reconstructed_first_tensor = th.cat((self.value_params[0], self.action_update), dim=1)
                    new_critic_params = [param if i != 0 else reconstructed_first_tensor for i, param in enumerate(self.value_params)]
                    new_params = TD3_policy_update(model, self.policy_params, new_critic_params)
                    model.set_parameters(new_params)

            
    def learn(self):
        self.u_r = [th.zeros_like(param) for param in self.policy.parameters()]
        self.generate_models()
        for r in range(self.global_iterations):
            self.train()
            self.iter_num += 1
        return self
    
    def get_rewards(self):
        return self.total_rewards

