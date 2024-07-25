import numpy as np
import torch as th
import gymnasium
import copy
import random
from buffers import OnPolicyBuffer
from policies import GaussianMLPPolicy
from utils import (multiply_and_sum_tensors, multiply_elements, multiply_tensors_in_place, subtract_lists_of_tensors,
                   add_lists_of_tensors, save_data_to_file, compute_gradient_norms, clip_log_probs, clip_gradients,
                   save_statistics)


class FEDSVRPG_M():
    def __init__(
            self,
            policy: th.nn.Module,
            env: gymnasium.Env,
            u_r, # Last global gradient estimate
            gamma, # Discount factor
            global_iter_num: int,
            name,
            beta: float = 0.2, # Momentum parameter
            K: int = 10, # Number of local iterations
            nu: float = 1e-2, # Local step-size
            max_episode_length: int = 2000,
            DR = False, # Domain Randomization
            DR_episode_th: float = 0.2,
            DR_step_th: float = 0.2,
            mass_range: list = [0.027, 0.027],
            wind_range: float = 0
    ):
        self.policy = policy # Not explicitely used
        self.local_policy = copy.deepcopy(self.policy)
        self.global_policy = copy.deepcopy(self.policy)
        self.env = env
        self.u_r = u_r
        self.gamma = gamma
        self.global_iter_num = global_iter_num
        self.agent_name = name
        self.beta = beta
        self.K = K
        self.nu = nu
        self.max_episode_length = max_episode_length
        self.global_rollout_buffer = OnPolicyBuffer(self.gamma)
        self.local_rollout_buffer = OnPolicyBuffer(self.gamma)
        self.DR = DR
        self.DR_episode_th = DR_episode_th
        self.DR_step_th = DR_step_th
        self.steps = 0
        self.mass_range = mass_range,
        self.wind_range = wind_range

        self.mean_rewards = []


    def train(self) -> None:
        global_log_probs, global_probs, global_returns, global_actions, global_states, global_rewards, global_dones = self.global_rollout_buffer.get(
            'log_probs', 'probs', 'returns', 'actions', 'states', 'rewards', 'dones')
        local_log_probs, local_probs, local_returns, local_actions, local_states, local_rewards, local_dones = self.local_rollout_buffer.get(
            'log_probs', 'probs', 'returns', 'actions', 'states', 'rewards', 'dones')
        
        
        save_data_to_file(global_log_probs, global_probs, global_returns, global_actions, 
                          global_states, global_rewards, global_dones, "global_data", self.iteration)
        save_data_to_file(local_log_probs, local_probs, local_returns, local_actions, 
                          local_states, local_rewards, local_dones, "local_data", self.iteration)
        
        # Calculate importance sampling weight
        glp_total, llp_total = sum(global_log_probs), sum(local_log_probs)
        log_weight = glp_total - llp_total
        IS_weight = th.exp(log_weight).clip(1e-3, 1.8)
        print("Importance sampling weight:", IS_weight.item(), end='\n\n')

        # Calculate gradient estimates of each policy
        lp_grads = [th.autograd.grad(outputs=pi, inputs=self.local_policy.parameters(), grad_outputs=th.ones_like(pi)) for pi in local_log_probs]
        gp_grads = [th.autograd.grad(outputs=pi, inputs=self.global_policy.parameters(), grad_outputs=th.ones_like(pi)) for pi in global_log_probs]
        g_curr_policy = multiply_and_sum_tensors(scalar_tensor=local_returns, tensor_lists=lp_grads)
        g_global_policy = multiply_and_sum_tensors(scalar_tensor=global_returns, tensor_lists=gp_grads)

        # Adaptive learning rate using IS_weight
        self.steps += 1
        eta_t = self.nu / ((IS_weight + 0.03 * self.steps) ** (1 / 3))

        # Adjust gradients with the adaptive learning rate
        weighted_g_global = multiply_tensors_in_place(g_global_policy, scalar=IS_weight)
        momentum_term = subtract_lists_of_tensors(add_lists_of_tensors(self.u_r, g_curr_policy), weighted_g_global)
        u = [self.beta * gc + (1 - self.beta) * mt for gc, mt in zip(g_curr_policy, momentum_term)]
        u = add_lists_of_tensors(
            list1=multiply_tensors_in_place(g_curr_policy, scalar=self.beta), 
            list2=multiply_tensors_in_place(momentum_term, scalar=1 - self.beta)
        )

        clip_gradients(u)

        with th.no_grad():
            for param, grad in zip(self.local_policy.parameters(), u):
                param.data.add_(eta_t * grad)

            # Optionally apply weight decay
            for param in self.local_policy.parameters():
                param.data.sub_(1e-4 * param.data)

    
    def calculate_g(self,
        policy_params,
        log_probs: list[th.Tensor],
        returns: th.Tensor) -> list[th.Tensor]:
            grads = []
            for pi in log_probs:
                grad_tuple = th.autograd.grad(outputs=pi, inputs=policy_params, grad_outputs=th.ones_like(pi))
                grads.append(grad_tuple)
            return multiply_and_sum_tensors(scalar_tensor=returns, tensor_lists=grads)
                
    def sample_trajectory(self,
        rollout_buffer: OnPolicyBuffer,
        policy: th.nn.Module = GaussianMLPPolicy,
        display: bool = True
        ):

        steps = 0
        rollout_buffer.reset()
        state, _ = self.env.reset()
        means = []
        stds = []

        while steps < self.max_episode_length:
            if self.DR_episode:
                if random.random() < self.DR_step_th:
                    wind = np.random.uniform(low = 0, high = self.wind_range, size = 3)
                    self.env.changeWind(wind)

            state_tensor = th.FloatTensor(state)
            action, mean, std = policy.get_action(state_tensor)
            log_prob = policy.get_log_prob(state_tensor, action)
            action_prob = np.exp(log_prob.detach().numpy())
            next_state, reward, done, _, _ = self.env.step(action.detach().numpy())

            rollout_buffer.store(state, action, reward, action_prob, log_prob, done)
            means.append(mean)
            stds.append(std)
            state = next_state
            steps += 1

            if done:
                break
        
        # Compute returns for the stored trajectory
        rollout_buffer.compute_returns()
        
        if display:
            episode_length = len(rollout_buffer.states)

    def learn(
    self,
    ):
        self.iteration = 0
        self.delta = []

        # Run through local iterations
        while self.iteration < self.K:
            if self.DR:
                self.DR_episode = True if random.random() < self.DR_episode_th else False
                if self.DR_episode:
                    if type(self.mass_range) == tuple:
                        self.mass_range = self.mass_range[0]
                    mass = np.random.uniform(low = self.mass_range[0], high = self.mass_range[1])
                    print("Episode has activated Domain Randomization. Wind will be applied with a probability of " + str(self.DR_step_th)
                       + " with a maximum magnitude of " + str(self.wind_range) + " Newtons.")
                    print("Mass parameter for next episode: " + str(mass))
                    self.env.changeMass(mass)
            # Get trajectory from global policy
            print("GLOBAL ITERATION: " + str(self.global_iter_num))
            print("LOCAL ITERATION: " + str(self.iteration), end='\n\n')
            self.sample_trajectory(self.global_rollout_buffer, policy = self.global_policy, display = True)
            self.sample_trajectory(self.local_rollout_buffer, policy = self.local_policy, display = True)
            episode_reward = self.local_rollout_buffer.returns[0]
            print("Episode Reward:", episode_reward)
            self.mean_rewards.append(episode_reward)
            print()

            self.iteration += 1

            # Update local policy
            self.train()

            # Clear buffers
            self.global_rollout_buffer.reset()
            self.local_rollout_buffer.reset()

        # Generate delta parameter to send to server
        iterator = 0 # Skip log_std layers
        for param1, param2 in zip(self.local_policy.parameters(), self.global_policy.parameters()):
            if iterator > 1:
                self.delta.append(param1 - param2)
            iterator += 1

        return self
    
    def get_model(self):
        return self.local_policy
    
    def get_rewards(self):
        return self.mean_rewards
            
    def update_attributes(self, new_u, new_global_iter, local_policy, global_policy):
        log_std_params = list(self.local_policy.parameters())
        head = log_std_params[:2]
        self.u_r = head + new_u
        self.global_iter_num = new_global_iter
        self.local_policy = local_policy
        self.global_policy = global_policy
        self.mean_rewards = []
        '''
        print("UPDATE ATTRIBUTES PARAM CHECK:")
        print([param for param in self.local_policy.parameters()])
        '''
        self.global_iter_num += 1