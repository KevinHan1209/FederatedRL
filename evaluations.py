import numpy as np
import torch as th
import gymnasium as gym
from buffers import OnPolicyBuffer

def evaluate_local_policy(
        env: gym.Env, 
        env_kwargs: dict,
        model,
        eval_episodes: int,
        max_episode_length: int,
        gamma: int = .99
):
        env = env(**env_kwargs)
        rollout_buffer = OnPolicyBuffer(gamma=gamma)
        policy = model.get_model()
        '''
        print("EVALUATE POLICY PARAMS")
        print([param for param in policy.parameters()])
        '''
        episode_rewards = []
        for episode in range(eval_episodes):
            steps = 0
            rollout_buffer.reset()
            state, _ = env.reset() 
            

            while steps < max_episode_length:
                state_tensor = th.FloatTensor(state)
                action, mean, std = policy.get_action(state_tensor)
                log_prob = policy.get_log_prob(state_tensor, action)
                action_prob = np.exp(log_prob.detach().numpy())
                next_state, reward, done, _, _ = env.step(action.detach().numpy())

                rollout_buffer.store(state, action, reward, action_prob, log_prob, done)
                state = next_state
                steps += 1

                if done:
                    break
            rollout_buffer.compute_returns()
            episode_rewards.append(rollout_buffer.returns[0])

        return np.mean(episode_rewards)
        

        
