import torch as th
import numpy as np
import os
from typing import List, Any

def multiply_elements(input_list, scalar):
    """
    Multiply every float element in a nested list by a scalar.

    Args:
        input_list (list): List of lists of tensors where each list is of same shape, but not necessarily each tensor
        scalar (float): The scalar to multiply each float element by.

    Returns:
        list: The resulting list with all float elements multiplied by the scalar.
    """
    # Initialize the result list with zeros tensors of the same shape
    result = [th.zeros_like(tensor) for tensor in input_list[0]]

    # Perform element-wise multiplication and summation
    for sublist in input_list:
        for i, tensor in enumerate(sublist):
            result[i].add_(tensor * scalar)  # In-place addition

    return result

def multiply_tensors_in_place(tensors, scalar):
    """
    Multiply all elements in a list of tensors by a scalar in-place.

    Args:
    - tensors (list of torch.Tensor): List of tensors to be multiplied.
    - scalar (float): Scalar value to multiply each tensor by.

    Returns:
    - list of torch.Tensor: List of tensors after multiplication.
    """
    with th.no_grad():
        for tensor in tensors:
            tensor.mul_(scalar)  # In-place multiplication

    return tensors

def multiply_and_sum_tensors(scalar_tensor: th.Tensor, tensor_lists: List[List[th.Tensor]]) -> List[th.Tensor]:
    """
    Multiplies each tensor in the lists of tensors by the corresponding scalar in the tensor of scalars and sums up the tensors at the end.

    Args:
    - scalar_tensor (torch.Tensor): Tensor of scalar values.
    - tensor_lists (List[List[torch.Tensor]]): List of lists, where each inner list contains PyTorch tensors.

    Returns:
    - List[torch.Tensor]: List of summed tensors.
    """
    # Initialize a list to hold the summed tensors
    summed_tensors = [th.zeros_like(tensor) for tensor in tensor_lists[0]]

    for scalar, tensors in zip(scalar_tensor, tensor_lists):
        for i, tensor in enumerate(tensors):
            summed_tensors[i] += scalar * tensor
    
    return summed_tensors

def add_lists_of_tensors(list1, list2):
    """
    Add two lists of tensors element-wise in-place.

    Args:
    - list1 (list of torch.Tensor): First list of tensors.
    - list2 (list of torch.Tensor): Second list of tensors.

    Returns:
    - list of torch.Tensor: List of tensors after addition.
    """
    assert len(list1) == len(list2), "Lists must have the same length"
    with th.no_grad():
        for t1, t2 in zip(list1, list2):
            t1.add_(t2)  # In-place addition

    return list1

def subtract_lists_of_tensors(list1, list2):
    """
    Subtract two lists of tensors element-wise in-place.

    Args:
    - list1 (list of torch.Tensor): First list of tensors.
    - list2 (list of torch.Tensor): Second list of tensors.

    Returns:
    - list of torch.Tensor: List of tensors after subtraction.
    """
    assert len(list1) == len(list2), "Lists must have the same length"

    with th.no_grad():
        for t1, t2 in zip(list1, list2):
            t1.sub_(t2)  # In-place subtraction

    return list1



def get_shape(element):
    """
    Recursively get the shape of a nested list.

    Args:
        element (any): The element to get the shape of.

    Returns:
        tuple: The shape of the element.
    """
    if isinstance(element, list):
        if not element:
            return (0,)
        else:
            return (len(element),) + get_shape(element[0])
    return ()

def check_shapes(input_list):
    """
    Check if every element in a list has the same shape.

    Args:
        input_list (list): The nested list to check.

    Returns:
        bool: True if all elements have the same shape, False otherwise.
    """
    if not input_list:
        return True
    
    first_shape = get_shape(input_list[0])
    
    def check_shape_recursive(lst, shape):
        for element in lst:
            if isinstance(element, list):
                if get_shape(element) != shape:
                    return False
                if not check_shape_recursive(element, shape[1:]):
                    return False
            else:
                if shape:
                    return False
        return True
    
    return check_shape_recursive(input_list, first_shape)
    
def sum_tensorLists(nested_list):
    if not nested_list or not nested_list[0]:
        raise ValueError("The input list is empty or improperly structured")
    
    num_tensors = len(nested_list[0])
    
    # Initialize a list of tensors with the same shapes as the tensors in the first inner list
    summed_tensors = [th.zeros_like(nested_list[0][i]) for i in range(num_tensors)]
    
    # Iterate through the outer list
    for inner_list in nested_list:
        if len(inner_list) != num_tensors:
            raise ValueError("All inner lists must contain the same number of tensors")
        
        # Iterate through the inner list and sum the tensors
        for i, tensor in enumerate(inner_list):
            summed_tensors[i] += tensor
    
    return summed_tensors

def tensors_to_list_of_lists(tensor_list: list[th.Tensor]) -> list[list]:
    return [tensor.tolist() for tensor in tensor_list]

def compare_policies(policy1, policy2):
    # Ensure both policies are in evaluation mode
    policy1.eval()
    policy2.eval()

    # Get the state dictionaries of both policies
    state_dict1 = policy1.state_dict()
    state_dict2 = policy2.state_dict()

    # Check if both dictionaries have the same keys
    if state_dict1.keys() != state_dict2.keys():
        return False

    # Check if all parameters are equal
    for key in state_dict1.keys():
        if not th.equal(state_dict1[key], state_dict2[key]):
            return False

    return True

def save_data_to_file(log_probs, probs, returns, actions, states, rewards, dones, folder_name, iteration):
    # Convert tensors to numpy arrays with .detach() to avoid gradients
    if isinstance(log_probs[0], th.Tensor):
        log_probs_np = [lp.detach().numpy() for lp in log_probs]
    else:
        log_probs_np = [np.array(lp) for lp in log_probs]

    if isinstance(probs[0], th.Tensor):
        probs_np = [p.detach().numpy() for p in probs]
    else:
        probs_np = [np.array(p) for p in probs]

    # Ensure returns is always a tensor or a numpy array
    if isinstance(returns, th.Tensor):
        returns_np = returns.detach().numpy()
    else:
        returns_np = np.array(returns)

    # Convert tensors to numpy arrays with .detach() to avoid gradients
    if isinstance(actions[0], th.Tensor):
        actions_np = [a.detach().numpy() for a in actions]
    else:
        actions_np = [np.array(a) for a in actions]

    # Convert tensors to numpy arrays with .detach() to avoid gradients
    if isinstance(states[0], th.Tensor):
        states_np = [s.detach().numpy() for s in states]
    else:
        states_np = [np.array(s) for s in states]
    
    # Ensure rewards is always a tensor or a numpy array
    if isinstance(rewards, th.Tensor):
        rewards_np = rewards.detach().numpy()
    else:
        rewards_np = np.array(rewards)

    # Ensure dones is always a tensor or a numpy array
    if isinstance(dones, th.Tensor):
        dones_np = dones.detach().numpy()
    else:
        dones_np = np.array(dones)

    # Create the directory if it doesn't exist
    dir_path = os.path.join(folder_name, f"iteration_{iteration}")
    os.makedirs(dir_path, exist_ok=True)
    
    # Save log probabilities to CSV
    np.savetxt(os.path.join(dir_path, "log_probs.csv"), np.vstack(log_probs_np), delimiter=',', fmt='%f')

    # Save probabilities to CSV
    np.savetxt(os.path.join(dir_path, "probs.csv"), np.vstack(probs_np), delimiter=',', fmt='%f')

    # Save returns to CSV
    np.savetxt(os.path.join(dir_path, "returns.csv"), returns_np, delimiter=',', fmt='%f')

    # Save actions to CSV
    np.savetxt(os.path.join(dir_path, "actions.csv"), np.vstack(actions_np), delimiter=',', fmt='%f')

    # Save states to CSV
    np.savetxt(os.path.join(dir_path, "states.csv"), np.vstack(states_np), delimiter=',', fmt='%f')

    # Save rewards to CSV
    np.savetxt(os.path.join(dir_path, "rewards.csv"), rewards_np, delimiter=',', fmt='%f')

    # Save dones to CSV
    np.savetxt(os.path.join(dir_path, "dones.csv"), dones_np, delimiter=',', fmt='%f')

    

def compute_gradient_norms(gradients):
    gradient_norms = []
    for grad_list in gradients:
        grad_norms = []
        for grad in grad_list:
            norm = th.norm(grad).item()
            grad_norms.append(norm)
        gradient_norms.append(grad_norms)
    return gradient_norms

def clip_log_probs(log_probs, min_value=-100, max_value=0):
    return [lp.clamp(min_value, max_value) for lp in log_probs]

def clip_gradients(grads, max_norm=0.5):
    total_norm = 0.0
    for grad in grads:
        param_norm = grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for grad in grads:
            grad.data.mul_(clip_coef)

def save_statistics(means, stds, file_name='means_and_stds.txt'):
    with open(file_name, 'w') as file:
        file.write("Means:\n")
        for mean in means:
            file.write(f"{mean}\n")
        file.write("\nStds:\n")
        for std in stds:
            file.write(f"{std}\n")


def PPO_policy_update(PPO_Model, policy_net_update, value_net_update, critic_net_update):
    '''
    Sample shape:
    log_std: torch.Size([1])
    mlp_extractor.policy_net.0.weight: torch.Size([512, 27])
    mlp_extractor.policy_net.0.bias: torch.Size([512])
    mlp_extractor.policy_net.2.weight: torch.Size([512, 512])
    mlp_extractor.policy_net.2.bias: torch.Size([512])
    mlp_extractor.policy_net.4.weight: torch.Size([256, 512])
    mlp_extractor.policy_net.4.bias: torch.Size([256])
    mlp_extractor.policy_net.6.weight: torch.Size([128, 256])
    mlp_extractor.policy_net.6.bias: torch.Size([128])
    mlp_extractor.value_net.0.weight: torch.Size([32, 27])
    mlp_extractor.value_net.0.bias: torch.Size([32])
    mlp_extractor.value_net.2.weight: torch.Size([32, 32])
    mlp_extractor.value_net.2.bias: torch.Size([32])
    action_net.weight: torch.Size([1, 128])
    action_net.bias: torch.Size([1])
    value_net.weight: torch.Size([1, 32])
    value_net.bias: torch.Size([1])
    '''
    policy_params = []
    value_params = []
    holder = PPO_Model.get_parameters()
    orig_params = holder['policy']
    for i in orig_params:
        if 'mlp' in i and 'policy' in i or 'action' in i:
            policy_params.append(orig_params[i])
        if 'value' in i:
            value_params.append(orig_params[i])
    # Directly assign the new parameters to replace the old ones
    new_policy = policy_net_update
    new_value = value_net_update
    for pp, npp in zip([param for param in orig_params if ('mlp' in param and 'policy' in param) or 'action' in param], new_policy):
        orig_params[pp] = npp
    if critic_net_update:
        for vp, nvp in zip([param for param in orig_params if 'value' in param], new_value):
            orig_params[vp] = nvp
    holder['policy'] = orig_params
    return holder

def TD3_policy_update(TD3_Model, policy_net_update, value_net_update, critic_net_update):
    '''
    Sample shape:
    actor.mu.0.weight: torch.Size([512, 27])
    actor.mu.0.bias: torch.Size([512])
    actor.mu.2.weight: torch.Size([512, 512])
    actor.mu.2.bias: torch.Size([512])
    actor.mu.4.weight: torch.Size([256, 512])
    actor.mu.4.bias: torch.Size([256])
    actor.mu.6.weight: torch.Size([128, 256])
    actor.mu.6.bias: torch.Size([128])
    actor.mu.8.weight: torch.Size([1, 128])
    actor.mu.8.bias: torch.Size([1])
    actor_target.mu.0.weight: torch.Size([512, 27])
    actor_target.mu.0.bias: torch.Size([512])
    actor_target.mu.2.weight: torch.Size([512, 512])
    actor_target.mu.2.bias: torch.Size([512])
    actor_target.mu.4.weight: torch.Size([256, 512])
    actor_target.mu.4.bias: torch.Size([256])
    actor_target.mu.6.weight: torch.Size([128, 256])
    actor_target.mu.6.bias: torch.Size([128])
    actor_target.mu.8.weight: torch.Size([1, 128])
    actor_target.mu.8.bias: torch.Size([1])
    critic.qf0.0.weight: torch.Size([32, 28])
    critic.qf0.0.bias: torch.Size([32])
    critic.qf0.2.weight: torch.Size([32, 32])
    critic.qf0.2.bias: torch.Size([32])
    critic.qf0.4.weight: torch.Size([1, 32])
    critic.qf0.4.bias: torch.Size([1])
    critic.qf1.0.weight: torch.Size([32, 28])
    critic.qf1.0.bias: torch.Size([32])
    critic.qf1.2.weight: torch.Size([32, 32])
    critic.qf1.2.bias: torch.Size([32])
    critic.qf1.4.weight: torch.Size([1, 32])
    critic.qf1.4.bias: torch.Size([1])
    critic_target.qf0.0.weight: torch.Size([32, 28])
    critic_target.qf0.0.bias: torch.Size([32])
    critic_target.qf0.2.weight: torch.Size([32, 32])
    critic_target.qf0.2.bias: torch.Size([32])
    critic_target.qf0.4.weight: torch.Size([1, 32])
    critic_target.qf0.4.bias: torch.Size([1])
    critic_target.qf1.0.weight: torch.Size([32, 28])
    critic_target.qf1.0.bias: torch.Size([32])
    critic_target.qf1.2.weight: torch.Size([32, 32])
    critic_target.qf1.2.bias: torch.Size([32])
    critic_target.qf1.4.weight: torch.Size([1, 32])
    critic_target.qf1.4.bias: torch.Size([1])
    '''
    policy_params = []
    value_params = []
    policy_targets = []
    value_targets = []
    holder = TD3_Model.get_parameters()
    orig_params = holder['policy']
    for i in orig_params:
        if 'actor.mu' in i:
            policy_params.append(orig_params[i])
        if 'critic.qf' in i:
            value_params.append(orig_params[i])
        if 'actor_target' in i:
            policy_targets.append(orig_params[i])
        if 'critic_target' in i:
            value_targets.append(orig_params[i])
    # Directly assign the new parameters to replace the old ones
    new_policy = policy_net_update
    new_value = value_net_update
    for pp, pt, npp in zip([param for param in orig_params if 'actor.mu' in param],
                           [param for param in orig_params if 'actor_target' in param], new_policy):
        orig_params[pp] = npp
        orig_params[pt] = npp
    if critic_net_update:
        for vp, vt, nvp in zip([param for param in orig_params if 'critic.qf' in param],
                            [param for param in orig_params if 'critic_target' in param], new_value):
            orig_params[vp] = nvp
            orig_params[vt] = nvp
    holder['policy'] = orig_params
    return holder

def SAC_policy_update(TD3_Model, policy_net_update, value_net_update, critic_net_update):
    '''
    Sample shape:
    actor.latent_pi.0.weight: torch.Size([512, 27])
    actor.latent_pi.0.bias: torch.Size([512])
    actor.latent_pi.2.weight: torch.Size([512, 512])
    actor.latent_pi.2.bias: torch.Size([512])
    actor.latent_pi.4.weight: torch.Size([256, 512])
    actor.latent_pi.4.bias: torch.Size([256])
    actor.latent_pi.6.weight: torch.Size([128, 256])
    actor.latent_pi.6.bias: torch.Size([128])
    actor.mu.weight: torch.Size([1, 128])
    actor.mu.bias: torch.Size([1])
    actor.log_std.weight: torch.Size([1, 128])
    actor.log_std.bias: torch.Size([1])
    critic.qf0.0.weight: torch.Size([32, 28])
    critic.qf0.0.bias: torch.Size([32])
    critic.qf0.2.weight: torch.Size([32, 32])
    critic.qf0.2.bias: torch.Size([32])
    critic.qf0.4.weight: torch.Size([1, 32])
    critic.qf0.4.bias: torch.Size([1])
    critic.qf1.0.weight: torch.Size([32, 28])
    critic.qf1.0.bias: torch.Size([32])
    critic.qf1.2.weight: torch.Size([32, 32])
    critic.qf1.2.bias: torch.Size([32])
    critic.qf1.4.weight: torch.Size([1, 32])
    critic.qf1.4.bias: torch.Size([1])
    critic_target.qf0.0.weight: torch.Size([32, 28])
    critic_target.qf0.0.bias: torch.Size([32])
    critic_target.qf0.2.weight: torch.Size([32, 32])
    critic_target.qf0.2.bias: torch.Size([32])
    critic_target.qf0.4.weight: torch.Size([1, 32])
    critic_target.qf0.4.bias: torch.Size([1])
    critic_target.qf1.0.weight: torch.Size([32, 28])
    critic_target.qf1.0.bias: torch.Size([32])
    critic_target.qf1.2.weight: torch.Size([32, 32])
    critic_target.qf1.2.bias: torch.Size([32])
    critic_target.qf1.4.weight: torch.Size([1, 32])
    critic_target.qf1.4.bias: torch.Size([1])
    '''
    policy_params = []
    value_params = []
    value_targets = []
    holder = TD3_Model.get_parameters()
    orig_params = holder['policy']
    for i in orig_params:
        if 'actor' in i:
            policy_params.append(orig_params[i])
        if 'critic.qf' in i:
            value_params.append(orig_params[i])
        if 'critic_target' in i:
            value_targets.append(orig_params[i])
    new_policy = policy_net_update
    new_value = value_net_update
    for pp, npp in zip([param for param in orig_params if 'actor' in param and 'std' not in param], new_policy):
        orig_params[pp] = npp
    if critic_net_update:
        for vp, vt, nvp in zip([param for param in orig_params if 'critic.qf' in param],
                        [param for param in orig_params if 'critic_target' in param], new_value):
            orig_params[vp] = nvp
            orig_params[vt] = nvp
    holder['policy'] = orig_params
    return holder
        
def get_ActorCritic_delta(global_policy, global_value, curr_policy, curr_value):
    policy_delta = [cp - gp for cp, gp in zip(curr_policy, global_policy)]
    value_delta = [cv - gv for cv, gv in zip(curr_value, global_value)]
    return policy_delta, value_delta

def replace_torch_parameters(model, new_params, start_index=2):
    # Get current parameters as a list
    policy = model.get_model()
    current_params = list(policy.parameters())
    
    # Replace parameters from start_index with new_params
    for i, new_param in enumerate(new_params):
        if start_index + i < len(current_params):
            current_params[start_index + i].data.copy_(new_param)
        else:
            raise IndexError("The new_params list is longer than the remaining parameters to be updated.")
    
    # Set the updated parameters back to the module
    for param, new_param in zip(policy.parameters(), current_params):
        param.data.copy_(new_param)

    return policy



