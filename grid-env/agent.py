from mdp import MDP

def generate_random_policy(actions):
    def random_policy(state,action):
        return 1/len(actions)
    return random_policy

class Agent:
    def __init__(self, policy =None, gamma:float=0.9) -> None:
        self.policy = policy
        self.gamma = gamma

    def policy_evaluation(self,omega:float,environment:MDP,verbose:bool=False):
        if self.policy is None:
            self.policy = generate_random_policy(environment.actions)
            if verbose:
                print('Random policy will be used for policy evaluation')
        value_function = {i:0 for i in environment.states}
        iteration = 0
        if verbose:
            print('Policy Evaluation-- Initial Value Function')
            environment.value_visualization(value_function,iteration)
        
        while True:
            delta = 0
            for state in environment.states:
                v_new = 0
                for action in environment.actions:
                    act_prob = self.policy(state,action)
                    next_state_and_rewards = environment.get_next_state_reward(state,action)
                    if next_state_and_rewards.size().numel() == 0:
                        continue

                    for i in range(next_state_and_rewards.size()[0]):
                        next_state = next_state_and_rewards[i,0].item()
                        reward = next_state_and_rewards[i,1].item()
                        prob = next_state_and_rewards[i,2].item()
                        v_new += act_prob * (reward + self.gamma * value_function[next_state]) * prob
                delta = max(delta,abs(v_new - value_function[state]))
                value_function[state] = v_new
            if delta < omega:
                break
            iteration += 1
            if verbose:
                print(f'Policy Evaluation-- Iteration {iteration}')
            environment.value_visualization(value_function,iteration)

        if verbose:
            print('Policy Evaluation-- Final Value Function')
            environment.value_visualization(value_function,iteration)
        return value_function
    





