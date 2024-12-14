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
    

    def policy_improvement(self,environment:MDP,value_function:dict,verbose:bool=False) -> dict:
        new_policy = {}

        for state in environment.states:
            max_val = float('-inf')
            max_act = None
            for action in environment.actions:
                next_info = environment.get_next_state_reward(state,action)
                if next_info.size().numel() == 0:
                    continue
                val = 0
                for i in range(next_info.size()[0]):
                    next_state = next_info[i,0].item()
                    reward = next_info[i,1].item()
                    prob = next_info[i,2].item()
                    val += prob * (reward + self.gamma * value_function[next_state])
                if val > max_val:
                    max_val = val
                    max_act = action
                
            new_policy[state] = max_act
            # If actions determined is certain for previous policy, then break
       

        def new_policy_func(state,action):
            return 1 if action == new_policy[state] else 0
        self.policy = new_policy_func
        if verbose:
            print('Policy Improvement-- New Policy')
            environment.policy_visualization(new_policy_func,iteration=1)
        return new_policy
    
    def policy_iteration(self,omega:float,environment:MDP,verbose:bool=False):

        policy_stable = True

        while policy_stable:
            value_function = self.policy_evaluation(omega,environment,False)
            new_policy = self.policy_improvement(environment,value_function,False)
            for k,v in new_policy.items():
                if self.policy(k,v) != 1:
                    policy_stable = False
            