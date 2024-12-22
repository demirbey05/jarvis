import numpy as np

class SalesEnv:
    def __init__(self, max_inventory, max_sale_time,buy_price= 14,scrap_price=5):
        self.max_inventory = max_inventory
        self.max_sale_time = max_sale_time
        self.state =[max_inventory,1]
        self.buy_price = buy_price
        self.scrap_price = scrap_price
        self.state_space = [[q,t] for q in range(1,max_inventory+1) for t in range(1,max_sale_time+1)]
        self.state_space.append([0,-1])
        self.action_space = [5,10,15,20,25]

    def reset(self):
        self.state[0] = self.max_inventory
        self.state[1] = 1
        return [self.max_inventory, 1]


    def step(self, action):

        if self.state[1] == self.max_sale_time:
            reward = self.scrap_price * self.state[0]
            return [0,-1], reward, True

        # Get demand for action 
        demand = self.get_demand(action)

        # Calculate reward
        if self.state[1] == 1:
            reward = action * demand - self.buy_price * demand
        else:
            reward = action * demand
            
        # Update inventory
        self.state[0] = self.state[0] - min(self.state[0], demand)
        self.state[1] += 1

        if self.state[0] == 0:
            return [0,-1], reward, True
        
        return self.state, reward,False

    def get_demand(self, action) -> int:
        # Define the price points and corresponding demands
        l1 = [(10, 20), (12, 12)]  # first line segment
        l2 = [(12, 12), (15, 10)]  # second line segment
        l3 = [(15, 10)]           # third segment (logarithmic)

        # Calculate demand based on price segments
        if action <= max(l1[0][0], l1[1][0]):
            # First line segment (linear)
            a = (l1[0][1] - l1[1][1]) / (l1[0][0] - l1[1][0])
            b = l1[0][1] - a * l1[0][0]
            d = a * action + b
            demand = d * np.random.uniform(0.75, 1.25)
            
        elif l2[0][0] <= action <= l2[1][0]:
            # Second line segment (linear)
            a = (l2[0][1] - l2[1][1]) / (l2[0][0] - l2[1][0])
            b = l2[0][1] - a * l2[0][0]
            d = a * action + b
            demand = d * np.random.uniform(0.75, 1.25)
            
        elif action >= l3[0][0]:
            # Third segment (logarithmic)
            d = -4 * np.log(action - l3[0][0] + 1) + l3[0][1]
            demand = d * np.random.uniform(1, 2)
        
        # Additional time-based adjustment
        if self.state[1] <= self.max_sale_time / 2:
            demand = np.random.uniform(1, 1.2) * demand

        return round(demand)

        