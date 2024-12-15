from gymnasium import Env, spaces
from gymnasium.envs.toy_text.utils import categorical_sample
from typing import Optional

class RecycleEnv(Env):

    def __init__(self, alpha=0.5, beta=0.2, duration=20,r_search=4, r_wait=1):
        self.observation_space = spaces.Discrete(2)  # 0: low 1: high
        self.action_space = spaces.Discrete(3)  # 0: search 1: wait 2: recharge

        self.state = "high"
        self.sum = 0

        self.StateActionPairings = {"low": ["search", "wait", "recharge"],
                                    "high": ["search", "wait"]}

        # action: {state:[(nextstate, reward, probability, termination)]}
        self.TransitionStatesandProbs = {
            "wait": {
                "high": [("high", r_wait, 1, False)],
                "low": [("low", r_wait, 1, False)]
            },
            "recharge": {
                "low": [("high", 0, 1, False)]
            },
            "search": {
                "high": [("high", r_search, round(alpha, 1), False),
                         ("low", r_search, round(1-alpha, 1), False)],
                "low": [("low", r_search, round(beta, 1), False),
                        ("high", -3, round(1-beta, 1), False)]
            }}

    def reset(self, seed=None, options: Optional[dict] = None):
        self.state = "high"
        self.time = 0
        self.sum = 0
        return self.state, {}

    def step(self, a):
        transitions = self.TransitionStatesandProbs[a][self.state]
        i = categorical_sample([t[2] for t in transitions], self.np_random)
        s, r, p, t = transitions[i]
        self.SumReward(r)
        self.state = s
        return (s, r, t, False, {})

    def SumReward(self, reward=0):
        self.sum = self.sum + reward
        return self.sum

    def getPossibleActions(self, state):
        return self.StateActionPairings[state]

    def getTransitionStatesandProbs(self, state, action):
        return self.TransitionStatesandProbs[action][state]