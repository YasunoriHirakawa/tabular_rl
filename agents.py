import random


class QLearning:

    def __init__(self, q_function, initial_observations, num_actions, epsilon=0.3, learning_rate=0.1, discount_rate=0.99):

        self.q_function = q_function
        self.prev_observations = initial_observations

        self.num_actions = num_actions
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

    def decide_action(self, observations):

        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions-1)
            return action
        else:
            action = self._calc_max_q(observations)["max_q_action"]
            return action

    def update_q_function(self, observations, prev_action, prev_reward):

        prev_q = self.q_function.get_value(self.prev_observations, prev_action)
        target = self.discount_rate * self._calc_max_q(observations)["max_q"] - prev_q
        updated_q = prev_q + self.learning_rate * (prev_reward + target)
        self.q_function.register(self.prev_observations, prev_action, updated_q)

        self.prev_observations = observations

    def _calc_max_q(self, observations):

        max_q = 0
        max_q_action = 0

        for action in range(self.num_actions):
            q_value = self.q_function.get_value(observations, action)
            max_q = max(max_q, q_value)
            max_q_action = action if q_value == max_q else max_q_action

        return {"max_q": max_q, "max_q_action": max_q_action}


class SARSA:

    def __init__(self, q_function, initial_observations, num_actions, epsilon=0.3, learning_rate=0.1, discount_rate=0.99):

        self.q_function = q_function
        self.experience = {"prev_prev_observations": None, "prev_prev_action": None, "prev_prev_reward": None, "prev_observations": initial_observations}

        self.num_actions = num_actions
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

    def decide_action(self, observations):

        if random.random() < self.epsilon:
            action = random.randint(0, self.num_actions-1)
            return action
        else:
            action = self._calc_max_q_action(observations)
            return action

    def update_q_function(self, observations, prev_action, prev_reward):

        if self.experience["prev_prev_observations"] is not None:
            prev_q = self.q_function.get_value(self.experience["prev_prev_observations"], self.experience["prev_prev_action"])
            target = self.discount_rate * self.q_function.get_value(self.experience["prev_observations"], prev_action) - prev_q
            updated_q = prev_q + self.learning_rate * (self.experience["prev_prev_reward"] + target)
            self.q_function.register(self.experience["prev_prev_observations"], self.experience["prev_prev_action"], updated_q)

        self.experience["prev_prev_observations"] = self.experience["prev_observations"]
        self.experience["prev_prev_action"] = prev_action
        self.experience["prev_prev_reward"] = prev_reward
        self.experience["prev_observations"] = observations

    def _calc_max_q_action(self, observations):

        max_q = 0
        max_q_action = 0

        for action in range(self.num_actions):
            q_value = self.q_function.get_value(observations, action)
            max_q = max(max_q, q_value)
            max_q_action = action if q_value == max_q else max_q_action

        return max_q_action


def test():

    import q_function as qf

    q_function = qf.test()
    methods = [QLearning(q_function, 0, 2), SARSA(q_function, 0, 2)]

    for method in methods:
        print()
        for i in range(10):
            print(method.decide_action([0.0]))
        print()
        for i in range(10):
            observations = [i/15]
            ret = method._calc_max_q(observations) if type(method) is QLearning else method._calc_max_q_action(observations)
            print(ret)


if __name__ == "__main__":
    test()
