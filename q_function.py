class QFunction:

    def __init__(self, round_digits):

        self.round_digits = round_digits
        self.q_table = {}

    def register(self, observations, action, registration_value):

        index = self._get_index(observations, action)
        self.q_table[index] = registration_value

    def get_value(self, observations, action):

        index = self._get_index(observations, action)
        if index in self.q_table:
            return self.q_table[index]
        else:
            return 0

    def _get_index(self, observations, action):

        index = [round(observation, round_digits) for observation, round_digits in zip(observations, self.round_digits)]
        index.append(action)

        return tuple(index)


def test():

    import random

    q_function = QFunction([1, 1])
    observations = []

    for i in range(10):
        observations = [i/15]
        value = random.random()
        q_function.register(observations, 0, value)
        print(i, observations, value)
    print()
    for i in range(20):
        observations = [i/30]
        print(i, observations, q_function.get_value(observations, 0))
    print("oor", q_function.get_value([100], 0))

    return q_function


if __name__ == "__main__":
    test()
