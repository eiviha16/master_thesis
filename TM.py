import numpy as np


class Clauses:
    def __init__(self):
        self.clauses = None


class TsetlinMachine(Clauses):
    def __init__(self, nr_of_clauses, T, s, memory_size, y_max, y_min):
        super().__init__()
        self.nr_of_clauses = nr_of_clauses
        self.T = T
        self.s = s
        self.memory_size = memory_size

        self.y_max = y_max
        self.y_min = y_min

    def init(self, data):
        literals = [0 for _ in range(2 * len(data[0]))]
        self.clauses = np.array([literals for _ in range(self.nr_of_clauses)])

    def fit(self, data, ys):
        y_preds = self.predict(data)
        for i in range(len(ys)):
            if y_preds[i] == 1:
                self.type_I_feedback()
            if y_preds[i] == 0:
                self.type_II_feedback()


    def predict(self, data):
        output = [0 for _ in range(len(data))]
        for i, sample in enumerate(data):
            for clause in self.clauses:
                for index in range(len(clause)):
                    if clause[index] != sample[index]:
                        clause_val = False
                        break
                    else:
                        clause_val = True
                if clause_val:
                    output[i] += 1
                output[i] = 1
            # Normalize
            output[i] = (sum(output[i]) - self.y_min) / (self.y_max - self.y_min)
        return output

    def type_I_feedback(self):
        pass

    def type_II_feedback(self):
        pass

    def update(self):
        pass


if __name__ == "__main__":
    TM = TsetlinMachine(nr_of_clauses=5, T=10, s=1, memory_size=4, y_max=10, y_min=0)
    TM.init([[1, 0, 1, 1]])
    a = 2
