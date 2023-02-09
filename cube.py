import numpy as np

class CubeEnvironment:
    def __init__(self):
        self.__cube = np.array(range(1, 3 * 3 * 3 + 1)).reshape(3, 3, 3)
        self.__ground_truth = np.array(range(1, 3 * 3 * 3 + 1)).reshape(3, 3, 3)
        # self.__actions = ['u', 'd', 'f', 'b', 'l', 'r', 'uu', 'du', 'fu', 'bu', 'lu', 'ru']
        self.__actions = np.array(range(12))
        
    def __front(self):
        self.__cube[2, :, :] = np.rot90(self.__cube[2, :, :], k=3)

    def __frontu(self):
        self.__cube[2, :, :] = np.rot90(self.__cube[2, :, :], k=1)

    def __back(self):
        self.__cube[0, :, :] = np.rot90(self.__cube[0, :, :], k=3)

    def __backu(self):
        self.__cube[0, :, :] = np.rot90(self.__cube[0, :, :], k=1)

    def __up(self):
        self.__cube[:, 0, :] = np.rot90(self.__cube[:, 0, :], k=3)

    def __upu(self):
        self.__cube[:, 0, :] = np.rot90(self.__cube[:, 0, :], k=1)

    def __down(self):
        self.__cube[:, 2, :] = np.rot90(self.__cube[:, 2, :], k=3)

    def __downu(self):
        self.__cube[:, 2, :] = np.rot90(self.__cube[:, 2, :], k=1)

    def __left(self):
        self.__cube[:, :, 0] = np.rot90(self.__cube[:, :, 0], k=3)

    def __leftu(self):
        self.__cube[:, :, 0] = np.rot90(self.__cube[:, :, 0], k=1)

    def __right(self):
        self.__cube[:, :, 2] = np.rot90(self.__cube[:, :, 2], k=3)

    def __rightu(self):
        self.__cube[:, :, 2] = np.rot90(self.__cube[:, :, 2], k=1)
    
    def suffle(self, count=20):
        actions = np.random.choice(self.__actions, size=(count,))
        
        for action in actions:
            self.move(action)
    
    def reset(self):
        self.__cube = np.array(range(1, 3 * 3 * 3 + 1)).reshape(3, 3, 3)
        
    @property
    def score(self):
        return np.sum(self.__cube == self.__ground_truth)
    
    @property
    def done(self):
        return np.all(self.__cube == self.__ground_truth)
    
    def move(self, action):
        if action == 'u' or action == 0:
            self.__up()
        elif action == 'uu' or action == 1:
            self.__upu()
        elif action == 'd' or action == 2:
            self.__down()
        elif action == 'du' or action == 3:
            self.__downu()
        elif action == 'f' or action == 4:
            self.__front()
        elif action == 'fu' or action == 5:
            self.__frontu()
        elif action == 'b' or action == 6:
            self.__back()
        elif action == 'bu' or action == 7:
            self.__backu()
        elif action == 'l' or action == 8:
            self.__left()
        elif action == 'lu' or action == 9:
            self.__leftu()
        elif action == 'r' or action == 10:
            self.__right()
        elif action == 'ru' or action == 11:
            self.__rightu()
        else:
            return False
        
        return np.all(self.__cube == self.__ground_truth)
    
    @property
    def actions(self):
        return self.__actions
    
    @property
    def tensor(self):
        return np.copy(self.__cube)
