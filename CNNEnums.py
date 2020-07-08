from enum import Enum


class PoolingType(Enum):
    NONE = 0
    MAXPOOLING = 1
    AVGPOOLING = 2

class ActivationType(Enum):
    SIGMOID = 0
    TANH = 1
    RELU = 2
    SOFTMAX = 3