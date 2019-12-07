import numpy as np

from src.networks.dla import dla_net


def main():
    _input = np.random.rand(1, 128, 128, 1).astype(np.float32)

    model = dla_net()

    outs = model(_input)

    print('stop here')




if __name__ == '__main__':
    main()