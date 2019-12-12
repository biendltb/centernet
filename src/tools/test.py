import numpy as np

from src.networks import dla


def main():
    _input = np.random.rand(1, 120, 160, 1).astype(np.float32)

    model = dla.dla_lite_net(mode='eval')

    outs = model(_input)

    print('stop here')




if __name__ == '__main__':
    main()