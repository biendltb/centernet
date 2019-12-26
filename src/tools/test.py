import numpy as np

from src.networks import gauface_dla


def main():
    _input = np.random.rand(32, 224, 224, 3).astype(np.float32)

    model = gauface_dla.dla_lite_net()

    outs = model(_input)

    print('stop here')




if __name__ == '__main__':
    main()