import argparse

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--num-episodes', type=int, default=50, help='the number of episodes to train the agent')
    parser.add_argument('--episode-length', type=int, default=100, help='the number of timesteps to count as an episode')
    parser.add_argument('--batch-size', type=int, default=128, help='the sample batch size for optimizing the models')
    parser.add_argument('--buffer-size', type=int, default=int(1000), help='the size of the buffer')
    parser.add_argument('--target-update-step', type=int, default=10, help='the step when the target network should be updated')
    parser.add_argument('--random-eps', type=float, default=0.1, help='random eps for action exploration')
    parser.add_argument('--gamma', type=float, default=0.99, help='the discount factor for training models')
    parser.add_argument('--lr', type=float, default=0.001, help='the learning rate of the networks')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')

    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--cuda', action='store_true', help='if use gpu do the acceleration')
    parser.add_argument('--load-path', type=str, default=None, help='the path to load the previous saved models')

    parser.add_argument('--num-junctions', type=int, default=2, help='number of junctions, only valid for multi junction control')


    args = parser.parse_args()

    return args
