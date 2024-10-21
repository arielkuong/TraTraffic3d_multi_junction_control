import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque
from itertools import count
import random
import os
import cv2
from datetime import datetime

from traffic3d_multi_junction import Traffic3DMultiJunction
from arguments import get_args

class DQN_joint(nn.Module):
    def __init__(self, n_actions, n_junctions):
        super(DQN_joint, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(64)

        self.n_junctions = n_junctions
        self.head = nn.Linear(576 * self.n_junctions, n_actions ** self.n_junctions)
        # self.saved_log_probs = []
        # self.basic_rewards = []

    # def img_process(self, x):
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = F.relu(self.bn3(self.conv3(x)))
    #     x = F.relu(self.bn4(self.conv4(x)))
    #     return x.reshape(x.size(0), -1)

    def forward(self, imgs):
        all = []
        for i in range(imgs.shape[1]):
            x = imgs[:,i,...]
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = x.view(x.size(0), -1)
            x = x.unsqueeze(1)
            all.append(x)
        all = torch.cat(all, 1)
        all = all.view(all.size(0), -1)
        #return self.head(x.view(x.size(0), -1))
        return self.head(all)

class DQN_joint_deeper(nn.Module):
    def __init__(self, n_actions, n_junctions):
        super(DQN_joint_deeper, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(64)


        self.n_junctions = n_junctions
        self.fc1 = nn.Linear(576 * self.n_junctions, 576 * self.n_junctions)
        self.fc2 = nn.Linear(576 * self.n_junctions, 576 * self.n_junctions)
        self.fc3 = nn.Linear(576 * self.n_junctions, 576 * self.n_junctions)
        self.head = nn.Linear(576 * self.n_junctions, n_actions ** self.n_junctions)
        # self.saved_log_probs = []
        # self.basic_rewards = []

    def forward(self, imgs):
        all = []
        for i in range(imgs.shape[1]):
            x = imgs[:,i,...]
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.relu(self.bn4(self.conv4(x)))
            x = x.view(x.size(0), -1)
            x = x.unsqueeze(1)
            all.append(x)
        all = torch.cat(all, 1)
        all = all.view(all.size(0), -1)
        all =  torch.relu(self.fc1(all))
        all =  torch.relu(self.fc2(all))
        all =  torch.relu(self.fc3(all))
        #return self.head(x.view(x.size(0), -1))
        return self.head(x)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_agent_joint:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.n_actions = env.get_action_space()
        print('Environment action space: {}'.format(self.n_actions))
        print('Number of junctions: {}'.format(self.args.num_junctions))

        self.policy_net = DQN_joint_deeper(self.n_actions, self.args.num_junctions)
        self.target_net = DQN_joint_deeper(self.n_actions, self.args.num_junctions)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr = self.args.lr)
        self.buffer = ReplayBuffer(self.args.buffer_size)

        if args.load_path != None:
            load_policy_model = torch.load(self.args.load_path, map_location=lambda storage, loc: storage)
            self.policy_net.load_state_dict(load_policy_model)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        if self.args.cuda:
            self.policy_net.cuda()
            self.target_net.cuda()

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.save_path = os.path.join(self.args.save_dir, 'DQN_simple_light_4Js', 'Seed' + str(self.args.seed))
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def learn(self):
        # env.reset()
        episode_J1_rewards = []
        episode_J2_rewards = []
        episode_J3_rewards = []
        episode_J4_rewards = []
        episode_traffic_load_sum = []
        print('Learning process starts. Total episodes: {}'.format(self.args.num_episodes))
        for episode in range(self.args.num_episodes):
            # Play an episode
            obs = self.env.reset()
            # print('Env reset')
            reward_sum_J1 = 0
            reward_sum_J2 = 0
            reward_sum_J3 = 0
            reward_sum_J4 = 0
            traffic_load_sum = 0
            for t in count():
                # Play a frame
                # Variabalize 210, 160
                # obs_tensor_all = []
                # actions_all = []
                # reward_all = []
                obs_tensor = self._preproc_inputs_joint(obs)
                # obs_tensor_all.append(obs_tensor)
                # if len(self.buffer) < self.args.batch_size:
                #     action = self._select_action_random()
                # else
                action_joint = self._select_action_joint(obs_tensor)
                action_processed = self._action_postpro(action_joint)
                # action = self._select_action_inorder(t)
                actions = {"actions": []}
                for junction_id in range(self.args.num_junctions):
                    actions["actions"].append({"junctionId": str(junction_id + 1), "action": str(action_processed[junction_id])})
                # action = self._select_action_random()
                # print('Action selected: {}'.format(action))
                # env.render()
                obs_new, rewards, done, info = self.env.step(actions)
                # reward_sum += reward
                # calculate the reward for junction 1 anf junction 2
                reward_sum_J1 += rewards[0]
                reward_sum_J2 += rewards[1]
                reward_sum_J3 += rewards[2]
                reward_sum_J4 += rewards[3]
                traffic_load_sum += info['traffic_load']
                # reward_J1 = -int((-reward_final) % 100)
                # reward_J2 = int(reward_final / 100)
                # reward_all.append(reward_J1)
                # reward_all.append(reward_J2)
                # reward_sum_J1 += reward_J1
                # reward_sum_J2 += reward_J2

                obs = obs_new
                # print('Episode {}, Timestep {}, Action J1: {}, Reward J1 : {}, Action J2: {}, Reward J2 : {}, Traffic_load: {}' \
                #         .format(episode, t, action_processed[0], rewards[0], action_processed[1], rewards[1], info['traffic_load']))

                print('Episode {}, Timestep {}, AJ1:{}, RJ1:{}, AJ2:{}, RJ2:{}, AJ3:{}, RJ3:{}, AJ4:{}, RJ4:{}, Traffic_load: {}' \
                        .format(episode, t, action_processed[0], rewards[0], action_processed[1], rewards[1], \
                        action_processed[2], rewards[2], action_processed[3], rewards[3],info['traffic_load']))

                # Store episode data into the buffers
                obs_next_tensor = self._preproc_inputs_joint(obs_new)
                action_tensor = torch.tensor(action_joint, dtype=torch.float32).unsqueeze(0)
                reward_sum = rewards.sum()
                r_tensor = torch.tensor(reward_sum, dtype=torch.float32).unsqueeze(0)
                if self.args.cuda:
                    action_tensor = action_tensor.cuda()
                    r_tensor = r_tensor.cuda()
                # save the timestep transitions into the replay buffer
                self.buffer.push(obs_tensor, action_tensor, obs_next_tensor, r_tensor)

                # Train the networks
                #print('Optimizing starts')
                if len(self.buffer) >= self.args.batch_size:
                    # print('Updating policy network')
                    self._optimize_model(self.args.batch_size)
                #print('Optimizing finishes')

                if done:
                    break

            # print('[{}] Episode {} finished. Reward total J1: {}, J2: {}, traffic_load: {}'.format(datetime.now(), episode, reward_sum_J1, reward_sum_J2, traffic_load_sum))

            print('[{}] Episode {} finished. Reward total J1: {}, J2: {}, J3: {}, J4: {}, traffic_load: {}' \
                  .format(datetime.now(), episode, reward_sum_J1, reward_sum_J2, reward_sum_J3, reward_sum_J4, traffic_load_sum))

            episode_J1_rewards.append(reward_sum_J1)
            episode_J2_rewards.append(reward_sum_J2)
            episode_J3_rewards.append(reward_sum_J3)
            episode_J4_rewards.append(reward_sum_J4)
            episode_traffic_load_sum.append(traffic_load_sum)
            np.save(self.save_path + '/episode_total_traffic_loads_jointlearning_deepernets_maxvehicle30.npy', episode_traffic_load_sum)
            np.save(self.save_path + '/episode_J1_rewards_jointlearning_deepernets_maxvehicle30.npy', episode_J1_rewards)
            np.save(self.save_path + '/episode_J2_rewards_jointlearning_deepernets_maxvehicle30.npy', episode_J2_rewards)
            np.save(self.save_path + '/episode_J3_rewards_jointlearning_deepernets_maxvehicle30.npy', episode_J3_rewards)
            np.save(self.save_path + '/episode_J4_rewards_jointlearning_deepernets_maxvehicle30.npy', episode_J4_rewards)
            torch.save(self.policy_net.state_dict(), self.save_path + '/policy_network_jointlearning_deepernets_maxvehicle30.pt')


             # Update the target network, copying all weights and biases in DQN
            if episode >= self.args.target_update_step and episode % self.args.target_update_step == 0:
                print('Updating target networks')
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print('Learning process finished')

    def _preproc_inputs_joint(self, input):
        inputs_tensor = []
        for junction_id in range(self.args.num_junctions):
            junction_id_str = str(junction_id+1)
            x = cv2.resize(input[junction_id_str], (100, 100))
            x = x.transpose(2, 0, 1)
            x = np.ascontiguousarray(x, dtype=np.float32) / 255
            x = torch.from_numpy(x)
            inputs_tensor.append(x)
        inputs_tensor = torch.stack(inputs_tensor)
        inputs_tensor = inputs_tensor.unsqueeze(0)
        if self.args.cuda:
            inputs_tensor = inputs_tensor.cuda()
        return inputs_tensor

    def _select_action_joint(self, state):
        # global steps_done
        sample = random.random()
        # eps_threshold = self.args.eps_end + (self.args.eps_start - self.args.eps_end) * \
        #     math.exp(-1. * steps_done / self.args.eps_decay)
        # steps_done += 1
        if sample > self.args.random_eps:
            with torch.no_grad():
                Q_values_joint = self.policy_net(state)
                # print(Q_values)
                # take the Q_value index with the largest expected return
                # action_tensor = Q_values.max(1)[1].view(1, 1)
                action_tensor = torch.argmax(Q_values_joint)
                action_joint = action_tensor.detach().cpu().numpy().squeeze()
                # print(action_tensor)
                return action_joint
        else:
            return random.randrange(self.n_actions ** self.args.num_junctions)

    def _action_postpro(self, action_joint):
        actions = []
        remain = action_joint
        for junction_id in range(self.args.num_junctions):
            actions.append(remain % self.n_actions)
            remain = int(remain / self.n_actions)
        # actions.append(action_joint % self.n_actions)
        # actions.append(int(action_joint / self.n_actions))
        actions = np.array(actions)
        return actions

    def _select_action_random(self):
        return random.randrange(self.n_actions)

    def _select_action_inorder(self, timestep):
        action = (timestep % (self.n_actions))
        return action

    def _optimize_model(self, batch_size):
        states, actions, next_states, rewards = Transition(*zip(*self.buffer.sample(batch_size)))

        states = torch.cat(states)
        next_states = torch.cat(next_states)
        rewards = torch.cat(rewards)
        actions = torch.cat(actions)

        predicted_values = torch.gather(self.policy_net(states), 1, actions.long().unsqueeze(-1)).squeeze(-1)
        next_state_values = self.target_net(next_states).max(1)[0].detach()
        expected_values = rewards + self.args.gamma * next_state_values

        # Compute loss
        loss = F.smooth_l1_loss(predicted_values, expected_values)

        # calculate loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return

if __name__ == '__main__':
    args = get_args()
    env = Traffic3DMultiJunction()

    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trainer = DQN_agent_joint(args, env)
    print('Run training with seed {}'.format(args.seed))
    trainer.learn()
