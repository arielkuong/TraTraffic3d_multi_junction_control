import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os
import pandas as pd
import csv

if __name__ == "__main__":

    args = get_args()
    eval_file_path_1 = args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_J1_rewards_jointlearning_maxvehicle30.npy'
    eval_file_path_2 = args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_J2_rewards_jointlearning_maxvehicle30.npy'
    eval_file_path_3 = args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_J3_rewards_jointlearning_maxvehicle30.npy'
    eval_file_path_4 = args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_J4_rewards_jointlearning_maxvehicle30.npy'

    eval_file_path_5 = args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_J1_rewards_independentlearning_maxvehicle30.npy'
    eval_file_path_6 = args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_J2_rewards_independentlearning_maxvehicle30.npy'
    eval_file_path_7 = args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_J3_rewards_independentlearning_maxvehicle30.npy'
    eval_file_path_8 = args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_J4_rewards_independentlearning_maxvehicle30.npy'

    eval_file_path_9 = args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_J1_rewards_jointlearning_deepernets_maxvehicle30.npy'
    eval_file_path_10 = args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_J2_rewards_jointlearning_deepernets_maxvehicle30.npy'
    eval_file_path_11 = args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_J3_rewards_jointlearning_deepernets_maxvehicle30.npy'
    eval_file_path_12 = args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_J4_rewards_jointlearning_deepernets_maxvehicle30.npy'

    # CSVData = pd.read_csv('/home/CAMPUS/180205312/traffic3d-develop/Traffic3D/Assets/Results/TrueRewards.csv')
    # traffic3d_rewards = np.genfromtxt(CSVData, delimiter=",")
    # traffic3d_rewards = traffic3d_rewards[:traffic3d_rewards.shape[0]-1]
    # traffic3d_rewards = traffic3d_rewards.astype(int)
    # traffic3d_ep_rewards = np.add.reduceat(traffic3d_rewards, np.arange(0, len(traffic3d_rewards), 100))
    # print(traffic3d_rewards)
    # print(traffic3d_rewards.shape)
    # print(traffic3d_ep_rewards)
    # print(traffic3d_ep_rewards.shape)
    # data_sumo = traffic3d_ep_rewards[:traffic3d_ep_rewards.shape[0]-1]

    # data = np.load(eval_file_path)
    data1 = np.load(eval_file_path_1)
    data2 = np.load(eval_file_path_2)
    data3 = np.load(eval_file_path_3)
    data4 = np.load(eval_file_path_4)
    data5 = np.load(eval_file_path_5)
    data6 = np.load(eval_file_path_6)
    data7 = np.load(eval_file_path_7)
    data8 = np.load(eval_file_path_8)
    data9 = np.load(eval_file_path_9)
    data10 = np.load(eval_file_path_10)
    data11 = np.load(eval_file_path_11)
    data12 = np.load(eval_file_path_12)
    print(data9)
    print(data10)
    print(data11)
    print(data12)

    data_all1  = data1 + data2 + data3 + data4
    data_all2 = data5 + data6 + data7 + data8
    data_all3 = data9 + data10 + data11 + data12

    np.save(args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_All_rewards_jointlearning_maxvehicle30.npy', data_all1)
    np.save(args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_All_rewards_independentlearning_maxvehicle30.npy', data_all2)
    np.save(args.save_dir + '/DQN_simple_light_4Js/Seed' + str(args.seed) + '/episode_All_rewards_jointlearning_deepernets_maxvehicle30.npy', data_all3)

    # x = np.linspace(0, len(data), len(data))
    # x1 = np.linspace(0, len(data1), len(data1))
    # x2 = np.linspace(0, len(data2), len(data2))
    # x3 = np.linspace(0, len(data3), len(data3))
    # x4 = np.linspace(0, len(data4), len(data4))
    #
    # x5 = np.linspace(0, len(data5), len(data5))
    # x6 = np.linspace(0, len(data6), len(data6))
    # x7 = np.linspace(0, len(data7), len(data7))
    # x8 = np.linspace(0, len(data8), len(data8))
    #
    x1 = np.linspace(0, len(data_all1), len(data_all1))
    x2 = np.linspace(0, len(data_all2), len(data_all2))
    x3 = np.linspace(0, len(data_all3), len(data_all3))

    mpl.style.use('ggplot')
    fig = plt.figure(1)
    fig.patch.set_facecolor('white')
    plt.xlabel('Episodes', fontsize=16)
    plt.ylabel('Cumulative Return', fontsize=16)
    plt.title('DQN', fontsize=20)

    plt.plot(x1, data_all1, color='red', linewidth=2, label='joint learning')
    plt.plot(x2, data_all2, color='blue', linewidth=2, label='independent learning')
    plt.plot(x3, data_all3, color='green', linewidth=2, label='joint learning(deeper nets)')

    # plt.plot(x1, data1, color='red', linewidth=2, label='joint learning, J1')
    # plt.plot(x2, data2, color='orange', linewidth=2, label='joint learning, J2')
    # plt.plot(x3, data3, color='blue', linewidth=2, label='joint learning, J3')
    # plt.plot(x4, data4, color='green', linewidth=2, label='joint learning, J4')

    # plt.plot(x5, data5, color='red', linewidth=2, label='independent learning, J1')
    # plt.plot(x6, data6, color='orange', linewidth=2, label='independent learning, J2')
    # plt.plot(x7, data7, color='blue', linewidth=2, label='independent learning, J3')
    # plt.plot(x8, data8, color='green', linewidth=2, label='independent learning, J4')
    plt.legend(loc='lower right')
    plt.show()
