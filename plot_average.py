import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from arguments import get_args
import os

seed = [123, 570, 766]

if __name__ == "__main__":

    args = get_args()
    eval_file_path_1 = args.save_dir + '/DQN_simple_light_2Js/Seed' + str(seed[0]) + '/episode_All_rewards_jointlearning_maxvehicle16.npy'
    eval_file_path_2 = args.save_dir + '/DQN_simple_light_2Js/Seed' + str(seed[1]) + '/episode_All_rewards_jointlearning_maxvehicle16.npy'
    eval_file_path_3 = args.save_dir + '/DQN_simple_light_2Js/Seed' + str(seed[2]) + '/episode_All_rewards_jointlearning_maxvehicle16.npy'
    eval_file_path_4 = args.save_dir + '/DQN_simple_light_2Js/Seed' + str(seed[0]) + '/episode_All_rewards_independentlearning_maxvehicle16.npy'
    eval_file_path_5 = args.save_dir + '/DQN_simple_light_2Js/Seed' + str(seed[1]) + '/episode_All_rewards_independentlearning_maxvehicle16.npy'
    eval_file_path_6 = args.save_dir + '/DQN_simple_light_2Js/Seed' + str(seed[2]) + '/episode_All_rewards_independentlearning_maxvehicle16.npy'

    if not os.path.isfile(eval_file_path_1) \
    or not os.path.isfile(eval_file_path_2) \
    or not os.path.isfile(eval_file_path_3) \
    or not os.path.isfile(eval_file_path_4) \
    or not os.path.isfile(eval_file_path_5) \
    or not os.path.isfile(eval_file_path_6):
        print("Result file do not exist!")
    else:
        data_len = 100
        data1 = np.load(eval_file_path_1)[:data_len]
        data2 = np.load(eval_file_path_2)[:data_len]
        data3 = np.load(eval_file_path_3)[:data_len]
        data4 = np.load(eval_file_path_4)[:data_len]
        data5 = np.load(eval_file_path_5)[:data_len]
        data6 = np.load(eval_file_path_6)[:data_len]

        x = np.linspace(0, data_len, data_len)

        data_comb_1 = [data1, data2, data3]
        data_mean_1 = np.mean(data_comb_1, axis=0)
        data_std_1 = np.std(data_comb_1, axis=0)
        data_low_1 = data_mean_1 - data_std_1
        data_high_1 = data_mean_1 + data_std_1

        data_comb_2 = [data3, data4, data5]
        data_mean_2 = np.mean(data_comb_2, axis=0)
        data_std_2 = np.std(data_comb_2, axis=0)
        data_low_2 = data_mean_2 - data_std_2
        data_high_2 = data_mean_2 + data_std_2

        # save_data_path = args.save_dir + args.env_name + '/Average_result_' + str(len(seed)) + 'seeds'

        # if not os.path.exists(save_data_path):
            # os.mkdir(save_data_path)

        # np.save(save_data_path + '/data_mean.npy', data_mean)
        # np.save(save_data_path + '/data_high.npy', data_high)
        # np.save(save_data_path + '/data_low.npy', data_low)

        mpl.style.use('ggplot')
        fig = plt.figure(1)
        fig.patch.set_facecolor('white')
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Cumulative Return', fontsize=16)
        plt.title('DQN', fontsize=20)

        plt.plot(x, data_mean_1, color='red', linewidth=2, label='joint learning')
        plt.fill_between(x, data_low_1, data_high_1, color='red', alpha=0.1)
        plt.plot(x, data_mean_2, color='blue', linewidth=2, label='independent learning')
        plt.fill_between(x, data_low_2, data_high_2, color='blue', alpha=0.1)
        plt.legend(loc='lower right')

        plt.show()
