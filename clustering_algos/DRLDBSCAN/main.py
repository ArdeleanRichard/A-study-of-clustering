import collections
import warnings
import torch
from clustering_algos.DRLDBSCAN.utils.utils import *
from clustering_algos.DRLDBSCAN.model.model import DrlDbscan

"""
    Training and testing DRL-DBSCAN.
    Paper: Automating DBSCAN via Reinforcement Learning
    Source: https://anonymous.4open.science/r/DRL-DBSCAN
"""





class DrlDbscanAlgorithm:
    def __init__(self, init=None):
        self.train_size = 0.20  # Sample size used to get rewards
        self.episode_num = 15  # The number of episodes
        self.block_num = 1  # The number of data blocks
        self.block_size = 5040  # The size of data block
        self.layer_num = 3  # The number of recursive layers
        self.eps_size = 5  # Eps parameter space size
        self.min_size = 4  # MinPts parameter space size
        self.reward_factor = 0.2  # The impact factor of reward
    
        self.device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 16  # Reinforcement learning sampling batch size
        self.step_num = 30  # Maximum number of steps per RL game

    def fit_predict(self, X):
        # standardize output records and ignore warnings
        warnings.filterwarnings('ignore')

        # sample index for reward
        idx_reward = random.sample(range(len(X)), int(len(X) * self.train_size))

        idx_reward, features = [idx_reward], [X]

        # generate parameter space size, step size, starting point of the first layer, limit bound
        p_size, p_step, p_center, p_bound = generate_parameter_space(features[0], self.layer_num, self.eps_size, self.min_size)

        # build a multi-layer agent collection, each layer has an independent agent
        agents = []
        for l in range(0, self.layer_num):
            drl = DrlDbscan(p_size, p_step[l], p_center, p_bound, self.device, self.batch_size,
                            self.step_num, features[0].shape[1])
            agents.append(drl)

        # Train agents with serialized data blocks
        for b in range(0, self.block_num):
            final_reward_test = [0, p_center, 0]
            label_dic_test = set()

            # test each layer agent
            for l in range(0, self.layer_num):
                agent = agents[l]

                # update starting point
                agent.reset(final_reward_test)

                # testing
                cur_labels, cur_cluster_num, p_log = agent.detect(features[b], collections.OrderedDict())
                final_reward_test = [0, p_log[-1], 0]


            max_max_reward = [0, p_center, 0]
            max_reward = [0, p_center, 0]
            label_dic = collections.OrderedDict()
            first_meet_num = 0

            # train each layer agent
            for l in range(0, self.layer_num):
                agent = agents[l]
                agent.reset(max_max_reward)
                max_max_reward_logs = [max_max_reward[0]]
                early_stop = False
                his_hash_size = len(label_dic)
                cur_hash_size = len(label_dic)
                for i in range(1, self.episode_num):
                    # begin training process
                    p_logs = np.array([[], []])
                    nmi_logs = np.array([])

                    # update starting point
                    agent.reset0()

                    # train the l-th layer
                    cur_labels, cur_cluster_num, p_log, nmi_log, max_reward = agent.train(i, idx_reward[b], features[b],
                                                                                          labels[b], label_dic,
                                                                                          self.reward_factor)
                    if max_max_reward[0] < max_reward[0]:
                        max_max_reward = list(max_reward)
                        cur_hash_size = len(label_dic)
                    max_max_reward_logs.append(max_max_reward[0])

                    # update starting point
                    agent.reset0()

                    # early stop
                    if len(max_max_reward_logs) > 3 and \
                            max_max_reward_logs[-1] == max_max_reward_logs[-2] == max_max_reward_logs[-3] and \
                            max_max_reward_logs[-1] != max_max_reward_logs[0]:
                        break
                first_meet_num += cur_hash_size - his_hash_size
                if cur_hash_size == his_hash_size:
                    break

            cur_labels = label_dic[str(max_max_reward[1][0]) + str("+") + str(max_max_reward[1][1])]

            # # evaluate clustering result
            # max_reward_nmi = 0
            # max_nmi = 0
            # max_nmi_logs = []
            # for cur_labels in label_dic.values():
            #     reward_nmi = metrics.normalized_mutual_info_score(labels[b][idx_reward[b]], cur_labels[idx_reward[b]])
            #     nmi = metrics.normalized_mutual_info_score(labels[b], cur_labels)
            #     if reward_nmi > max_reward_nmi:
            #         max_reward_nmi, max_nmi = reward_nmi, nmi
            #     max_nmi_logs.append(max_nmi)
            # print(max_nmi_logs)

            return cur_labels


if __name__ == '__main__':
    data_path = 'data/Shape-Aggregation.txt'  # Path of features and labels
    extract_data = []
    with open(data_path, 'r') as f:
        for line in f:
            data = line.split()
            data[0] = float(data[0])
            data[1] = float(data[1])
            data[2] = int(data[2])
            extract_data.append(data)

    # get sample serial numbers for rewards, out-of-order data features and labels
    # feature normalization
    features = np.array(MinMaxScaler().fit_transform([i[0:2] for i in extract_data]))
    labels = np.array([i[2] for i in extract_data])

    model = DrlDbscanAlgorithm()
    cur_labels = model.fit_predict(features, labels)
    nmi, ami, ari = dbscan_metrics(labels, cur_labels)
    print(nmi, ami, ari)
