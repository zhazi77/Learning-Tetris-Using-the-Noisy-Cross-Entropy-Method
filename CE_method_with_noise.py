import math
import numpy as np
from scipy import stats
import Tetris


# NOTE: 固定噪声的版本
def simulation_CE_const_noise(alpha, num_iter,rho,noise): 
    # 初始化参数
    mu_0 = [5] * 21                # 初始均值向量，长度为21, 其中每个元素初始值为 5
    sigma_0 = np.diag([100] * 21)  # 初始协方差矩阵，大小为21x21，对角线元素初始化为 100
    V_0 = (mu_0, sigma_0)          # 初始(分布)参数 tuple，包含均值向量和协方差矩阵
    parameters = [V_0]             # 参数列表，初始只包含 V_0
    t = 1                          # 当前迭代步数，初始化为 1

    L_plot = []  # 用于存储每次迭代后最佳样本的模拟结果
    L_norm = []  # 用于存储每次迭代后协方差矩阵的范数

    for _ in range (num_iter):
        distribution = stats.multivariate_normal(parameters[t-1][0], parameters[t-1][1],allow_singular=True) # type: ignore

        # 评估每个参数池
        N = 100            # 样本数量
        sample_list = []   # 样本列表
        sample_score = []  # 样本得分列表

        for _ in range(N):
            sample = distribution.rvs()                     # 从分布中随机抽取样本（参数向量 W）
            sample_score.append(Tetris.simulation(sample))  # 对样本进行模拟，获取得分
            sample_list.append(sample)                      # 将样本添加到列表
            

        # 保留分数最高的 rho*N 个向量
        k=math.floor(N*rho)
        indices = sorted(range(len(sample_score)), key=lambda i: sample_score[i], reverse=True)[:k]
        sample_high = [sample_list[i] for i in indices]
        best_sample = sample_list[indices[0]]
    

        mean = np.mean(sample_high, axis=0)                  # 计算保留下来的向量的均值
        cov  = np.cov(sample_high, rowvar=False, bias=True)  # 计算保留下来的向量的协方差矩阵
        res  = (mean, cov)                                   # 结果 tuple，包含均值和协方差矩阵

        L_norm.append(np.linalg.norm(cov))  # 计算并存储协方差矩阵的范数

        # NOTE: 更新参数(公式 2.3、2.5), 添加固定大小的噪声
        matrix_noise = np.diag([noise]*21) 
        parameters.append((alpha * np.array(res[0]) + (1 - alpha) * np.array(parameters[-1][0]),
                        alpha ** 2 * np.array(res[1]) + (1 - alpha) ** 2 * np.array(parameters[-1][1])+matrix_noise))    

        # 统计本轮分布“最好能有多好”。
        L_mean = [sample_score[indices[0]]]  # 初始化平均得分列表，包含最佳样本的得分
        for k in range(29):
            L_mean.append(Tetris.simulation(best_sample))  # 进行剩余29次模拟，添加得分

        print(np.mean(L_mean))  # 打印平均得分
        L_plot.append(L_mean)   # 存储平均得分列表
        t += 1  # 时间步增加
        print(L_plot, L_norm, mean)  # 打印当前迭代的结果
    return L_plot, L_norm, mean  # 返回模拟结果，范数列表和最终均值 # type: ignore


# NOTE: 使用逐渐衰减的噪声的版本
def simulation_CE_deacr_noise(alpha, num_iter, rho,a,b): 
    #alpha : taux d'actualistion 
    #N_mean: nombre de simulation par vecteur
    #N_iteration : nombre d'iterations
    #rho : the fraction of verctors that are selected
    #retourne L_plot : le score maximal par itération
    #noise : value of the constant noise to add
    #a,b : params of the decreasing noise, a=5 , b=100 in the paper

    # 初始化参数
    mu_0 = [5] * 21                # 初始均值向量，长度为21, 其中每个元素初始值为 5
    sigma_0 = np.diag([100] * 21)  # 初始协方差矩阵，大小为21x21，对角线元素初始化为 100
    V_0 = (mu_0, sigma_0)          # 初始(分布)参数 tuple，包含均值向量和协方差矩阵
    parameters = [V_0]             # 参数列表，初始只包含 V_0
    t = 1                          # 当前迭代步数，初始化为 1

    L_plot = []  # 用于存储每次迭代后最佳样本的模拟结果
    L_norm = []  # 用于存储每次迭代后协方差矩阵的范数

    for _ in range (num_iter):
        distribution = stats.multivariate_normal(parameters[t-1][0], parameters[t-1][1],allow_singular=True) # type: ignore

        # 评估每个参数池
        N = 100            # 样本数量
        sample_list = []   # 样本列表
        sample_score = []  # 样本得分列表

        for _ in range(N):
            sample = distribution.rvs()                     # 从分布中随机抽取样本（参数向量 W）
            sample_score.append(Tetris.simulation(sample))  # 对样本进行模拟，获取得分
            sample_list.append(sample)                      # 将样本添加到列表

        # 保留分数最高的 rho*N 个向量
        k=math.floor(N*rho)
        indices = sorted(range(len(sample_score)), key=lambda i: sample_score[i], reverse=True)[:k]
        sample_high = [sample_list[i] for i in indices]
        best_sample = sample_list[indices[0]]

        mean = np.mean(sample_high, axis=0)                  # 计算保留下来的向量的均值
        cov  = np.cov(sample_high, rowvar=False, bias=True)  # 计算保留下来的向量的协方差矩阵
        res  = (mean, cov)                                   # 结果 tuple，包含均值和协方差矩阵

        L_norm.append(np.linalg.norm(cov))  # 计算并存储协方差矩阵的范数

        # NOTE: 更新参数(公式 2.3、2.5), 添加逐渐减小的噪声
        noise = max(0, a - N / b)
        matrix_noise = np.diag([noise] * 21)
        parameters.append((alpha * np.array(res[0]) + (1 - alpha) * np.array(parameters[-1][0]),
                           alpha ** 2 * np.array(res[1]) + (1 - alpha) ** 2 * np.array(parameters[-1][1])+matrix_noise))    

        # 统计本轮分布“最好能有多好”。
        L_mean = [sample_score[indices[0]]]  # 初始化平均得分列表，包含最佳样本的得分
        for k in range(29):
            L_mean.append(Tetris.simulation(best_sample))  # 进行剩余29次模拟，添加得分

        print(np.mean(L_mean))  # 打印平均得分
        L_plot.append(L_mean)   # 存储平均得分列表
        t += 1  # 时间步增加
        print(L_plot, L_norm, mean)  # 打印当前迭代的结果
    return L_plot, L_norm, mean  # 返回模拟结果，范数列表和最终均值 # type: ignore
