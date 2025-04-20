import math
import numpy as np
from scipy import stats
import Tetris


# NOTE:
# 下面描述中:
#   - “参数” 指的是描述某个分布的参数。具体来说，这里指的多元正态分布的参数，包含两部分：均值向量（`mu`）和协方差矩阵（`sigma`）。
#   - “权重向量” 指的是该优化算法要优化的对象。具体来说，这里指的是状态值函数中的每个基函数的权重，大小为 21 （使用了 21 个特征）。
# 注：
#   - 正态分布可以通过均值向量、协方差矩阵两个属性来表示
#   - 分布可以进行采样操作，获得样本；此外，给定一批样本，也可以估计一个分布。
#   - 从分布中采样得到的样本就是一个 “权重向量”，可以送入模拟器得到一个游戏分数
#   - 选择分数比较高的那些样本，然后用它们估计新的分布。
#   - 优化的目标是一个好的分布，从这个分布中采样得到的样本，送到模拟器中都能得到一个不错的分数。
#   - 得到一个好的分布后，使用分布的均值做为最终模拟器使用的 “权重向量”
def simulation_CE(alpha, num_iter, rho): 
    """
    使用交叉熵方法进行模拟优化。

    参数:

        alpha (float): 更新率，用于控制新参数与旧参数的混合程度。
        num_iter (int): 迭代次数，指定算法运行的步数。
        rho (float): 被选择的向量比例，用于确定每轮迭代中保留的顶级样本比例。

    迭代过程:

        1. 基于当前的参数（均值向量和协方差矩阵), 创建多元正态分布。
        2. 生成 N 个样本，并对每个样本进行模拟评估。
        3. 保留得分最高的 rho*N 个样本。
        4. 使用论文中的公式 2.3、2.4 来估计新的分布（计算新的参数)。
        5. 使用 alpha 对新参数与旧参数做线性混合，更新参数列表。
        6. 计算并存储最佳样本在 30 次模拟中的平均得分。
        7. 存储协方差矩阵的范数。（范数下降表明算法收敛）
        8. 打印每次迭代的平均得分和协方差矩阵的范数。

    返回:
    L_plot (list): 每次迭代后最佳样本的模拟结果列表。
    L_norm (list): 每次迭代后协方差矩阵的范数列表。
    mean (np.array): 最终均值向量。
    """
    
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

        # NOTE: 更新参数(公式 2.3、2.4)
        parameters.append((alpha * np.array(res[0]) + (1 - alpha) * np.array(parameters[-1][0]),
                           alpha**2 * np.array(res[1]) + (1 - alpha)**2 * np.array(parameters[-1][1])))
        
        # 统计本轮分布“最好能有多好”。
        L_mean = [sample_score[indices[0]]]  # 初始化平均得分列表，包含最佳样本的得分
        for k in range(29):
            L_mean.append(Tetris.simulation(best_sample))  # 进行剩余29次模拟，添加得分

        print(np.mean(L_mean))  # 打印平均得分
        L_plot.append(L_mean)   # 存储平均得分列表
        t += 1  # 时间步增加
        print(L_plot, L_norm, mean)  # 打印当前迭代的结果
    return L_plot, L_norm, mean  # 返回模拟结果，范数列表和最终均值 # type: ignore
