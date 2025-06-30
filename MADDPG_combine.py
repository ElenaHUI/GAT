import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------ 参数设置 ------------------
N_BS = 5         # 基站数量
M_UE = 8         # 每个基站下用户数
K = 5            # 服务类型数
T = 100          # 每个episode的时隙数
EPISODES = 1000    # 训练轮数

C_j = [50] * N_BS
c_k = [10, 15, 20, 10, 15]
omega_d, omega_e = 0.5, 0.5
p_min, p_max = 0.1, 1.0
w = 1e6
N0 = 1e-9
f_j = [10e9] * N_BS
eta_1, eta_2 = 1.0, 1.0
gamma = 0.95

# ------------------ 任务生成（Zipf分布） ------------------
def generate_zipf_tasks(M, K, T):
    a = 1.2
    zipf_probs = np.array([1/(k+1)**a for k in range(K)])
    zipf_probs /= zipf_probs.sum()
    tasks = np.random.choice(K, size=(M, T), p=zipf_probs)
    return tasks

# ------------------ 环境定义 ------------------
class MECEnv:
    def __init__(self):
        self.N_BS = N_BS
        self.M_UE = M_UE
        self.K = K
        self.C_j = C_j
        self.c_k = c_k
        self.f_j = f_j
        self.g_ij = np.random.uniform(1e-4, 1e-3, (N_BS, M_UE))
        self.reset()
        self.hit_count = 0
        self.total_count = 0

    def reset(self):
        self.beta = np.zeros((self.N_BS, self.K), dtype=int)
        self.p = np.ones((self.N_BS, self.M_UE)) * p_min
        self.tasks = [generate_zipf_tasks(self.M_UE, self.K, T) for _ in range(self.N_BS)]
        self.t = 0
        # 随机生成任务参数
        self.d = np.random.uniform(500, 1000, (self.N_BS, self.M_UE, T))  # 数据量
        self.rho = np.random.uniform(500, 1000, (self.N_BS, self.M_UE, T))  # 每比特CPU周期
        self.tau = np.random.uniform(0.05, 0.2, (self.N_BS, self.M_UE, T))  # 最大延迟
        self.hit_count = 0
        self.total_count = 0
        return self.get_state()

    def get_state(self):
        # 返回每个BS的观测（本地UE任务类型、数据量、CPU需求、最大延迟、缓存状态、功率）
        obs = []
        for j in range(self.N_BS):
            o = np.concatenate([
                self.tasks[j][:, self.t] / (self.K-1),  # 归一化服务类型
                self.d[j,:,self.t] / 1000,
                self.rho[j,:,self.t] / 1000,
                self.tau[j,:,self.t] / 0.2,
                self.beta[j]  # 当前缓存
            ])
            obs.append(o)
        return obs

    def step(self, actions):
        # actions: [(beta, p)] for each BS
        rewards = []
        next_beta = []
        next_q = []
        next_p = []
        for j in range(self.N_BS):
            # 动作拆分
            beta = (actions[j][:self.K] > 0).astype(int)
            q = np.round((actions[j][self.K:self.K+self.M_UE] + 1) / 2 * 2).astype(int)
            q = np.clip(q, 0, 2)
            p = np.clip(actions[j][self.K+self.M_UE:], p_min, p_max)
            # 检查缓存容量约束
            if np.sum(beta * self.c_k) > self.C_j[j]:
                beta = self.beta[j]  # 若超限则不变
            next_beta.append(beta)
            next_q.append(q)
            next_p.append(p)
        self.beta = np.array(next_beta)
        self.p = np.array(next_p)

        # 计算奖励
        reward = 0
        for j in range(self.N_BS):
            U_j = eta_1 * int(self.C_j[j] - np.sum(self.beta[j] * self.c_k) >= 0)
            sum_cost = 0
            for i in range(self.M_UE):
                k = self.tasks[j][i, self.t]
                # 统计总任务数
                self.total_count += 1
                # 统计命中
                if self.beta[j, k] == 1:
                    self.hit_count += 1
                k = self.tasks[j][i, self.t]
                d_ = self.d[j, i, self.t]
                rho_ = self.rho[j, i, self.t]
                tau_ = self.tau[j, i, self.t]
                q_ = next_q[j][i]
                if self.beta[j, k] == 1:
                    # 缓存命中，按原逻辑
                    T_tran = d_ / (w * np.log2(1 + self.p[j, i] * self.g_ij[j, i] / N0))
                    T_comp = d_ * rho_ / self.f_j[j]
                    T_total = T_tran + T_comp
                    E = self.p[j, i] * T_tran
                else:
                    if q_ == 0:
                        # 本地计算
                        local_comp_ability = 1 # 设定本地计算能力
                        local_power = 2        # 设定本地功耗
                        T_total = d_ / local_comp_ability
                        E = local_power * T_total
                    elif q_ == 1:
                        # 基站计算
                        T_tran = d_ / (w * np.log2(1 + self.p[j, i] * self.g_ij[j, i] / N0))
                        T_comp = d_ * rho_ / self.f_j[j]
                        T_total = T_tran + T_comp
                        E = self.p[j, i] * T_tran
                    elif q_ == 2:
                        # 云端计算
                        net_delay = 1 # 设定网络时延
                        cloud_comp_ability = 2 # 云端计算能力
                        T_tran = d_ / (w * np.log2(1 + self.p[j, i] * self.g_ij[j, i] / N0))
                        T_comp = d_ * rho_ / cloud_comp_ability
                        T_total = T_tran + net_delay + T_comp
                        E = self.p[j, i] * T_tran
                    else:
                        # 防止q_为非法值时出错
                        T_total = 1.0
                        E = 1.0
                cost = omega_d * T_total + omega_e * E
                Y = eta_2 * int(tau_ - T_total >= 0)
                sum_cost += (Y - cost)
            rewards.append(U_j + sum_cost)
        done = self.t >= T - 1
        next_state = self.get_state()
        self.t += 1
        return next_state, rewards, done, {}
    def get_hit_rate(self):
        if self.total_count == 0:
            return 0.0
        return self.hit_count / self.total_count
# ------------------ MADDPG智能体 ------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.fc(x)

class Critic(nn.Module):
    def __init__(self, state_dim, act_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + act_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, state, act):
        x = torch.cat([state, act], dim=-1)
        return self.fc(x)

# 经验回放池
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def push(self, data):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(data)

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in idx]

# ------------------ 主训练流程 ------------------
env = MECEnv()
obs_dim = env.get_state()[0].shape[0]
act_dim = K + M_UE + M_UE  # K个缓存决策 + M_UE个卸载决策 + M_UE个功率分配

actors = [Actor(obs_dim, act_dim) for _ in range(N_BS)]
critics = [Critic(obs_dim*N_BS, act_dim*N_BS) for _ in range(N_BS)]
actor_opts = [optim.Adam(actors[j].parameters(), lr=1e-3) for j in range(N_BS)]
critic_opts = [optim.Adam(critics[j].parameters(), lr=1e-3) for j in range(N_BS)]
buffers = [ReplayBuffer() for _ in range(N_BS)]
hit_rates = []
rewards_history = []

for episode in range(EPISODES):
    obs = env.reset()
    episode_rewards = np.zeros(N_BS)
    for t in range(T):
        actions = []
        for j in range(N_BS):
            o = torch.tensor(obs[j], dtype=torch.float32)
            a = actors[j](o).detach().numpy()
            # 动作后K位归一化到功率范围
            a[K+M_UE:] = (a[K+M_UE:] + 1) / 2 * (p_max - p_min) + p_min
            actions.append(a)
        # 输出每个基站的动作决策
        for j in range(N_BS):
            beta = (actions[j][:K] > 0).astype(int)
            q = np.round((actions[j][K:K+M_UE] + 1) / 2 * 2).astype(int)
            q = np.clip(q, 0, 2)
            p = actions[j][K+M_UE:]
            #print(f"时隙{t+1}，基站{j+1}：缓存决策={beta}，卸载决策={q}，功率分配={p}")
        next_obs, rewards, done, _ = env.step(actions)
        for j in range(N_BS):
            buffers[j].push((obs[j], actions[j], rewards[j], next_obs[j], done))
            episode_rewards[j] += rewards[j]
        obs = next_obs
        if done:
            break

        # 采样一批数据后，统一训练
        batch_size = 32
        if len(buffers[0].buffer) > batch_size:
            # 先采样所有智能体的batch
            batch_data = [buffers[j].sample(batch_size) for j in range(N_BS)]
            obs_b_list = [torch.tensor(np.array([b[0] for b in batch_data[j]]), dtype=torch.float32) for j in range(N_BS)]
            act_b_list = [torch.tensor(np.array([b[1] for b in batch_data[j]]), dtype=torch.float32) for j in range(N_BS)]
            rew_b_list = [torch.tensor(np.array([b[2] for b in batch_data[j]]), dtype=torch.float32).unsqueeze(1) for j in range(N_BS)]
            next_obs_b_list = [torch.tensor(np.array([b[3] for b in batch_data[j]]), dtype=torch.float32) for j in range(N_BS)]
            done_b_list = [torch.tensor(np.array([b[4] for b in batch_data[j]]), dtype=torch.float32).unsqueeze(1) for j in range(N_BS)]

            # Critic和Actor更新
            for j in range(N_BS):
                # 构造联合观测和动作
                joint_obs_b = torch.cat(obs_b_list, dim=1)      # [batch, N_BS*obs_dim]
                joint_act_b = torch.cat(act_b_list, dim=1)      # [batch, N_BS*act_dim]
                # Critic更新
                with torch.no_grad():
                    # 生成下一个联合动作
                    next_act_b_list = [actors[i](next_obs_b_list[i]) for i in range(N_BS)]
                    next_joint_act_b = torch.cat(next_act_b_list, dim=1)
                    target_q = rew_b_list[j] + gamma * critics[j](torch.cat(next_obs_b_list, dim=1), next_joint_act_b)
                q = critics[j](joint_obs_b, joint_act_b)
                critic_loss = nn.MSELoss()(q, target_q)
                critic_opts[j].zero_grad()
                critic_loss.backward()
                critic_opts[j].step()
                # Actor更新
                act_b_list_for_update = []
                for i in range(N_BS):
                    if i == j:
                        act_b_list_for_update.append(actors[i](obs_b_list[i]))
                    else:
                        act_b_list_for_update.append(actors[i](obs_b_list[i]).detach())
                joint_act_b_for_update = torch.cat(act_b_list_for_update, dim=1)
                actor_loss = -critics[j](joint_obs_b, joint_act_b_for_update).mean()
                actor_opts[j].zero_grad()
                actor_loss.backward()
                actor_opts[j].step()
    rewards_history.append(episode_rewards.mean())
    hit_rate = env.get_hit_rate()
    hit_rates.append(hit_rate)
    print(f"Episode {episode + 1}, 平均奖励: {episode_rewards.mean():.2f}, 命中率: {hit_rate:.3f}")
# ------------------ 结果可视化 ------------------
plt.plot(rewards_history)
plt.xlabel("eposides")
plt.ylabel("reward")
plt.figure()
plt.plot(hit_rates)
plt.xlabel("eposides")
plt.ylabel("hit rate")
plt.show()