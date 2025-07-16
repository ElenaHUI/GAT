import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------ 参数设置 ------------------
N_BS = 5         # 基站数量
M_UE = 5         # 每个基站下用户数
K = 10           # 服务类型数
T = 100          # 每个episode的时隙数
EPISODES = 2000  # 训练轮数
C_j = [100] * N_BS  # 每个基站缓存容量 100GB
c_k = [20] * K      # 每种服务类型所需缓存 20GB
omega_d, omega_e = 0.05, 0.01
p_min, p_max = 0.03, 2.0
w = 30000000           # rj,j=2Mbit/s
N0 = 1e-8
f_j = [2.5e9] * N_BS  # 2.5GHz
eta_1, eta_2 = 1.0, 1.0
gamma = 0.9
TAU = 0.01           # 软更新率

# ------------------ 信道增益建模 ------------------
def generate_ue_positions(bs_radius, m_ue):
    theta = np.random.uniform(0, 2*np.pi, m_ue)
    r = np.sqrt(np.random.uniform(0, 1, m_ue)) * bs_radius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)

def calc_channel_gain(ue_pos):
    d = np.linalg.norm(ue_pos, axis=1) + 1e-3
    return d ** (-3)

# ------------------ 任务生成 ------------------
def generate_zipf_tasks(M, K, T, delta, order=None):
    if order is None:
        order = np.arange(K)
    zipf_probs = np.array([1/(order[k]+1)**delta for k in range(K)])
    zipf_probs /= zipf_probs.sum()
    tasks = np.random.choice(K, size=(M, T), p=zipf_probs)
    return tasks

def generate_joint_tasks(M, K, T):
    seg = T // 8
    tasks = np.zeros((M, T), dtype=int)
    for seg_idx in range(8):
        start = seg_idx * seg
        end = (seg_idx + 1) * seg if seg_idx < 7 else T
        if seg_idx % 2 == 0:
            hot = np.arange(3)
            cold = np.arange(3, K)
            for m in range(M):
                for t in range(start, end):
                    if np.random.rand() < 0.8:
                        tasks[m, t] = np.random.choice(hot)
                    else:
                        tasks[m, t] = np.random.choice(cold)
        else:
            window_size = 3
            for m in range(M):
                for t in range(start, end):
                    window = np.arange((t//window_size) % (K-window_size+1), (t//window_size) % (K-window_size+1) + window_size)
                    if np.random.rand() < 0.9:
                        tasks[m, t] = np.random.choice(window)
                    else:
                        tasks[m, t] = np.random.randint(0, K)
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
        self.bs_radius = 200
        self.ue_pos = [generate_ue_positions(self.bs_radius, self.M_UE) for _ in range(self.N_BS)]
        self.g_ij = np.array([calc_channel_gain(self.ue_pos[j]) for j in range(self.N_BS)])
        self.delta = [0.8, 1, 0.8, 1, 1.2]
        self.zipf_orders = [np.arange(K), np.arange(K), np.random.permutation(K), np.random.permutation(K), np.random.permutation(K)]
        self.reset()
        self.hit_count = 0
        self.total_count = 0

    def reset(self):
        self.beta = np.zeros((self.N_BS, self.K), dtype=int)
        self.p = np.ones((self.N_BS, self.M_UE)) * p_min
        self.tasks = [generate_joint_tasks(self.M_UE, self.K, T) for _ in range(self.N_BS)]
        self.t = 0
        self.d = np.random.uniform(10, 30, (self.N_BS, self.M_UE, T))
        self.rho = np.ones((self.N_BS, self.M_UE, T)) * 700
        self.tau = np.random.uniform(0.05, 0.2, (self.N_BS, self.M_UE, T))
        self.hit_count = 0
        self.total_count = 0
        return self.get_state()

    def get_state(self):
        obs = []
        for j in range(self.N_BS):
            o = np.concatenate([
                self.tasks[j][:, self.t] / (self.K-1),
                self.d[j,:,self.t] / 1000,
                self.rho[j,:,self.t] / 1000,
                self.tau[j,:,self.t] / 0.2,
                self.beta[j]
            ])
            obs.append(o)
        return obs

    def step(self, actions):
        rewards = []
        next_beta = []
        next_q = []
        next_p = []
        total_energy = 0.0
        total_delay = 0.0
        for j in range(self.N_BS):
            beta = (actions[j][:self.K] > 0).astype(int)
            q = np.round((actions[j][self.K:self.K+self.M_UE] + 1) / 2 * 2).astype(int)
            q = np.clip(q, 0, 2)
            p = np.clip(actions[j][self.K+self.M_UE:], p_min, p_max)
            if np.sum(beta * self.c_k) > self.C_j[j]:
                beta = self.beta[j]
            next_beta.append(beta)
            next_q.append(q)
            next_p.append(p)
        self.beta = np.array(next_beta)
        self.p = np.array(next_p)

        reward = 0
        for j in range(self.N_BS):
            U_j = eta_1 * int(self.C_j[j] - np.sum(self.beta[j] * self.c_k) >= 0)
            sum_cost = 0
            for i in range(self.M_UE):
                k = self.tasks[j][i, self.t]
                self.total_count += 1
                if self.beta[j, k] == 1:
                    self.hit_count += 1
                d_ = self.d[j, i, self.t]
                rho_ = self.rho[j, i, self.t]
                tau_ = self.tau[j, i, self.t]
                q_ = next_q[j][i]
                if self.beta[j, k] == 1:
                    T_tran = d_ / (w * np.log2(1 + self.p[j, i] * self.g_ij[j, i] / N0))
                    T_comp = d_ * rho_ / self.f_j[j]
                    T_total = T_tran + T_comp
                    E = self.p[j, i] * T_tran
                else:
                    if q_ == 0:
                        local_comp_ability = 1
                        local_power = 2
                        T_total = d_ / local_comp_ability
                        E = local_power * T_total
                    elif q_ == 1:
                        T_tran = d_ / (w * np.log2(1 + self.p[j, i] * self.g_ij[j, i] / N0))
                        T_comp = d_ * rho_ / self.f_j[j]
                        T_total = T_tran + T_comp
                        E = self.p[j, i] * T_tran
                    elif q_ == 2:
                        T_tran = d_ / 500000
                        T_total = T_tran
                        E = self.p[j, i] * T_tran
                    else:
                        T_total = 1.0
                        E = 1.0
                cost = omega_d * T_total + omega_e * E
                Y = eta_2 * int(tau_ - T_total >= 0)
                sum_cost += (Y - cost)
                total_energy += E
                total_delay += T_total
            rewards.append(U_j + sum_cost)
        done = self.t >= T - 1
        next_state = self.get_state()
        self.t += 1
        return next_state, rewards, done, {'energy': total_energy, 'delay': total_delay}

    def get_hit_rate(self):
        if self.total_count == 0:
            return 0.0
        return self.hit_count / self.total_count

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

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('使用GPU加速')
else:
    device = torch.device('cpu')
    print('使用CPU')

env = MECEnv()
obs_dim = env.get_state()[0].shape[0]
act_dim = K + M_UE + M_UE

actors = [Actor(obs_dim, act_dim).to(device) for _ in range(N_BS)]
critics = [Critic(obs_dim*N_BS, act_dim*N_BS).to(device) for _ in range(N_BS)]
actor_opts = [optim.Adam(actors[j].parameters(), lr=1e-4) for j in range(N_BS)]
critic_opts = [optim.Adam(critics[j].parameters(), lr=1e-3) for j in range(N_BS)]
buffers = [ReplayBuffer(max_size=500000) for _ in range(N_BS)]
hit_rates = []
rewards_history = []
energy_history = []
delay_history = []

SLOT_UPDATE_INTERVAL = 10

for episode in range(EPISODES):
    obs = env.reset()
    episode_rewards = np.zeros(N_BS)
    episode_energy = 0.0
    episode_delay = 0.0
    for t in range(T):
        actions = []
        for j in range(N_BS):
            o = torch.tensor(obs[j], dtype=torch.float32, device=device)
            a = actors[j](o).detach().cpu().numpy()
            a[K+M_UE:] = (a[K+M_UE:] + 1) / 2 * (p_max - p_min) + p_min
            actions.append(a)

        next_obs, rewards, done, info = env.step(actions)
        for j in range(N_BS):
            buffers[j].push((obs[j], actions[j], rewards[j], next_obs[j], done))
            episode_rewards[j] += rewards[j]
        episode_energy += info['energy']
        episode_delay += info['delay']
        obs = next_obs

        if (t + 1) % SLOT_UPDATE_INTERVAL == 0:
            hit_rates.append(env.get_hit_rate())

        if done:
            break

        batch_size = 32
        if len(buffers[0].buffer) > batch_size:
            batch_data = [buffers[j].sample(batch_size) for j in range(N_BS)]
            obs_b_list = [torch.tensor(np.array([b[0] for b in batch_data[j]]), dtype=torch.float32, device=device) for j in range(N_BS)]
            act_b_list = [torch.tensor(np.array([b[1] for b in batch_data[j]]), dtype=torch.float32, device=device) for j in range(N_BS)]
            rew_b_list = [torch.tensor(np.array([b[2] for b in batch_data[j]]), dtype=torch.float32, device=device).unsqueeze(1) for j in range(N_BS)]
            next_obs_b_list = [torch.tensor(np.array([b[3] for b in batch_data[j]]), dtype=torch.float32, device=device) for j in range(N_BS)]
            done_b_list = [torch.tensor(np.array([b[4] for b in batch_data[j]]), dtype=torch.float32, device=device).unsqueeze(1) for j in range(N_BS)]

            for j in range(N_BS):
                joint_obs_b = torch.cat(obs_b_list, dim=1)
                joint_act_b = torch.cat(act_b_list, dim=1)
                with torch.no_grad():
                    next_act_b_list = [actors[i](next_obs_b_list[i]) for i in range(N_BS)]
                    next_joint_act_b = torch.cat(next_act_b_list, dim=1)
                    target_q = rew_b_list[j] + gamma * critics[j](torch.cat(next_obs_b_list, dim=1), next_joint_act_b)
                q = critics[j](joint_obs_b, joint_act_b)
                critic_loss = nn.MSELoss()(q, target_q)
                critic_opts[j].zero_grad()
                critic_loss.backward()
                critic_opts[j].step()

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
    energy_history.append(episode_energy)
    delay_history.append(episode_delay)
    print(f"Episode {episode + 1}, 平均奖励: {episode_rewards.mean():.2f}, 当前命中率: {env.get_hit_rate():.3f}, 能耗: {episode_energy:.2f}, 时延: {episode_delay:.2f}")

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.plot(rewards_history)
plt.xlabel("episodes")
plt.ylabel("reward")
plt.subplot(2,2,2)
plt.plot(np.arange(len(hit_rates)) * SLOT_UPDATE_INTERVAL)
plt.xlabel("slot")
plt.ylabel("hit rate")
plt.subplot(2,2,3)
plt.plot(energy_history)
plt.xlabel("episodes")
plt.ylabel("energy consumption")
plt.subplot(2,2,4)
plt.plot(delay_history)
plt.xlabel("episodes")
plt.ylabel("total delay")
plt.tight_layout()
plt.show()
