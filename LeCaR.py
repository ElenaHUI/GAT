import numpy as np
import gym
from gym import spaces
import random
import torch
import torch.nn as nn
import torch.optim as optim

# 系统参数
NUM_BS = 5  # 基站数量
NUM_UE_PER_BS = 5  # 每个基站的用户数
NUM_TASK_TYPES = 100  # 任务类型数
CACHE_SIZE = 5  # 每个基站缓存容量（可调整）
TASK_GEN_PROB = 0.6  # 每个用户每时隙生成任务的概率
W1, W2 = 0.5, 0.5  # 加权系数，可调整

# 任务类型定义
np.random.seed(42)
task_profiles = []
for _ in range(NUM_TASK_TYPES):
    service_size = np.random.randint(10, 50)  # MB
    task_size = np.random.randint(1, 100)      # MB
    cpu_cycles = np.random.randint(500, 2000) # MHz
    max_delay = np.random.uniform(0.5, 2.0)   # 秒
    task_profiles.append({
        'service_size': service_size,
        'task_size': task_size,
        'cpu_cycles': cpu_cycles,
        'max_delay': max_delay
    })

# Zipf分布生成任务类型概率
zipf_param = 1.2
zipf_probs = np.random.zipf(zipf_param, NUM_TASK_TYPES)
task_type_probs = np.array([np.sum(zipf_probs == (i+1)) for i in range(NUM_TASK_TYPES)])
task_type_probs = task_type_probs / np.sum(task_type_probs)
def sample_task_type():
    return np.random.choice(NUM_TASK_TYPES, p=task_type_probs)

class Cache:
    def __init__(self, size):
        self.size = size  # 最大总service_size
        self.cache = []
        self.lru_count = {}
        self.lfu_count = {}
        self.service_size_sum = 0  # 当前缓存service_size总和

    def hit(self, service_id):
        return service_id in self.cache

    def access(self, service_id, use_lru=True):
        if service_id in self.cache:
            if use_lru:
                self.lru_count[service_id] = 0
                for k in self.lru_count:
                    if k != service_id:
                        self.lru_count[k] += 1
            else:
                self.lfu_count[service_id] += 1
            return True
        else:
            return False

    def add(self, service_id, lru_prob=1.0):
        global task_profiles
        if service_id in self.cache:
            return False  # 没有新增
        service_size = task_profiles[service_id]['service_size']
        # 检查是否有空间
        while self.service_size_sum + service_size > self.size:
            # 需要淘汰
            if len(self.cache) == 0:
                break
            if np.random.rand() < lru_prob:
                lru_item = max(self.lru_count, key=self.lru_count.get)
                self.cache.remove(lru_item)
                self.service_size_sum -= task_profiles[lru_item]['service_size']
                del self.lru_count[lru_item]
                del self.lfu_count[lru_item]
            else:
                lfu_item = min(self.lfu_count, key=self.lfu_count.get)
                self.cache.remove(lfu_item)
                self.service_size_sum -= task_profiles[lfu_item]['service_size']
                del self.lru_count[lfu_item]
                del self.lfu_count[lfu_item]
        # 添加新内容
        self.cache.append(service_id)
        self.lru_count[service_id] = 0
        self.lfu_count[service_id] = 1
        self.service_size_sum += service_size
        return True  # 新增了内容

class MultiBS_Env(gym.Env):
    def __init__(self):
        super(MultiBS_Env, self).__init__()
        # 状态空间：每个基站的缓存状态 + 每个用户的任务状态
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(NUM_BS, NUM_UE_PER_BS, NUM_TASK_TYPES+1), dtype=np.float32
        )
        # 动作空间：每个基站输出三个决策
        # 1. 卸载决策: [NUM_UE_PER_BS]，每个用户0/1/2
        # 2. 功率分配: [NUM_UE_PER_BS]，连续值[0,1]
        # 3. 缓存替换概率: [1]，连续值[0,1]
        self.action_space = []
        for _ in range(NUM_BS):
            self.action_space.append(spaces.Dict({
                'offload': spaces.MultiDiscrete([3]*NUM_UE_PER_BS),
                'power': spaces.Box(low=0, high=1, shape=(NUM_UE_PER_BS,), dtype=np.float32),
                'cache_prob': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            }))
        self.reset()

    def reset(self):
        # 初始化缓存
        self.caches = [Cache(CACHE_SIZE) for _ in range(NUM_BS)]
        # 初始化统计
        self.cache_hits = 0
        self.cache_queries = 0
        self.total_delay = 0
        self.total_energy = 0
        self.total_cost = 0
        # 初始化任务队列
        self.tasks = [[None for _ in range(NUM_UE_PER_BS)] for _ in range(NUM_BS)]
        # 生成新任务
        self._generate_tasks()
        return self._get_obs()

    def _generate_tasks(self):
        for bs in range(NUM_BS):
            for ue in range(NUM_UE_PER_BS):
                if np.random.rand() < TASK_GEN_PROB:
                    task_type = sample_task_type()
                    self.tasks[bs][ue] = {
                        'type': task_type,
                        'profile': task_profiles[task_type]
                    }
                else:
                    self.tasks[bs][ue] = None

    def _get_obs(self):
        # 状态可自定义，这里简单返回任务类型one-hot和是否有任务
        obs = np.zeros((NUM_BS, NUM_UE_PER_BS, NUM_TASK_TYPES+1), dtype=np.float32)
        for bs in range(NUM_BS):
            for ue in range(NUM_UE_PER_BS):
                if self.tasks[bs][ue] is not None:
                    t = self.tasks[bs][ue]['type']
                    obs[bs, ue, t] = 1
                    obs[bs, ue, -1] = 1
        return obs

    def step(self, actions):
        # actions: list of dicts, 每个基站一个dict
        delay_sum, energy_sum, cost_sum = 0, 0, 0
        cache_hit, cache_query = 0, 0
        over_penalty = 0
        for bs in range(NUM_BS):
            act = actions[bs]
            offload = act['offload']
            power = act['power']
            cache_prob = act['cache_prob'][0]
            for ue in range(NUM_UE_PER_BS):
                task = self.tasks[bs][ue]
                if task is None:
                    continue
                ttype = task['type']
                profile = task['profile']
                # 查询本地缓存
                cache_query += 1
                if self.caches[bs].hit(ttype):
                    cache_hit += 1
                    self.caches[bs].access(ttype, use_lru=(np.random.rand()<cache_prob))
                    upload_size = profile['task_size']
                else:
                    # 查询合作基站
                    found = False
                    for other_bs in range(NUM_BS):
                        if other_bs != bs and self.caches[other_bs].hit(ttype):
                            found = True
                            cache_hit += 1
                            self.caches[other_bs].access(ttype, use_lru=(np.random.rand()<cache_prob))
                            break
                    if found:
                        upload_size = profile['task_size']
                    else:
                        upload_size = profile['service_size'] + profile['task_size']
                        # 缓存替换
                        if offload[ue] == 1:  # 选择基站处理才缓存
                            self.caches[bs].add(ttype, lru_prob=cache_prob)
                # 任务处理决策
                if offload[ue] == 0:  # 用户本地
                    delay = profile['cpu_cycles'] / 1000 * 0.01  # 简化延迟
                    energy = upload_size * 0.01
                elif offload[ue] == 1:  # 基站
                    delay = upload_size / (power[ue]*10+1) + profile['cpu_cycles'] / 2000 * 0.01
                    energy = upload_size * power[ue] * 0.02
                else:  # 云端
                    delay = upload_size / (power[ue]*10+1) + 0.1
                    energy = upload_size * power[ue] * 0.03
                delay_sum += delay
                energy_sum += energy
                cost_sum += W1*delay + W2*energy
            # 检查缓存容量是否超限，超限则惩罚
            if self.caches[bs].service_size_sum > CACHE_SIZE:
                over_penalty += (self.caches[bs].service_size_sum - CACHE_SIZE)
        self.cache_hits += cache_hit
        self.cache_queries += cache_query
        self.total_delay += delay_sum
        self.total_energy += energy_sum
        self.total_cost += cost_sum
        # 生成新任务
        self._generate_tasks()
        obs = self._get_obs()
        reward = -cost_sum - 10 * over_penalty  # 施加惩罚
        done = False
        info = {
            'cache_hit_rate': self.cache_hits / (self.cache_queries+1e-6),
            'avg_delay': self.total_delay / (self.cache_queries+1e-6),
            'avg_energy': self.total_energy / (self.cache_queries+1e-6),
            'total_cost': self.total_cost
        }
        return obs, reward, done, info

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

obs_dim = NUM_UE_PER_BS * (NUM_TASK_TYPES + 1)
act_dim = NUM_UE_PER_BS*3 + NUM_UE_PER_BS + NUM_UE_PER_BS  # 每个用户3维卸载+功率+缓存概率

actors = [Actor(obs_dim, act_dim) for _ in range(NUM_BS)]
critics = [Critic(obs_dim*NUM_BS, act_dim*NUM_BS) for _ in range(NUM_BS)]
actor_opts = [optim.Adam(actor.parameters(), lr=1e-3) for actor in actors]
critic_opts = [optim.Adam(critic.parameters(), lr=1e-3) for critic in critics]
buffer = ReplayBuffer()

env = MultiBS_Env()
EPISODES = 1000
steps_per_episode = 100
gamma = 0.95

reward_history = []
hit_rate_history = []

for ep in range(EPISODES):
    obs = env.reset()
    episode_reward = 0
    for step in range(steps_per_episode):
        obs_tensor = [torch.tensor(obs[bs].flatten(), dtype=torch.float32) for bs in range(NUM_BS)]
        # 联合动作
        act_list = []
        for bs in range(NUM_BS):
            with torch.no_grad():
                act = actors[bs](obs_tensor[bs])
            act_list.append(act)
        # 解析动作
        actions = []
        for bs in range(NUM_BS):
            act = act_list[bs].numpy()
            offload_logits = act[:NUM_UE_PER_BS*3].reshape(NUM_UE_PER_BS, 3)
            power = (act[NUM_UE_PER_BS*3:NUM_UE_PER_BS*4] + 1) / 2  # [-1,1] -> [0,1]
            cache_prob = (act[NUM_UE_PER_BS*4:] + 1) / 2  # [-1,1] -> [0,1]
            # 每个用户独立softmax采样
            offload = np.array([
                np.random.choice(3, p=torch.softmax(torch.tensor(offload_logits[i]), dim=0).numpy())
                for i in range(NUM_UE_PER_BS)
            ])
            actions.append({
                'offload': offload,
                'power': power,
                'cache_prob': cache_prob
            })
        next_obs, reward, done, info = env.step(actions)
        buffer.push((obs, act_list, reward, next_obs))
        obs = next_obs
        episode_reward += reward
        if done:
            break

        # 训练
        if len(buffer.buffer) > 64:
            batch = buffer.sample(64)
            for j in range(NUM_BS):
                state_b = torch.tensor(np.array([b[0][j].flatten() for b in batch]), dtype=torch.float32)
                act_b = torch.stack([b[1][j] for b in batch])
                reward_b = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32).unsqueeze(1)
                next_state_b = torch.tensor(np.array([b[3][j].flatten() for b in batch]), dtype=torch.float32)
                # 联合状态动作
                joint_state_b = torch.cat([torch.tensor(np.array([b[0][i].flatten() for b in batch]), dtype=torch.float32) for i in range(NUM_BS)], dim=1)
                joint_act_b = torch.cat([torch.stack([b[1][i] for b in batch]) for i in range(NUM_BS)], dim=1)
                next_joint_state_b = torch.cat([torch.tensor(np.array([b[3][i].flatten() for b in batch]), dtype=torch.float32) for i in range(NUM_BS)], dim=1)
                next_joint_act_b = torch.cat([actors[i](next_state_b).detach() for i in range(NUM_BS)], dim=1)
                # Critic更新
                q_target = reward_b + gamma * critics[j](next_joint_state_b, next_joint_act_b).detach()
                q_val = critics[j](joint_state_b, joint_act_b)
                critic_loss = nn.MSELoss()(q_val, q_target)
                critic_opts[j].zero_grad()
                critic_loss.backward()
                critic_opts[j].step()
                # Actor更新
                pred_act = actors[j](state_b)
                all_acts = [pred_act if i == j else actors[i](state_b).detach() for i in range(NUM_BS)]
                all_acts_tensor = torch.cat(all_acts, dim=1)
                actor_loss = -critics[j](joint_state_b, all_acts_tensor).mean()
                actor_opts[j].zero_grad()
                actor_loss.backward()
                actor_opts[j].step()

    reward_history.append(episode_reward)
    hit_rate_history.append(info['cache_hit_rate'])
    if ep % 10 == 0:
        print(f"Episode {ep}: reward={episode_reward:.2f}, cache_hit_rate={info['cache_hit_rate']:.3f}")

import matplotlib.pyplot as plt
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(reward_history)
plt.title("Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.subplot(1,2,2)
plt.plot(hit_rate_history)
plt.title("Cache Hit Rate")
plt.xlabel("Episode")
plt.ylabel("Hit Rate")
plt.tight_layout()
plt.show()

