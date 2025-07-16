import random
import matplotlib.pyplot as plt


# 生成两种不同特征的访问序列
def generate_sequences():
    # LFU友好序列:有一些频繁访问的数据
    lfu_friendly = []
    hot_data = list(range(1, 11))  # 热点数据 1~10
    cold_data = list(range(11, 1001))  # 冷数据 11~100
    for _ in range(200000):  # 序列长度增大
        if random.random() < 0.7:  # 80%概率访问热点数据
            lfu_friendly.append(random.choice(hot_data))
        else:
            lfu_friendly.append(random.choice(cold_data))
    # LRU友好序列:局部性较强的访问模式，任务号区间101~200
    lru_friendly = []
    window_size = 6  # 窗口增大
    current_window = list(range(1001, 1001 + window_size))
    for _ in range(200000):  # 序列长度增大
        if random.random() < 0.9:  # 90%概率访问当前窗口
            lru_friendly.append(random.choice(current_window))
        else:
            # 窗口移动
            new_num = max(current_window) + 1
            current_window = current_window[1:] + [new_num]
            lru_friendly.append(new_num)
    return lfu_friendly, lru_friendly


# LRU缓存实现
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
        self.hits = 0
        self.total = 0

    def get(self, key):
        self.total += 1
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            self.hits += 1
            return self.cache[key]
        return -1

    def put(self, key):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            lru_key = self.order.pop(0)
            del self.cache[lru_key]
        self.cache[key] = True
        self.order.append(key)


# LFU缓存实现
class LFUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.freq = {}
        self.hits = 0
        self.total = 0

    def get(self, key):
        self.total += 1
        if key in self.cache:
            if key not in self.freq:
                self.freq[key] = 1  # 频率字典被清空后，重新初始化
            else:
                self.freq[key] += 1
            self.hits += 1
            return self.cache[key]
        return -1

    def put(self, key):
        if key in self.cache:
            self.freq[key] += 1
        else:
            if len(self.cache) >= self.capacity:
                # 频率字典为空时不淘汰
                if self.freq:
                    min_freq = min(self.freq.values())
                    for k, v in self.freq.items():
                        if v == min_freq:
                            del self.cache[k]
                            del self.freq[k]
                            break
            self.cache[key] = True
            self.freq[key] = 1


# RLCache缓存实现（Q-learning，动作：选择LRU或LFU淘汰）
class RLCache:
    def __init__(self, capacity, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.capacity = capacity
        self.cache = {}
        self.freq = {}
        self.order = []
        self.hits = 0
        self.total = 0
        self.epsilon = epsilon  # 探索概率
        self.alpha = alpha      # 学习率
        self.gamma = gamma      # 折扣因子
        # Q表，key: (state), value: [Q_LRU, Q_LFU]
        self.Q = {}
        self.last_state = 0  # 上一次是否命中（0/1）
        self.last_action = 0 # 上一次动作（0:LRU, 1:LFU）

    def get(self, key):
        self.total += 1
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        return -1

    def put(self, key):
        # 状态：上一次是否命中
        state = self.last_state
        # ε-贪婪选择动作
        if random.random() < self.epsilon:
            action = random.choice([0, 1])  # 0:LRU, 1:LFU
        else:
            q = self.Q.get(state, [0, 0])
            action = 0 if q[0] >= q[1] else 1
        # 淘汰策略
        if key not in self.cache and len(self.cache) >= self.capacity:
            if action == 0:  # LRU
                lru_key = self.order.pop(0)
                del self.cache[lru_key]
                if lru_key in self.freq:
                    del self.freq[lru_key]
            else:  # LFU
                if self.freq:
                    min_freq = min(self.freq.values())
                    for k, v in self.freq.items():
                        if v == min_freq:
                            if k in self.order:
                                self.order.remove(k)
                            del self.cache[k]
                            del self.freq[k]
                            break
        # 更新cache和order
        if key in self.cache:
            if key in self.order:
                self.order.remove(key)
            if key in self.freq:
                self.freq[key] += 1
            else:
                self.freq[key] = 1
        else:
            self.freq[key] = 1
        self.cache[key] = True
        self.order.append(key)
        # Q-learning更新
        reward = 1 if key in self.cache else 0  # 命中为1，否则0
        next_state = reward
        q = self.Q.get(state, [0, 0])
        next_q = self.Q.get(next_state, [0, 0])
        q[action] = q[action] + self.alpha * (reward + self.gamma * max(next_q) - q[action])
        self.Q[state] = q
        # 更新last_state/last_action
        self.last_state = next_state
        self.last_action = action


# 主测试函数
def test_caches():
    lfu_friendly, lru_friendly = generate_sequences()
    cache_size = 5

    # 保存序列到txt文件
    with open('lfu_friendly.txt', 'w') as f:
        for num in lfu_friendly:
            f.write(f"{num}\n")
    with open('lru_friendly.txt', 'w') as f:
        for num in lru_friendly:
            f.write(f"{num}\n")

    # 记录命中率随访问长度变化（滑动窗口）
    def run_and_record(seq, cache_cls, clear_freq_every=None, window_size=500, rl_params=None):
        if cache_cls == RLCache:
            cache = cache_cls(cache_size, **(rl_params or {}))
        else:
            cache = cache_cls(cache_size)
        hit_list = []
        for idx, num in enumerate(seq):
            prev_hits = cache.hits
            cache.get(num)
            cache.put(num)
            hit_list.append(cache.hits - prev_hits)
            if clear_freq_every and isinstance(cache, LFUCache) and (idx + 1) % clear_freq_every == 0:
                cache.freq = {}
            if clear_freq_every and isinstance(cache, RLCache) and (idx + 1) % clear_freq_every == 0:
                cache.freq = {}
        # 计算滑动窗口命中率
        hit_rates = []
        for i in range(len(hit_list)):
            window = hit_list[max(0, i - window_size + 1):i + 1]
            hit_rates.append(sum(window) / len(window))
        return hit_rates

    # LFU友好序列
    lru_hit_lfu_seq = run_and_record(lfu_friendly, LRUCache)
    lfu_hit_lfu_seq = run_and_record(lfu_friendly, LFUCache, clear_freq_every=200)
    rl_hit_lfu_seq = run_and_record(lfu_friendly, RLCache, clear_freq_every=200, rl_params={'epsilon':0.1, 'alpha':0.1, 'gamma':0.9})
    # LRU友好序列
    lru_hit_lru_seq = run_and_record(lru_friendly, LRUCache)
    lfu_hit_lru_seq = run_and_record(lru_friendly, LFUCache, clear_freq_every=200)
    rl_hit_lru_seq = run_and_record(lru_friendly, RLCache, clear_freq_every=200, rl_params={'epsilon':0.1, 'alpha':0.1, 'gamma':0.9})
    # 交替混合序列
    mixed_seq = []
    for i in range(8):
        if i % 2 == 0:
            mixed_seq += lfu_friendly[:20000]
        else:
            mixed_seq += lru_friendly[:20000]
    # 只取前16000个
    mixed_seq = mixed_seq[:160000]
    # 保存混合序列
    with open('mixed_seq1.txt', 'w') as f:
        for num in mixed_seq:
            f.write(f"{num}\n")
    lru_hit_mixed = run_and_record(mixed_seq, LRUCache)
    lfu_hit_mixed = run_and_record(mixed_seq, LFUCache, clear_freq_every=200)
    rl_hit_mixed = run_and_record(mixed_seq, RLCache, clear_freq_every=200, rl_params={'epsilon':0.1, 'alpha':0.1, 'gamma':0.9})
    # 绘图
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(lru_hit_lfu_seq, label='LRU')
    plt.plot(lfu_hit_lfu_seq, label='LFU')
    plt.plot(rl_hit_lfu_seq, label='RL')
    plt.title('LFUsequence')
    plt.xlabel('number of access')
    plt.ylabel('cumulative hit rate')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(lru_hit_lru_seq, label='LRU')
    plt.plot(lfu_hit_lru_seq, label='LFU')
    plt.plot(rl_hit_lru_seq, label='RL')
    plt.title('LRUsequence')
    plt.xlabel('number of access')
    plt.ylabel('cumulative hit rate')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(lru_hit_mixed, label='LRU')
    plt.plot(lfu_hit_mixed, label='LFU')
    plt.plot(rl_hit_mixed, label='RL')
    plt.title('Mixed sequence (alternate every 2000, 8 times)')
    plt.xlabel('number of access')
    plt.ylabel('cumulative hit rate')
    plt.legend()
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    test_caches()
