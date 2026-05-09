import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

# =========================================================
# PROJECT: AI-Driven Autonomous Mine Dispatching System
# AUTHOR: S.Reza Parsa
# DESCRIPTION: Smart Mining Assistant with Illustration

# REFERENCES:

# 1. Mirzaei-Nasirabad, H., Mohtasham, M., Askari-Nasab, H. et al.
# An optimization model for the real-time truck dispatching problem in open-pit mining operations.
# Optim Eng 24, 2449–2473 (2023). https://doi.org/10.1007/s11081-022-09780-x.

# 2. Mirzaei Nasirabad, H., Mohtasham, M., Rahimzadeh-Nanekaran, F. (2023).
# 'Evaluating the Effect of Fleet Management on the Performance of Mining Operations Using Integer Linear Programming Approach and Two Different Strategies',
# Interdisciplinary Journal of Management Studies, 16(1), pp. 139-155.
# doi: 10.22059/ijms.2022.330990.674769

# 3. Mehrnaz Mohtasham, Hossein Mirzaei-Nasirabad, Hooman Askari-Nasab & Behrooz Alizadeh (2021),
# A multi-objective model for fleet allocation schedule in open-pit mines considering the impact of prioritising objectives on transportation system performance,
# International Journal of Mining,
# Reclamation and Environment, 35:10, 709-727,
# DOI: 10.1080/17480930.2021.1949861

# 4. Yonggang Chang & Huizhi Ren & Shijie Wang, 2015.
# "Modelling and Optimizing an Open-Pit Truck Scheduling Problem",
# Discrete Dynamics in Nature and Society,
# Hindawi, vol. 2015, pages 1-8, March.

# 5. Arelovich, F. Masson, O. Agamennoni, S. Worrall and E. Nebot,
# "Heuristic rule for truck dispatching in open-pit mines with local information-based decisions,
# " 13th International IEEE Conference on Intelligent Transportation Systems,
# Funchal, Portugal, 2010, pp. 1408-1414,
# doi: 10.1109/ITSC.2010.5625231.

# 6. Temeng, V. A., Otuonye, F. O., & Frendewey, J. O. (1998).
# A Nonpreemptive Goal Programming Approach to Truck Dispatching in Open Pit Mines.
# Mineral Resources Engineering, 7(2), 59-67.
# =========================================================

# --- Configuration ---
TARGET_PRODUCTION = np.array([2000, 1500, 2500])
CAPACITY = 50
EPISODES = 1000
STEPS_PER_EP = 80   # Increased steps to allow reaching high production targets
MODEL_PATH = "mining_model.pth"

class MineEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the environment for a new shift"""
        self.queues = np.zeros(3)
        self.production = np.zeros(3)
        self.steps = 0
        return self._get_state()

    def _get_state(self):
        """Returns current state: Queues and production gaps"""
        gaps = TARGET_PRODUCTION - self.production
        return np.concatenate([self.queues, gaps])

    def step(self, action):
        """Executes one dispatch decision"""
        self.queues[action] += 1
        loaded_this_step = [0, 0, 0]
        tonnage_loaded_now = 0

        for i in range(3):
            # Success rate for loading (Optimized for faster learning)
            if self.queues[i] > 0 and random.random() > 0.4:
                self.queues[i] -= 1
                self.production[i] += CAPACITY
                loaded_this_step[i] = 1
                tonnage_loaded_now += CAPACITY

        # --- Reward Logic Optimization ---
        # Positive reward for production to encourage activity
        # Mild penalty for queue accumulation
        reward = (tonnage_loaded_now / 5.0) - (np.sum(self.queues) * 1.5)

        # Bonus for reaching specific shovel targets
        for i in range(3):
            if self.production[i] >= TARGET_PRODUCTION[i] and self.production[i] < TARGET_PRODUCTION[i] + 50:
                reward += 30

        self.steps += 1
        done = self.steps >= STEPS_PER_EP
        return self._get_state(), reward, done, loaded_this_step

class QNetwork(nn.Module):
    """Deep Q-Network Architecture"""
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
    def forward(self, x): return self.fc(x)

class Agent:
    """The Intelligent Dispatcher Agent"""
    def __init__(self):
        self.model = QNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.gamma = 0.98
        self.epsilon = 1.0
        self.epsilon_decay = 0.996
        self.epsilon_min = 0.05
        self.memory = []

        # Load existing weights if the model file exists
        if os.path.exists(MODEL_PATH):
            print(f"\n[SYSTEM] Persistent Memory Linked: Loading {MODEL_PATH}")
            checkpoint = torch.load(MODEL_PATH)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"[SYSTEM] Intelligence Restored. Current Epsilon: {self.epsilon:.4f}\n")

    def act(self, state):
        """Action selection using Epsilon-Greedy policy"""
        if random.random() < self.epsilon:
            return random.randint(0, 2), "Exploring (Random Choice)"
        state_t = torch.FloatTensor(state)
        return torch.argmax(self.model(state_t)).item(), "Strategic (AI Decision)"

    def remember(self, s, a, r, s2, d):
        """Store experience in replay memory"""

        self.memory.append((s, a, r, s2, d))
        if len(self.memory) > 10000: self.memory.pop(0)

    def replay(self, batch_size=64):
        """Train the model using random samples from memory"""
        if len(self.memory) < batch_size: return
        samples = random.sample(self.memory, batch_size)
        for s, a, r, s2, d in samples:
            s_t, s2_t = torch.FloatTensor(s), torch.FloatTensor(s2)
            q_values = self.model(s_t)
            target_q = q_values.clone().detach()
            if d: target_q[a] = r
            else: target_q[a] = r + (self.gamma * torch.max(self.model(s2_t)))
            loss = nn.MSELoss()(q_values, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_decay

# --- Main Logic with Human-Friendly Narrative ---
print("=" * 65)
print("SYSTEM BOOT: AI MINE DISPATCHER (REZA'S LOGIC)")
print(f"Project Lead: S.Reza Parsa")
print(f"Current Targets: {list(TARGET_PRODUCTION)} tons (Shovels 1, 2, 3)")
print("=" * 65)

env = MineEnv()
agent = Agent()

try:
    for episode in range(1, EPISODES + 1):
        state = env.reset()
        total_reward = 0
        print(f"\n--- COMMENCING SHIFT: Episode {episode} ---")

        for step in range(1, STEPS_PER_EP + 1):
            action, mode = agent.act(state)
            next_state, reward, done, loaded = env.step(action)

            # Real-time shift reporting
            current_queues = list(map(int, state[:3]))
            print(f"Step {step}: {mode} -> Dispatching to Shovel {action+1}. Queues: {current_queues}")

            for i, status in enumerate(loaded):
                if status:
                    print(f"   [INFO] Shovel {i+1} completed loading (+50 tons).")

            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward

        # Final shift summary
        final_prod = list(map(int, env.production))
        print(f"\n>> SHIFT REPORT (Episode {episode})")
        print(f"   Final Production: {final_prod} tons")
        print(f"   Efficiency Score: {total_reward:.2f}")

        # Narrative analysis based on shift performance
        if np.all(env.production >= TARGET_PRODUCTION):
            print("   Narrator: Outstanding coordination! All production targets have been achieved.")
        elif total_reward > 0:
            print("   Narrator: Excellent work. The AI is successfully balancing production and queues.")
        else:
            print("   Narrator: The AI is still identifying patterns to reduce bottlenecks.")

        # Save model state after each episode for persistence
        torch.save({
            'model_state_dict': agent.model.state_dict(),
            'epsilon': agent.epsilon
        }, MODEL_PATH)
        print(f"   [SYSTEM] Knowledge saved to {MODEL_PATH}")
        print("-" * 65)

except KeyboardInterrupt:
    print("\n\n[STOP] Manual override detected. Progress safely stored.")

print(f"\nTraining completed. Optimized by Parsa's logic.")
