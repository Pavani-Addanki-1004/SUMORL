import gymnasium as gym
import sumo_rl
import os
import pandas as pd
import numpy as np
from datetime import datetime

print("SUMO-RL imported successfully!")

# Enhanced Q-learning implementation with positive rewards
class SimpleQLearning:
    def __init__(self, n_actions, learning_rate=0.1, discount_factor=0.99, epsilon=0.15):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.n_actions = n_actions
        
        # Traffic metrics tracking
        self.previous_queue = 0
        self.previous_waiting_time = 0
        self.previous_stopped = 0
        self.min_waiting_time_seen = float('inf')
        self.max_stopped_seen = 0
        self.last_waiting_time_threshold = 0
        
        # Action tracking
        self.previous_action = None
        self.current_action = None
        
        # Learning parameters
        self.max_positive_reward = 10  # Maximum positive reward
        self.min_epsilon = 0.05  # Higher minimum exploration
        self.epsilon_decay = 0.998  # Slower decay
        
        # Performance tracking
        self.best_actions = {}  # Store best performing actions
        self.action_success_count = {}  # Track successful actions
    
    def get_state_key(self, state):
        # Convert state array to tuple for dictionary key
        return tuple(state.astype(int))
    
    def act(self, state):
        state_key = self.get_state_key(state)
        
        # Initialize state if not seen before
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
            
        # Track this state's best action
        if state_key not in self.best_actions:
            self.best_actions[state_key] = None
            self.action_success_count[state_key] = np.zeros(self.n_actions)
        
        # Adaptive exploration
        if np.random.random() < self.epsilon:
            # Weighted random choice based on action success
            if np.sum(self.action_success_count[state_key]) > 0:
                probs = self.action_success_count[state_key] / np.sum(self.action_success_count[state_key])
                action = np.random.choice(self.n_actions, p=probs)
            else:
                action = np.random.randint(self.n_actions)
        else:
            # Choose best action with some randomness in ties
            max_value = np.max(self.q_table[state_key])
            best_actions = np.where(self.q_table[state_key] == max_value)[0]
            action = np.random.choice(best_actions)
        
        # Adaptive epsilon decay
        if self.best_actions[state_key] is not None:
            # Slower decay if we're performing well
            self.epsilon = max(self.min_epsilon, 
                            self.epsilon * (self.epsilon_decay if action == self.best_actions[state_key] else 0.999))
        else:
            # Normal decay during initial exploration
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        self.current_action = action
        return action
    
    def calculate_reward(self, info):
        # Get current traffic metrics
        current_queue = info.get('system_total_queued', 0)
        waiting_time = info.get('system_total_waiting_time', 0)
        stopped = info.get('system_total_stopped', 0)
        
        # Calculate improvements
        queue_improvement = self.previous_queue - current_queue
        waiting_time_improvement = self.previous_waiting_time - waiting_time
        stopped_improvement = self.previous_stopped - stopped
        
        # Update previous values
        self.previous_queue = current_queue
        self.previous_waiting_time = waiting_time
        self.previous_stopped = stopped
        
        # Base reward starts at 5.0
        reward = 5.0
        
        # Major reward components
        if waiting_time_improvement > 0:
            # Strong incentive for reducing waiting time
            reward += min(3.0, waiting_time_improvement / 1000)
        
        if stopped_improvement > 0:
            # Reward for reducing stopped vehicles
            reward += min(2.0, stopped_improvement * 0.1)
        
        # Penalize if metrics are too high (but keep reward positive)
        if waiting_time > self.last_waiting_time_threshold:
            reward *= 0.8
            # Increase threshold for next comparison
            self.last_waiting_time_threshold = waiting_time
        
        if stopped > self.max_stopped_seen:
            reward *= 0.8
            self.max_stopped_seen = stopped
        
        # Immediate state rewards
        if current_queue == 0:
            reward += 1.0  # Perfect queue condition
        
        if stopped == 0:
            reward += 1.0  # Perfect flow
        
        # Learning guidance rewards
        if self.previous_action is not None:
            if waiting_time < self.min_waiting_time_seen:
                # Found a better traffic management strategy
                reward += 2.0
                self.min_waiting_time_seen = waiting_time
        
        # Update previous action
        self.previous_action = self.current_action
        
        return max(0, min(reward, self.max_positive_reward))
    
    def learn(self, state, action, info, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Calculate positive reward based on traffic metrics
        reward = self.calculate_reward(info)
        
        # Initialize next state if not seen before
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.n_actions)
        
        # Q-learning update
        best_next_action = np.max(self.q_table[next_state_key])
        self.q_table[state_key][action] = (1 - self.lr) * self.q_table[state_key][action] + \
                                        self.lr * (reward + self.gamma * best_next_action)
        return reward  # Return reward for logging

# Create the environment with more traffic and longer duration
out_csv = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

env = gym.make('sumo-rl-v0',
               net_file=r'C:\Users\USER\Desktop\SUMORL\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection.net.xml',
               route_file=r'C:\Users\USER\Desktop\SUMORL\sumo-rl\sumo_rl\nets\2way-single-intersection\single-intersection-vhvh.rou.xml',
               out_csv_name=out_csv,
               use_gui=True,
               num_seconds=3600,
               min_green=5,
               max_green=50)

# Initialize our enhanced Q-Learning agent
initial_state = env.reset()[0]
agent = SimpleQLearning(
    n_actions=env.action_space.n,
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=0.05
)

print("Starting Q-Learning simulation with positive rewards - will run for 3600 simulation seconds...")

total_reward = 0
step = 0
done = False
current_state = initial_state
best_reward = 0
running_avg_reward = []

while not done:
    # Get action from Q-Learning agent
    action = agent.act(current_state)
    
    # Take action in environment
    next_state, _, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Learn from the experience with positive rewards
    reward = agent.learn(current_state, action, info, next_state)
    current_state = next_state
    total_reward += reward
    
    # Track best and average rewards
    best_reward = max(best_reward, reward)
    running_avg_reward.append(reward)
    
    # Print progress every 100 steps
    step += 1
    if step % 100 == 0:
        avg_reward = np.mean(running_avg_reward[-100:])  # Average of last 100 rewards
        print(f"Step {step} completed:")
        print(f"  - Current Reward: {reward:.2f}")
        print(f"  - Average Reward (last 100): {avg_reward:.2f}")
        print(f"  - Total Reward: {total_reward:.2f}")
        print(f"  - Average Queue: {info.get('system_total_queued', 0):.2f}")
        print(f"  - Waiting Time: {info.get('system_total_waiting_time', 0):.2f}")
        print(f"  - Stopped Vehicles: {info.get('system_total_stopped', 0):.2f}")

print(f"\nSimulation complete!")
print(f"Total steps: {step}")
print(f"Final total reward: {total_reward:.2f}")
print(f"Best single reward achieved: {best_reward:.2f}")
print(f"Average reward: {np.mean(running_avg_reward):.2f}")
print(f"Results saved to {out_csv}")

# Close the environment
env.close()

# Read and display the results
try:
    results = pd.read_csv(out_csv)
    print("\nTraffic Performance Metrics:")
    print(f"Average waiting time: {results['system_total_waiting_time'].mean():.2f} seconds")
    print(f"Average queue length: {results['system_total_queued'].mean():.2f} vehicles")
    print(f"Total stopped vehicles: {results['system_total_stopped'].sum()}")
except Exception as e:
    print(f"Could not read results file: {e}")
