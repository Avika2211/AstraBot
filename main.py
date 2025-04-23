import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv

# Define the drone swarm environment
class DroneSwarmEnv(gym.Env):
    def __init__(self, num_drones=5, grid_size=10):
        super(DroneSwarmEnv, self).__init__()
        self.num_drones = num_drones
        self.grid_size = grid_size
        self.positions = np.zeros((num_drones, 2))  # positions of drones on grid
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Box(low=0, high=grid_size, shape=(num_drones, 2), dtype=np.float32)

    def reset(self):
        self.positions = np.zeros((self.num_drones, 2))
        return self.positions

    def step(self, action):
        drone_idx = np.random.randint(self.num_drones)
        action_taken = self.actions[action]
        if action_taken == 'UP':
            self.positions[drone_idx][1] += 1
        elif action_taken == 'DOWN':
            self.positions[drone_idx][1] -= 1
        elif action_taken == 'LEFT':
            self.positions[drone_idx][0] -= 1
        elif action_taken == 'RIGHT':
            self.positions[drone_idx][0] += 1
        
        reward = -np.sum(self.positions[drone_idx]**2)  # Penalize distance from origin
        done = False
        return self.positions, reward, done, {}

# Main function
def main():
    # Create the environment
    env = DummyVecEnv([lambda: DroneSwarmEnv()])

    # Train using PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    # Test the trained model
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == "__main__":
    main()
