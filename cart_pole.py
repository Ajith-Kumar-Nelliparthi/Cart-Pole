import gymnasium as gym
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

# Create the environment
env = gym.make('CartPole-v1', render_mode='rgb_array')
BEST_MODEL_FILE = "best_q_table.npy"
best_score = float('-inf')
video_folder = "Videos"

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 20000
SHOW_EVERY = 1000

# Handle infinite observation space values
obs_high = np.where(env.observation_space.high == np.inf, 10, env.observation_space.high)
obs_low = np.where(env.observation_space.low == -np.inf, -10, env.observation_space.low)

# Discretization
DISCRETE_OS_SIZE = [20, 20, 20, 20]
discrete_os_win_size = (obs_high - obs_low) / DISCRETE_OS_SIZE

epsilon = 1.0
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 3
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Create Q-table
q_table = np.random.uniform(low=0, high=1, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    state = np.clip(state, obs_low, obs_high)  # Ensure state values are within bounds
    discrete_state = (state - obs_low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

successful_episode = None
frames = []

for episode in range(EPISODES):
    total_reward = 0
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False
    render = episode % SHOW_EVERY == 0
    
    if render:
        print(f"Episode: {episode}")

    while not done:
        action = np.argmax(q_table[discrete_state]) if np.random.random() > epsilon else np.random.randint(0, env.action_space.n)
        new_state, reward, terminated, truncated, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        done = terminated or truncated

        # Penalize large pole angles to encourage balance
        reward -= abs(new_state[2]) * 10
        
        if render:
            frame = env.render()
            frames.append(frame)
        
        total_reward += reward
        
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        else:
            q_table[discrete_state + (action,)] = reward

        discrete_state = new_discrete_state
    
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon = max(0, epsilon - epsilon_decay_value)
    
    if total_reward > best_score:
        best_score = total_reward
        np.save(BEST_MODEL_FILE, q_table)
        print(f"New best model saved at Episode {episode}, Score: {best_score}")
        successful_episode = episode

env.close()

# Save video of the successful episode
if successful_episode is not None:
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(f"{video_folder}/successs.mp4", codec="libx264")
    print("Saved successful episode as success.mp4")

# Testing the trained model
env = gym.make('CartPole-v1', render_mode='human')
state, _ = env.reset()
discrete_state = get_discrete_state(state)
done = False

while not done:
    action = np.argmax(q_table[discrete_state])
    new_state, reward, terminated, truncated, _ = env.step(action)
    discrete_state = get_discrete_state(new_state)
    done = terminated or truncated
    env.render()

env.close()
