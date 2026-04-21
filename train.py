import numpy as np
from environment import LearningEnvironment
from agents.q import QLearningAgent


def train_q_learning(n_episodes=500):
    env = LearningEnvironment()
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.action_size,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )

    rewards_per_episode = []

    for episode in range(n_episodes):
        state = env.reset()
        state_idx = env.state_to_index(state)
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state_idx)
            next_state, reward, done, _ = env.step(action)
            next_state_idx = env.state_to_index(next_state)

            agent.update(state_idx, action, reward, next_state_idx, done)

            state_idx = next_state_idx
            total_reward += reward

        rewards_per_episode.append(total_reward)

    agent.epsilon = 0.0
    return agent, rewards_per_episode