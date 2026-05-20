import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from env.swarm_env import SwarmEnv
from agents.maddpg import MADDPG
from agents.replay_buffer import ReplayBuffer
import config

env    = SwarmEnv(n_agents=config.N_AGENTS, n_obstacles=config.N_DYN_OBSTACLES)
maddpg = MADDPG(
    n_agents   = config.N_AGENTS,
    obs_dim    = env.obs_dim,
    action_dim = env.action_dim,
    hidden     = config.HIDDEN_DIM,
    lr_actor   = config.LR_ACTOR,
    lr_critic  = config.LR_CRITIC,
    gamma      = config.GAMMA,
    tau        = config.TAU
)
buffer = ReplayBuffer(capacity=config.BUFFER_SIZE)
writer = SummaryWriter('runs/experiment1')

best_reward = -999999

for episode in range(config.MAX_EPISODES):
    obs_all    = env.reset()
    ep_reward  = 0
    ep_goals   = 0
    ep_collisions = 0

    for step in range(config.MAX_STEPS):
        actions = maddpg.select_actions(obs_all, noise=config.EXPLORATION_NOISE)
        next_obs, rewards, dones, _ = env.step(actions)
        buffer.push(obs_all, actions, rewards, next_obs, dones)
        obs_all    = next_obs
        ep_reward += sum(rewards)
        ep_goals  += sum(dones)
        maddpg.update(buffer, config.BATCH_SIZE)

    writer.add_scalar('reward/episode',     ep_reward, episode)
    writer.add_scalar('goals/episode',      ep_goals,  episode)

    if episode % 100 == 0:
        print(f'Episode {episode:4d} | Reward: {ep_reward:8.2f} | Goals reached: {ep_goals}')

    if ep_reward > best_reward:
        best_reward = ep_reward
        maddpg.save('checkpoints/best_model.pt')

writer.close()
print('Training complete!')
