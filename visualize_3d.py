"""
visualize_3d.py - 3D PyBullet visualization with safety filter + online learning.

Three layers of defense against collisions:
  1. The trained MADDPG actor policy (learned long-horizon goal navigation).
  2. A reactive safety filter on top - artificial potential fields that
     repel each agent away from nearby agents and obstacles before the
     action is executed. This dramatically reduces collisions even when
     the underlying policy is imperfect.
  3. Continuous online learning - every transition is stored, every
     collision is logged to logs/collisions.jsonl, and after a warmup the
     MADDPG update loop runs live so the policy keeps improving.

The safety filter uses pure potential-field repulsion. For each agent we
compute a repulsion vector from every other agent within DANGER_AGENT and
every obstacle within DANGER_OBSTACLE, with magnitude proportional to
(danger_radius - distance) / danger_radius (so it's strongest when about
to collide and zero when far). This vector is added to the actor's action
and clipped to [-1, 1].

The original checkpoint at checkpoints/best_model.pt is preserved.
Improvements are saved to checkpoints/best_model_finetuned.pt.
"""
import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import json
from datetime import datetime
from env.swarm_env import SwarmEnv
from agents.maddpg import MADDPG
from agents.replay_buffer import ReplayBuffer
import config

AGENT_COLORS = [
    [0.0, 0.83, 0.67, 1.0],
    [0.23, 0.51, 0.96, 1.0],
    [0.65, 0.55, 0.98, 1.0],
    [0.96, 0.62, 0.04, 1.0],
]
GOAL_COLORS = [
    [0.0,  0.71, 0.55, 0.65],
    [0.16, 0.39, 0.78, 0.65],
    [0.51, 0.39, 0.78, 0.65],
    [0.78, 0.47, 0.0,  0.65],
]
OBS_COLOR  = [0.94, 0.27, 0.27, 1.0]
WALL_COLOR = [0.30, 0.35, 0.45, 1.0]
SCALE = 2.5

# ===== Safety filter parameters =====
DANGER_AGENT    = 0.30   # start repelling from other agents within this distance
DANGER_OBSTACLE = 0.35   # start repelling from obstacles within this distance
REPEL_AGENT     = 1.8    # repulsion strength against other agents
REPEL_OBSTACLE  = 2.8    # repulsion strength against obstacles (stronger -> they move)

# ===== Online learning parameters =====
WARMUP_STEPS  = 1500
UPDATE_EVERY  = 10
SAVE_EVERY    = 500
EXPLORE_NOISE = 0.03
BUFFER_SIZE   = 20000
BATCH_SIZE    = 128


def safety_filter(actions, agents, obstacles):
    """Potential-field safety overlay. Pushes actions away from nearby threats."""
    safe = []
    for i, agent in enumerate(agents):
        a = np.array(actions[i], dtype=np.float32)
        repel = np.zeros(2, dtype=np.float32)
        for j, other in enumerate(agents):
            if j == i:
                continue
            diff = agent.pos - other.pos
            d = float(np.linalg.norm(diff))
            if 1e-6 < d < DANGER_AGENT:
                strength = (DANGER_AGENT - d) / DANGER_AGENT
                repel += (diff / d) * strength * REPEL_AGENT
        for o in obstacles:
            diff = agent.pos - o.pos
            d = float(np.linalg.norm(diff))
            if 1e-6 < d < DANGER_OBSTACLE:
                strength = (DANGER_OBSTACLE - d) / DANGER_OBSTACLE
                repel += (diff / d) * strength * REPEL_OBSTACLE
        safe.append(np.clip(a + repel, -1.0, 1.0))
    return safe


def main():
    p.connect(p.GUI)
    p.setGravity(0, 0, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    p.resetDebugVisualizerCamera(
        cameraDistance=5.8, cameraYaw=45, cameraPitch=-42,
        cameraTargetPosition=[0, 0, 0]
    )

    p.loadURDF("plane.urdf")

    arena = SCALE
    wall_h = 0.2
    wall_t = 0.05
    for (px, py, sx, sy) in [
        ( arena,  0,     wall_t, arena),
        (-arena,  0,     wall_t, arena),
        ( 0,      arena, arena,  wall_t),
        ( 0,     -arena, arena,  wall_t),
    ]:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, wall_h])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx, sy, wall_h], rgbaColor=WALL_COLOR)
        p.createMultiBody(0, col, vis, [px, py, wall_h])

    env    = SwarmEnv(n_agents=config.N_AGENTS, n_obstacles=config.N_DYN_OBSTACLES)
    maddpg = MADDPG(config.N_AGENTS, env.obs_dim, env.action_dim)

    base_path      = 'checkpoints/best_model.pt'
    finetuned_path = 'checkpoints/best_model_finetuned.pt'
    if os.path.exists(finetuned_path):
        maddpg.load(finetuned_path)
        print('Loaded fine-tuned model from', finetuned_path)
    elif os.path.exists(base_path):
        maddpg.load(base_path)
        print('Loaded base model from', base_path)
    else:
        print('No trained model found, starting from scratch')

    buffer = ReplayBuffer(capacity=BUFFER_SIZE)

    os.makedirs('logs', exist_ok=True)
    log_path = 'logs/collisions.jsonl'
    log_file = open(log_path, 'a')
    print('Collision log:', log_path)
    os.makedirs('checkpoints', exist_ok=True)

    agent_bodies = []
    agent_caps   = []
    for i in range(config.N_AGENTS):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.15, height=0.30)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.15, length=0.30, rgbaColor=AGENT_COLORS[i])
        body = p.createMultiBody(0, col, vis, [0, 0, 0.15])
        agent_bodies.append(body)
        cap_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[1, 1, 1, 1])
        cap = p.createMultiBody(0, -1, cap_vis, [0, 0, 0.34])
        agent_caps.append(cap)

    goal_bodies = []
    for i in range(config.N_AGENTS):
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.20, length=0.02, rgbaColor=GOAL_COLORS[i])
        body = p.createMultiBody(0, -1, vis, [0, 0, 0.011])
        goal_bodies.append(body)

    obstacle_bodies = []
    for i in range(config.N_DYN_OBSTACLES):
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.20)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.20, rgbaColor=OBS_COLOR)
        body = p.createMultiBody(0, col, vis, [0, 0, 0.20])
        obstacle_bodies.append(body)

    speed_slider = p.addUserDebugParameter("Step delay (sec)", 0.0, 0.5, 0.10)

    obs_all          = env.reset()
    episode          = 1
    steps            = 0
    total_steps      = 0
    total_goals      = 0
    total_collisions = 0
    update_count     = 0
    save_count       = 0
    hud_id           = -1

    print('Safety filter: ON')
    print('Online learning starts after', WARMUP_STEPS, 'data-collection steps.')

    try:
        while p.isConnected():
            # 1. Trained policy proposes actions
            raw_actions = maddpg.select_actions(obs_all, noise=EXPLORE_NOISE)
            # 2. Safety filter overlays repulsion from nearby threats
            actions = safety_filter(raw_actions, env.agents, env.obstacles)
            # 3. Step the world
            next_obs, rewards, dones, _ = env.step(actions)

            steps       += 1
            total_steps += 1
            total_goals += sum(dones)

            # Collision detection + logging
            events = []
            for i in range(len(env.agents)):
                for j in range(i + 1, len(env.agents)):
                    d = float(np.linalg.norm(env.agents[i].pos - env.agents[j].pos))
                    if d < 0.12:
                        events.append({'type': 'agent-agent', 'agents': [i, j], 'distance': d})
                for k, o in enumerate(env.obstacles):
                    d = float(np.linalg.norm(env.agents[i].pos - o.pos))
                    if d < 0.13:
                        events.append({'type': 'agent-obstacle', 'agent': i, 'obstacle': k, 'distance': d})

            if events:
                total_collisions += len(events)
                snapshot = {
                    'timestamp':        datetime.now().isoformat(),
                    'episode':          episode,
                    'step':             steps,
                    'global_step':      total_steps,
                    'agent_positions':  [a.pos.tolist() for a in env.agents],
                    'agent_velocities': [a.vel.tolist() for a in env.agents],
                    'goals':            [g.tolist() for g in env.goals],
                    'obstacles':        [o.pos.tolist() for o in env.obstacles],
                    'actions':          [a.tolist() for a in actions],
                    'rewards':          [float(r) for r in rewards],
                    'events':           events,
                }
                log_file.write(json.dumps(snapshot) + '\n')
                log_file.flush()

            buffer.push(obs_all, actions, rewards, next_obs, dones)
            obs_all = next_obs

            if total_steps >= WARMUP_STEPS and total_steps % UPDATE_EVERY == 0:
                maddpg.update(buffer, batch_size=BATCH_SIZE)
                update_count += 1

            if total_steps > 0 and total_steps % SAVE_EVERY == 0:
                maddpg.save(finetuned_path)
                save_count += 1

            for i, agent in enumerate(env.agents):
                x = float(agent.pos[0]) * SCALE
                y = float(agent.pos[1]) * SCALE
                vel = agent.vel
                yaw = float(np.arctan2(vel[1], vel[0])) if np.linalg.norm(vel) > 1e-3 else 0.0
                quat = p.getQuaternionFromEuler([0, 0, yaw])
                p.resetBasePositionAndOrientation(agent_bodies[i], [x, y, 0.15], quat)
                p.resetBasePositionAndOrientation(agent_caps[i],   [x, y, 0.34], [0, 0, 0, 1])

            for i, goal in enumerate(env.goals):
                x = float(goal[0]) * SCALE
                y = float(goal[1]) * SCALE
                p.resetBasePositionAndOrientation(goal_bodies[i], [x, y, 0.011], [0, 0, 0, 1])

            for i, o in enumerate(env.obstacles):
                x = float(o.pos[0]) * SCALE
                y = float(o.pos[1]) * SCALE
                p.resetBasePositionAndOrientation(obstacle_bodies[i], [x, y, 0.20], [0, 0, 0, 1])

            if total_steps < WARMUP_STEPS:
                mode = 'WARMUP ' + str(total_steps) + '/' + str(WARMUP_STEPS)
            else:
                mode = 'LEARNING (updates ' + str(update_count) + ', saves ' + str(save_count) + ')'

            hud = ('Ep ' + str(episode) +
                   '   Goals ' + str(total_goals) +
                   '   Collisions ' + str(total_collisions) +
                   '   SafetyFilter ON   ' + mode)
            if hud_id == -1:
                hud_id = p.addUserDebugText(hud, [0, 0, 1.6], [1, 1, 1], textSize=1.4)
            else:
                hud_id = p.addUserDebugText(hud, [0, 0, 1.6], [1, 1, 1], textSize=1.4,
                                            replaceItemUniqueId=hud_id)

            if steps >= config.MAX_STEPS or all(dones):
                obs_all = env.reset()
                steps   = 0
                episode += 1

            delay = p.readUserDebugParameter(speed_slider)
            time.sleep(delay)

    except KeyboardInterrupt:
        pass
    finally:
        if total_steps > 0:
            maddpg.save(finetuned_path)
        log_file.close()
        print('---')
        print('Session summary:')
        print('  Total steps:       ', total_steps)
        print('  Episodes completed:', episode - 1)
        print('  Goals reached:     ', total_goals)
        print('  Collisions logged: ', total_collisions)
        print('  Policy updates:    ', update_count)
        print('  Checkpoints saved: ', save_count)
        if p.isConnected():
            p.disconnect()


if __name__ == '__main__':
    main()
