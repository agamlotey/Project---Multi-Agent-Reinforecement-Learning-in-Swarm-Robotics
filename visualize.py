import pygame
import numpy as np
import sys
import os
from collections import deque
from env.swarm_env import SwarmEnv
from agents.maddpg import MADDPG
import config

AGENT_COLORS = [(0,212,170),(59,130,246),(167,139,250),(245,158,11)]
GOAL_COLORS  = [(0,180,140),(40,100,200),(130,100,200),(200,120,0)]
OBS_COLOR    = (239,68,68)
BG_COLOR     = (15,20,30)
GRID_COLOR   = (25,30,40)
PANEL_COLOR  = (22,28,40)
TEXT_COLOR   = (200,210,225)
DIM_TEXT     = (130,140,160)
WHITE        = (255,255,255)
FLASH        = (255,80,80)

ARENA  = 700
PANEL  = 280
WIDTH  = ARENA + PANEL
HEIGHT = ARENA
SCALE  = 300
CENTER = np.array([ARENA//2, ARENA//2])

def w2s(pos):
    return (CENTER + pos * SCALE).astype(int)

env    = SwarmEnv(n_agents=config.N_AGENTS, n_obstacles=config.N_DYN_OBSTACLES)
maddpg = MADDPG(config.N_AGENTS, env.obs_dim, env.action_dim)

model_path = 'checkpoints/best_model.pt'
if os.path.exists(model_path):
    maddpg.load(model_path)
    print('Loaded trained model from ' + model_path)
else:
    print('No trained model found, using untrained policy')

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Project 7 - MARL Swarm Navigation')
font_big = pygame.font.SysFont('Helvetica', 22, bold=True)
font_med = pygame.font.SysFont('Helvetica', 16, bold=True)
font_sm  = pygame.font.SysFont('Helvetica', 13)
font_xs  = pygame.font.SysFont('Helvetica', 11)
clock = pygame.time.Clock()

obs_all     = env.reset()
episode     = 1
steps       = 0
ep_goals    = 0
total_goals = 0
collisions  = 0
trails      = [deque(maxlen=50) for _ in range(config.N_AGENTS)]

fps_options = [3, 5, 10, 15, 20, 30, 60]
fps_idx     = 2
paused      = False
show_trails = True
show_safety = True
show_arrows = True

flash_a = [0]*config.N_AGENTS
flash_o = [0]*config.N_DYN_OBSTACLES

def reset_ep():
    return env.reset(), 0, 0, [deque(maxlen=50) for _ in range(config.N_AGENTS)]

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                pygame.quit(); sys.exit()
            if event.key == pygame.K_r:
                obs_all, steps, ep_goals, trails = reset_ep()
                episode += 1
            if event.key == pygame.K_SPACE:
                paused = not paused
            if event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                fps_idx = min(len(fps_options)-1, fps_idx+1)
            if event.key == pygame.K_MINUS:
                fps_idx = max(0, fps_idx-1)
            if event.key == pygame.K_t:
                show_trails = not show_trails
            if event.key == pygame.K_s:
                show_safety = not show_safety
            if event.key == pygame.K_v:
                show_arrows = not show_arrows

    if not paused:
        actions = maddpg.select_actions(obs_all, noise=0.0)
        obs_all, rewards, dones, _ = env.step(actions)
        ep_goals    += sum(dones)
        total_goals += sum(dones)
        steps += 1

        for i, agent in enumerate(env.agents):
            trails[i].append(tuple(w2s(agent.pos)))

        for i in range(len(env.agents)):
            for j in range(i+1, len(env.agents)):
                if np.linalg.norm(env.agents[i].pos - env.agents[j].pos) < 0.12:
                    flash_a[i] = 6; flash_a[j] = 6
                    collisions += 1
            for k, o in enumerate(env.obstacles):
                if np.linalg.norm(env.agents[i].pos - o.pos) < 0.13:
                    flash_a[i] = 6; flash_o[k] = 6
                    collisions += 1

        if steps >= config.MAX_STEPS or all(dones):
            episode += 1
            obs_all, steps, ep_goals, trails = reset_ep()

    for i in range(len(flash_a)):
        if flash_a[i] > 0: flash_a[i] -= 1
    for i in range(len(flash_o)):
        if flash_o[i] > 0: flash_o[i] -= 1

    screen.fill(BG_COLOR)
    pygame.draw.rect(screen, (12,16,24), (0,0,ARENA,ARENA))
    for x in range(0, ARENA, 50):
        pygame.draw.line(screen, GRID_COLOR, (x,0),(x,ARENA))
    for y in range(0, ARENA, 50):
        pygame.draw.line(screen, GRID_COLOR, (0,y),(ARENA,y))
    pygame.draw.rect(screen, (60,70,90), (0,0,ARENA,ARENA), 2)

    overlay = pygame.Surface((ARENA, ARENA), pygame.SRCALPHA)

    if show_trails:
        for i, trail in enumerate(trails):
            pts = list(trail)
            if len(pts) > 1:
                for k in range(len(pts)-1):
                    a = int(40 + 200 * k / max(1, len(pts)-1))
                    c = AGENT_COLORS[i]
                    pygame.draw.line(overlay, (c[0],c[1],c[2],a), pts[k], pts[k+1], 2)

    if show_safety:
        for k, o in enumerate(env.obstacles):
            px = w2s(o.pos)
            pygame.draw.circle(overlay, (239,68,68,40), tuple(px), int(0.15*SCALE))
        for i, a in enumerate(env.agents):
            px = w2s(a.pos)
            c = AGENT_COLORS[i]
            pygame.draw.circle(overlay, (c[0],c[1],c[2],25), tuple(px), int(0.075*SCALE))

    for i, agent in enumerate(env.agents):
        px = w2s(agent.pos)
        gpx = w2s(env.goals[i])
        c = GOAL_COLORS[i]
        pygame.draw.line(overlay, (c[0],c[1],c[2],70), tuple(px), tuple(gpx), 1)

    screen.blit(overlay, (0,0))

    for i, goal in enumerate(env.goals):
        px = w2s(goal)
        pulse = int(14 + 3 * np.sin(pygame.time.get_ticks()*0.005 + i))
        pygame.draw.circle(screen, GOAL_COLORS[i], px, pulse, 2)
        pygame.draw.circle(screen, GOAL_COLORS[i], px, 3)
        pygame.draw.line(screen, GOAL_COLORS[i], (px[0]-10,px[1]),(px[0]+10,px[1]),1)
        pygame.draw.line(screen, GOAL_COLORS[i], (px[0],px[1]-10),(px[0],px[1]+10),1)
        lbl = font_xs.render('G'+str(i+1), True, GOAL_COLORS[i])
        screen.blit(lbl, (px[0]+12, px[1]-6))

    for k, o in enumerate(env.obstacles):
        px = w2s(o.pos)
        c = FLASH if flash_o[k] > 0 else OBS_COLOR
        pygame.draw.circle(screen, c, px, 18)
        pygame.draw.circle(screen, (40,0,0), px, 18, 2)
        if show_arrows:
            end = (px + o.vel * SCALE * 5).astype(int)
            pygame.draw.line(screen, (255,200,200), tuple(px), tuple(end), 2)

    for i, agent in enumerate(env.agents):
        px  = w2s(agent.pos)
        c = FLASH if flash_a[i] > 0 else AGENT_COLORS[i]
        pygame.draw.circle(screen, c, px, 12)
        pygame.draw.circle(screen, WHITE, px, 12, 2)
        if show_arrows:
            end = (px + agent.vel * SCALE * 5).astype(int)
            pygame.draw.line(screen, WHITE, tuple(px), tuple(end), 2)
        lbl = font_sm.render(str(i+1), True, WHITE)
        screen.blit(lbl, (px[0]-4, px[1]-9))

    pygame.draw.rect(screen, PANEL_COLOR, (ARENA,0,PANEL,HEIGHT))
    pygame.draw.line(screen, (50,60,80), (ARENA,0),(ARENA,HEIGHT),1)

    p0 = ARENA + 16
    py = 18
    title = font_big.render('Project 7', True, WHITE)
    screen.blit(title, (p0, py)); py += 26
    sub = font_sm.render('MARL Swarm Navigation', True, DIM_TEXT)
    screen.blit(sub, (p0, py)); py += 22
    pygame.draw.line(screen, (50,60,80), (p0, py),(WIDTH-16, py)); py += 12

    rows = [
        ('Episode',         str(episode),                              TEXT_COLOR),
        ('Step',            str(steps)+' / '+str(config.MAX_STEPS),    TEXT_COLOR),
        ('Goals this ep',   str(ep_goals),                             (0,212,170)),
        ('Goals total',     str(total_goals),                          (0,212,170)),
        ('Collisions',      str(collisions),                           (239,68,68)),
        ('Speed (FPS)',     str(fps_options[fps_idx]),                 TEXT_COLOR),
        ('Status',          'PAUSED' if paused else 'RUNNING',         (245,158,11) if paused else (0,212,170)),
    ]
    for label, value, color in rows:
        l = font_sm.render(label, True, DIM_TEXT)
        v = font_med.render(value, True, color)
        screen.blit(l, (p0, py))
        screen.blit(v, (WIDTH - 16 - v.get_width(), py-2))
        py += 22

    py += 6
    pygame.draw.line(screen, (50,60,80), (p0, py),(WIDTH-16, py)); py += 10

    h = font_med.render('Agents', True, WHITE)
    screen.blit(h, (p0, py)); py += 20

    for i, agent in enumerate(env.agents):
        d = float(np.linalg.norm(agent.pos - env.goals[i]))
        pygame.draw.circle(screen, AGENT_COLORS[i], (p0+8, py+8), 6)
        lbl = font_sm.render('Agent '+str(i+1), True, TEXT_COLOR)
        screen.blit(lbl, (p0+22, py))
        dlbl = font_sm.render('d='+('%.2f'%d), True, DIM_TEXT)
        screen.blit(dlbl, (WIDTH-16-dlbl.get_width(), py))
        py += 16
        bar_w = PANEL - 32
        pygame.draw.rect(screen, (35,42,55), (p0, py, bar_w, 4))
        progress = max(0.0, min(1.0, 1.0 - d/1.5))
        pygame.draw.rect(screen, AGENT_COLORS[i], (p0, py, int(bar_w*progress), 4))
        py += 12

    py += 6
    pygame.draw.line(screen, (50,60,80), (p0, py),(WIDTH-16, py)); py += 10

    h = font_med.render('Controls', True, WHITE)
    screen.blit(h, (p0, py)); py += 20

    keys = [
        ('SPACE', 'pause / play'),
        ('+ / -', 'speed up / slow down'),
        ('R',     'reset episode'),
        ('T',     'toggle trails'),
        ('S',     'toggle safety zones'),
        ('V',     'toggle velocity arrows'),
        ('Q',     'quit'),
    ]
    for k, desc in keys:
        kk = font_sm.render(k, True, (245,158,11))
        screen.blit(kk, (p0, py))
        dd = font_sm.render(desc, True, DIM_TEXT)
        screen.blit(dd, (p0+62, py))
        py += 16

    pygame.display.flip()
    clock.tick(fps_options[fps_idx])
