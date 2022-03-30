import gym
import time
import argparse
import matplotlib.pyplot as plt
from drawnow import drawnow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pygame
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
last_score_plot = []
avg_score_plot = []

def draw_fig():
  plt.ylabel('reward')
  plt.xlabel('episode')
  plt.plot(last_score_plot, '-')
  plt.plot(avg_score_plot, 'r-')

size = width, height = 500, 500
nummembers = 3
radius = torch.tensor(width/4).to(device)
angle = torch.tensor(2*np.pi/nummembers).to(device)
center = torch.tensor([width/2,height/2]).to(device)
nodesize = width/40
ballsize = width/30
shootspeed = 20
sleep_time=0

# ENVIRONMENT
pygame.init()
black = 0, 0, 0
pygame.display.set_caption("Backclap Simulation")
screen = pygame.display.set_mode(size)
balls = torch.zeros(nummembers,2).to(device)
nodes = torch.zeros(nummembers,2).to(device)
speeds = torch.zeros(nummembers,2).to(device)
inair = torch.zeros(nummembers).to(device) # Stores 0 for stationary or nummembers+1 for target
for i in range(nummembers):   
    ang = torch.mul(angle,i)
    nodes[i,0]= torch.add(center[0],torch.mul(radius,torch.cos(ang)))
    nodes[i,1]= torch.add(center[1],torch.mul(radius,torch.sin(ang)))
def resetenv():
    for i in range(nummembers):      
        balls[i,0]=nodes[i,0]
        balls[i,1]=nodes[i,1]
        speeds[i,0] = 0
        speeds[i,1] = 0
        inair[i] = 0
        
def shoot(targetnodenum,ballnum):
    if inair[ballnum]==0 and targetnodenum!=ballnum:
        #print("Shooting ",targetnodenum," at ",ballnum)
        #targetnodenum= targetnodenum.item()
        #ballnum=ballnum.item()
        diff = torch.sub(nodes[targetnodenum,:],balls[ballnum,:]).to(device)
        speeds[ballnum,:]=speeds[ballnum,:]+torch.div(diff,torch.norm(diff))*shootspeed
        inair[ballnum]=targetnodenum+1

#Actions torch.tensor([origin,target]) oder [0,0] Wait
def step(action):
    done = False
    collisions = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    if action!=torch.tensor([-1]).to(device):
        #print("Action In: ",action)
        #print("Action Out: ",torch.div(action-1,nummembers,rounding_mode="floor"),action%nummembers)
        target = torch.div(action,nummembers,rounding_mode="floor")
        if target>2: target =torch.tensor([2]).to(device)
        shoot(target,action%nummembers)
        
    reward = torch.tensor([-0.01]).to(device)
    screen.fill(black)
    for i in range(nummembers):
        pygame.draw.circle(screen, (0,255,0), (balls[i][0].item(),balls[i][1].item()), ballsize) # Draw Balls
        balls[i,:]=torch.add(balls[i,:],speeds[i,:]) # Update Positions
        for o in range(nummembers): # Ball to Ball collision
            if inair[i]-1==o:
                if torch.norm(torch.subtract(balls[i],nodes[o]))<1*(ballsize+nodesize):
                    #balls[i] = torch.clone(nodes[o]) #BUGGY
                    #balls[i]=nodes[o]
                    speeds[i]=torch.tensor([0,0]).to(device)
                    reward = torch.add(reward,0.3)
                    inair[i]=0
 #                   print("Ball Caught ",i)
            if i!=o:
                if torch.norm(torch.subtract(balls[i],balls[o]))<(2*ballsize):      
                    reward = torch.add(reward,-0.5)
 #                   print("Ball Collision with",i," and",o)
                    done = True

        pygame.draw.circle(screen, (255,255,255), (nodes[i][0].item(),nodes[i][1].item()), nodesize) # Draw Nodes
        time.sleep(sleep_time)
    pygame.display.flip()
    next_state= torch.flatten(balls).nan_to_num(nan=0.0).to(device)
    return next_state, reward, False, {}


# LEARNING
gamma=0.6
lr=1e-3
max_episode=1000
epsilon = 0.01 # Exploration Rate

obs_size = torch.numel(balls.flatten())
n_actions = nummembers*nummembers+1
class Policy(nn.Module):
  def __init__(self):
    super(Policy, self).__init__()
    self.fc1 = nn.Linear(obs_size, 128)
    self.fc2 = nn.Linear(128, n_actions)

  def forward(self, x):
    x = F.elu(self.fc1(x))
    action_probs = F.softmax(self.fc2(x), dim=1)
    return action_probs


env = gym.make('CartPole-v0')
policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=lr)

def get_action(state):
  sample = random.random() # 0 - 1
  action_probs = policy(state)
  action_dist = torch.distributions.Categorical(action_probs)
  action = action_dist.sample()
  if sample>epsilon: return random.randrange(n_actions)
  return action.item()


def update_policy(states, actions, returns):
  action_probs = policy(states)
  action_dist = torch.distributions.Categorical(action_probs)
  action_loss = -action_dist.log_prob(actions.to(device)) * returns.to(device)
  entropy = action_dist.entropy()
  loss = torch.mean(action_loss - 1e-4 * entropy)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  return


def main():
  resetenv()

  episode = 0
  episode_score = 0
  episode_steps = 0

  states = []
  actions = []
  rewards = []
  start_time = time.perf_counter()
  state = torch.flatten(balls).nan_to_num(nan=0.0).to(device)
  while episode < max_episode:
    action = get_action(state.float()[None, :])
    next_state, reward, done, _ = step(action)
    states.append(state)
    actions.append(action)
    rewards.append(reward)
    episode_score += reward
    episode_steps += 1
    if episode_steps>20: done=True
    #print("Steps ",episode_steps)
    if done:
      print("Episode Done")
      # calculate discounted reward
      returns = [rewards[-1]]
      for r in rewards[-2::-1]:
        returns.append(r + gamma * returns[-1])
      state_batch = torch.column_stack(states).transpose(0,1)
      action_batch = torch.tensor(actions).float()
      return_batch = torch.tensor(returns[::-1]).float()
      return_batch = (return_batch - return_batch.mean()) / return_batch.std()
      update_policy(state_batch, action_batch[:, None], return_batch)
      print('episode: %d score %.5f, steps %d, (%.2f sec/eps)' %
            (episode, episode_score, episode_steps, time.perf_counter() - start_time))
      last_score_plot.append(episode_score.cpu())
      if len(avg_score_plot) == 0:
        avg_score_plot.append(episode_score.cpu())
      else:
        avg_score_plot.append(avg_score_plot[-1] * 0.95 + episode_score.cpu() * 0.05)
      drawnow(draw_fig)
      start_time = time.perf_counter()
      episode += 1
      episode_score = 0
      episode_steps = 0
      resetenv()
      state = torch.flatten(balls).nan_to_num(nan=0.0).to(device)
      states.clear()
      actions.clear()
      rewards.clear()
    else:
      state = next_state
  env.close()

if __name__ == '__main__':
  main()
  plt.pause(0)
