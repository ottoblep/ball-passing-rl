import pfrl
import torch
import torch.nn
import gym
import numpy 
import pygame
import time
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import pandas as pd
from drawnow import drawnow, figure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
size = width, height = 500, 500
nummembers = 3
radius = torch.tensor(width/4).to(device)
angle = torch.tensor(2*numpy.pi/nummembers).to(device)

center = torch.tensor([width/2,height/2]).to(device)
nodesize = width/40
ballsize = width/30
shootspeed = 20
sleep_time=0

def draw_fig():
    plt.plot(rewardlist,color="green")
    plt.plot(avg_rewlist,color="blue")
    plt.plot(loss2list,color="red")

netsize = 30
episode_length= 12
learning_rate=1e-3
GAMMA = 0.999
    
num_episodes = 2000
BATCH_SIZE = episode_length*20
TARGET_UPDATE = 50
EPS_START = 0.3
EPS_END = 0
memory_size=episode_length*1000

maxsteps=num_episodes*episode_length

# Setup game env
pygame.init()
black = 0, 0, 0
ballcolor = (0,255,0)
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
    
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    def push(self, *args):
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)
    
#Setup NN
class DQN(torch.nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, netsize,bias=True,)
        self.l1b = torch.nn.Linear(netsize, netsize,bias=True,)
        self.l2 = torch.nn.Linear(netsize, n_actions,bias=True,)
    def forward(self, x):
        h = x
        h = torch.nn.functional.elu(self.l1(torch.div(h,width))) 
        h = torch.nn.functional.elu(self.l1b(h)) 
        h = self.l2(h)
        return h

obs_size = torch.numel(balls.flatten())
n_actions = nummembers*nummembers+1
# Environment Functions
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
        targetnodenum= targetnodenum.item()
        ballnum=ballnum.item()
        diff = torch.sub(nodes[targetnodenum,:],balls[ballnum,:]).to(device)
        speeds[ballnum,:]=speeds[ballnum,:]+torch.div(diff,torch.norm(diff))*shootspeed
        inair[ballnum]=targetnodenum+1

#Actions torch.tensor([origin,target]) oder [0,0] Wait
def step(action):
    done=False
    reward = torch.tensor([0.0]).to(device)
    collisions = 0
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()
    if action!=torch.tensor([0]).to(device):
        reward = torch.add(reward,0.01)
        #print("Action In: ",action)
        #print("Action Out: ",torch.div(action-1,nummembers,rounding_mode="floor"),action%nummembers)
        target = torch.div(action,nummembers,rounding_mode="floor")
        if target>2: target =torch.tensor([2]).to(device)
        shoot(target,action%nummembers)
    else: reward = torch.add(reward,-0.01)
         
    screen.fill(black)
    for i in range(nummembers):
        pygame.draw.circle(screen,ballcolor , (balls[i][0].item(),balls[i][1].item()), ballsize) # Draw Balls
        balls[i,:]=torch.add(balls[i,:],speeds[i,:]) # Update Positions
        for o in range(nummembers): # Ball to Ball collision
            if inair[i]-1==o:
                if torch.norm(torch.subtract(balls[i],nodes[o]))<1*(nodesize):
                    #balls[i] = torch.clone(nodes[o]) #BUGGY
                    #balls[i]=nodes[o]
                    speeds[i]=torch.tensor([0,0]).to(device)
                    reward = torch.add(reward,0.3)
                    inair[i]=0    
                    #print("Ball Caught ",i)
            if i!=o:
                if torch.norm(torch.subtract(balls[i],balls[o]))<(2*ballsize):      
                    reward = torch.add(reward,-0.05)
                    done=True
                    #print("Ball Collision with",i," and",o)
            if  balls[i][0]>width or balls[i][1]>width or balls[i][0]<0 or balls[i][1]<0:
                done=True
                #print("Out of Bounds")
        pygame.draw.circle(screen, (255,255,255), (nodes[i][0].item(),nodes[i][1].item()), nodesize) # Draw Nodes
        time.sleep(sleep_time)
    pygame.display.flip()
    return reward,done

#agent.load('agent')
# Main Loop

policy_net = DQN(obs_size, n_actions).to(device)
target_net = DQN(obs_size, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
torch.autograd.set_detect_anomaly(True)

optimizer = optim.RMSprop(policy_net.parameters(),lr=learning_rate) #,centered=True
memory = ReplayMemory(memory_size)

steps_done = 0

def randact():
    return torch.randint(n_actions,(1,)).to(device)
eps_threshold = 1
def select_action(state):
    global steps_done,eps_threshold
    sample = random.random() # 0 - 1
#    eps_threshold = EPS_END + (EPS_START - EPS_END) * \ # Exponential
#        math.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold = (1-steps_done / maxsteps)*EPS_START+EPS_END # Linear
 #   eps_threshold = 1 # Constant
    steps_done += 1
    if steps_done>0.95*maxsteps: return policy_net(state).argmax()
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax()
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

episode_durations = []
loss2list = []
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.column_stack(batch.next_state).transpose(0,1)
    state_batch = torch.column_stack(batch.state).transpose(0,1)
    action_batch = torch.column_stack(batch.action).transpose(0,1)
    reward_batch = torch.column_stack(batch.reward).transpose(0,1)
    
    state_action_values = policy_net.forward(state_batch).gather(1, action_batch)
    
    next_state_values, next_state_indices = target_net(non_final_next_states).max(dim=1)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values* GAMMA) + reward_batch.t()

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values[:,0], expected_state_action_values)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    loss2list.append(loss.detach().cpu())
rewardlist = []
avg_rewlist = [0]
best_rew = -1000
best_policy = []

for i_episode in range(num_episodes):
#    global next_state
    # Initialize the environment and state
    resetenv()
    state = torch.flatten(balls).nan_to_num(nan=0.0).to(device)
    rewcul = 0
    #print("FIRST STATE RESULT: ",policy_net.forward(state))
    episode_policy = []
    for t in range(episode_length):      
        # Select and perform an action
        action = select_action(state)
        episode_policy.append(action)
        reward,done  = step(action)
        rewcul+=reward
        reward = torch.tensor([reward], device=device)
        next_state = torch.flatten(balls).nan_to_num(nan=0.0)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state
        if done==True: break

    print("Episode Reward ",rewcul," Epsilon: ",eps_threshold)
    print(len(memory)/episode_length," of ",num_episodes," Episodes")   
    #print(policy_net.l1.state_dict()['weight'])
    rewardlist.append(rewcul.detach().cpu())
    if rewcul.detach().cpu()>best_rew:
        best_rew=rewcul.detach().cpu()
        best_policy =  episode_policy
    avg_rewlist.append(avg_rewlist[-1] * 0.95 + rewcul.detach().cpu() * 0.05)
    drawnow(draw_fig)
    optimize_model()   
           
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

ballcolor = (255,0,0) # Set to Blue 
print("Complete. Best Policy: ",best_policy)
time.sleep(1)
resetenv()
sleep_time=0.1
for a in best_policy:
    print("Step with Action: ",a)
    step(a)
    

