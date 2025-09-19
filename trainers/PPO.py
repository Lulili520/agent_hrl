import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from argument import *


class Actor(nn.Module):
    def __init__(self, network_size):
        super(Actor, self).__init__()

        inp, hid1, hid2, out = network_size

        self.fc1 = nn.Linear(inp, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, out)
        self.distribution = torch.distributions.Categorical


    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        logits = self.fc3(x)
        return logits


# Critic网洛
class Critic(nn.Module):
    def __init__(self, network_size):
        super(Critic, self).__init__()

        inp, hid1, hid2, out = network_size

        self.fc1 = nn.Linear(inp, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, out)
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


class Ppo():
    def __init__(self, observation_inp, actor_inp, hidden_size, device="cpu"):

        actor_size = (observation_inp, *hidden_size, actor_inp)
        critic_size = (observation_inp, *hidden_size, 1)

        self.actor_net = Actor(actor_size).to(device)
        self.critic_net = Critic(critic_size).to(device)
        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=lr_critic, weight_decay=l2_rate)
        self.critic_loss_func = torch.nn.MSELoss()
        self.device = device

    def take_action(self, x):
        logits = self.actor_net(x)
        probs = torch.softmax(logits, dim=-1)
        Pi = self.actor_net.distribution(probs)
        return Pi.sample().cpu().numpy()

    def take_action_eval(self, x):
        logits = self.actor_net(x)
        probs = torch.softmax(logits, dim=-1)
        act = torch.argmax(probs, dim=-1)
        return act.detach().cpu().numpy()

    def set_device(self, device):
        self.actor_net.to(device)
        self.critic_net.to(device)
    
    def train(self, memory):
        states = torch.stack([torch.tensor(item[0], dtype=torch.float32) for item in memory]).to(self.device)
        actions = torch.tensor([int(item[1]) for item in memory], dtype=torch.long).to(self.device)
        rewards = torch.tensor([item[2] for item in memory], dtype=torch.float32).to(self.device)
        masks = torch.tensor([1.0 - item[3] for item in memory], dtype=torch.float32).to(self.device)


        values = self.critic_net(states)
        returns, advants = self.get_gae(rewards, masks, values)
        old_logits = self.actor_net(states)
        old_logits = torch.softmax(old_logits, dim=-1)

        pi = self.actor_net.distribution(old_logits)
        old_log_prob = pi.log_prob(actions).unsqueeze(1)

        n = len(states)
        arr = np.arange(n)


        for epoch in range(epoch_num):
            np.random.shuffle(arr)
            for i in tqdm(range(n // batch_size), desc=f"Epoch {epoch+1}/{epoch_num}"):
                b_index = arr[batch_size * i:batch_size * (i + 1)]
                b_states = states[b_index]
                b_advants = advants[b_index].unsqueeze(1)
                b_actions = actions[b_index]
                b_returns = returns[b_index].unsqueeze(1)

                logits = self.actor_net(b_states)
                logits = torch.softmax(logits, dim=-1)
                pi = self.actor_net.distribution(logits)
                new_prob = pi.log_prob(b_actions).unsqueeze(1)
                old_prob = old_log_prob[b_index].detach()

                ratio = torch.exp(new_prob - old_prob)
                surrogate_loss = ratio * b_advants
                values = self.critic_net(b_states)
                critic_loss = self.critic_loss_func(values, b_returns)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                ratio = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
                clipped_loss = ratio * b_advants
                actor_loss = -torch.min(surrogate_loss, clipped_loss).mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                wandb.log({"Actor Loss": actor_loss.item(), "Critic Loss": critic_loss.item()})

                tqdm.write(f"Epoch {epoch+1}, Batch {i+1}/{n//batch_size}, "
                       f"ActorLoss={actor_loss.item():.4f}, CriticLoss={critic_loss.item():.4f}")
        
    # 计算KL散度
    def kl_divergence(self, old_mu, old_sigma, mu, sigma):

        old_mu = old_mu.detach()
        old_sigma = old_sigma.detach()

        kl = torch.log(old_sigma) - torch.log(sigma) + (old_sigma.pow(2) + (old_mu - mu).pow(2)) / (
                    2.0 * sigma.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    # 计算GAE
    def get_gae(self, rewards, masks, values):
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)
        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + gamma * running_returns * masks[t]
            running_tderror = rewards[t] + gamma * previous_value * masks[t] - values.data[t]
            running_advants = running_tderror + gamma * lamb * running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values.data[t]
            advants[t] = running_advants
        advants = (advants - advants.mean()) / advants.std()
        return returns, advants
