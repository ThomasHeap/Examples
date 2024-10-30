import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def set_seeds(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

def init_model():
    model = nn.Sequential(nn.Linear(1, 20), nn.ReLU(), nn.Linear(20, 1))
    return model

def copy_model(source):
    target = init_model()
    target.load_state_dict(source.state_dict())
    return target

class CustomAdam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
                
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

class CustomAdamNoEMA:
    def __init__(self, params, lr=0.001, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.eps = eps
        self.m = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
                
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data
            self.m[i] = g
            m_hat = self.m[i]
            p.data -= self.lr * m_hat / (torch.sqrt(g*g + self.eps))

class SignedGradient:
    def __init__(self, params, lr=0.001):
        self.params = list(params)
        self.lr = lr
        
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
                
    def step(self):
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * torch.sign(p.grad.data)

def create_test_problems():
    X = torch.linspace(-5, 5, 100).reshape(-1, 1)
    
    problems = {
        'Clean': (X, 0.2 * X**2 + torch.randn_like(X) * 0.1),
        'Noisy': (X, 0.2 * X**2 + torch.randn_like(X) * 2.0),
        'Non-convex': (X, 0.2 * X**2 + 2 * torch.sin(X) + torch.randn_like(X) * 0.3)
    }
    return problems

def train_and_plot():
    set_seeds(42)
    problems = create_test_problems()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (name, (X, y)) in enumerate(problems.items()):
        base_model = init_model()
        models = [copy_model(base_model) for _ in range(3)]
        
        optimizers = [
            CustomAdam(models[0].parameters(), lr=0.01),
            CustomAdamNoEMA(models[1].parameters(), lr=0.01),
            SignedGradient(models[2].parameters(), lr=0.01)
        ]
        
        losses = [[] for _ in range(3)]
        criterion = nn.MSELoss()
        
        for epoch in range(400):
            for i, (model, opt) in enumerate(zip(models, optimizers)):
                opt.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                opt.step()
                losses[i].append(loss.item())
        
        ax = axes[idx]
        ax.plot(losses[0], label='Adam+EMA')
        ax.plot(losses[1], label='Adam no-EMA')
        ax.plot(losses[2], label='Signed Gradient')
        ax.set_yscale('log')
        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True)
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.show()
    
train_and_plot()