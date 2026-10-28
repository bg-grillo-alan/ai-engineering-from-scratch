import sys
import time
import tracemalloc
import logging
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

EPOCHS = 10
STEPS_PER_EPOCH = 100
BATCH_SIZE = 4

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def training_step(model, x, target):
    optimizer.zero_grad()

    output = model(x)
    loss = criterion(output, target)

    loss.backward()
    optimizer.step()

    return loss.item()


model.train()

writer = SummaryWriter("runs/debugging_demo")

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for step in range(STEPS_PER_EPOCH):
        # Génération de données synthétiques
        x = torch.randn(BATCH_SIZE, 784)
        target = torch.randint(0, 10, (BATCH_SIZE,))

        # Entraînement
        loss = training_step(model, x, target)
        writer.add_scalar("loss/train",loss, epoch * STEPS_PER_EPOCH + step)
        if loss > 0.5 or torch.isnan(torch.tensor(loss)): 
            breakpoint()  # Déclenche un point d'arrêt si la perte est anormalement élevée ou NaN
        epoch_loss += loss
        if step % 10 == 0:
            for name, param in model.named_parameters():
                writer.add_histogram(f"weights/{name}", param, epoch * STEPS_PER_EPOCH + step)
                if param.grad is not None:
                    writer.add_histogram(f"gradients/{name}", param.grad, epoch * STEPS_PER_EPOCH + step)

    avg_loss = epoch_loss / STEPS_PER_EPOCH
    writer.add_scalar("loss/epoch", avg_loss, epoch)

writer.close()
