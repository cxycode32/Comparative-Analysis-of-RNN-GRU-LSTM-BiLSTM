import os
import torch


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("Saving checkpoint...")
    torch.save(state, filename)
    print("Checkpoint saved.")


def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    if not os.path.exists(filename):
        print(f"Checkpoint file '{filename}' not found.")
        return

    print("Loading checkpoint...")
    checkpoint = torch.load(filename, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("Checkpoint loaded.")