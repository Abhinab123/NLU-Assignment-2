import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import os

from dataset_prep import IndianNamesDataset
from models import VanillaRNN, BidirectionalLSTM, AttentionRNN

def generate_sample(model, dataset, device, temp=0.6):
    model.eval()
    with torch.no_grad():
        chars = [dataset.char_to_idx[dataset.sos]]
        for _ in range(20):
            input_seq = torch.tensor([chars], dtype=torch.long).to(device)
            logits, _ = model(input_seq)
            
            # Apply temperature scaling for realistic variety
            probs = torch.softmax(logits[0, -1, :] / temp, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            
            if next_idx == dataset.char_to_idx[dataset.eos]: break
            chars.append(next_idx)
        return ''.join([dataset.idx_to_char[i] for i in chars[1:]]).capitalize()

def run_experiment():
    # 1. Environment and Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {"embed": 128, "hidden": 256, "layers": 2, "epochs": 50, "lr": 0.002}
    
    # 2. Data Loading
    dataset = IndianNamesDataset("TrainingNames.txt")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    architectures = {
        "Vanilla_RNN": VanillaRNN(dataset.vocab_size, params['embed'], params['hidden']),
        "BiLSTM": BidirectionalLSTM(dataset.vocab_size, params['embed'], params['hidden']),
        "Attention_RNN": AttentionRNN(dataset.vocab_size, params['embed'], params['hidden'])
    }
    
    final_output = []

    for name, model in architectures.items():
        print(f"--- Training {name} ---")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = nn.CrossEntropyLoss(ignore_index=dataset.char_to_idx[dataset.pad])
        
        for epoch in range(params['epochs']):
            model.train()
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred, _ = model(x)
                loss = criterion(pred.view(-1, dataset.vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()
        
        # Generation and Evaluation
        generated = [generate_sample(model, dataset, device) for _ in range(100)]
        novelty = len([n for n in generated if n.lower() not in dataset.names])
        diversity = len(set(generated))
        
        # Calculate Model Size
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_mb = (param_count * 4) / (1024**2)

        result_entry = (
            f"MODEL: {name}\n"
            f"Parameters: {param_count:,}\n"
            f"Size (MB): {size_mb:.2f} MB\n"
            f"Novelty: {novelty}%\n"
            f"Diversity: {diversity}%\n"
            f"Samples: {', '.join(random.sample(generated, 5))}\n"
        )
        final_output.append(result_entry)
        print(result_entry)

    # 3. Save Results to TXT
    with open("Problem2_Results.txt", "w") as f:
        f.write("PROBLEM 2 EXPERIMENT RESULTS\n" + "="*30 + "\n")
        f.writelines(final_output)
    print(f"✅ Results saved to Problem2_Results.txt")

if __name__ == "__main__":
    run_experiment()