from model import model, criterion, optimizer
import torch

model_path = "siamesemodel2 (1).pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def fine_tune(model, fine_dataloader, device):
    model.train()
    for epoch in range(5):
        for i, data in enumerate(fine_dataloader,0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            optimizer.zero_grad()
            output1 = model(img0)
            output2 = model(img1)
            loss_contrastive = criterion(output1,output2,label)
            loss_contrastive.backward()
            optimizer.step()
    return model