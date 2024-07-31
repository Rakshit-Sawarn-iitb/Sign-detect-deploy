import torch
from PIL import Image

def predict_signature(model, original_image_path, test_image_path, transform, device, threshold=0.001):
    model.eval()
    original_image = Image.open(original_image_path).convert("L")
    test_image = Image.open(test_image_path).convert("L")

    if transform is not None:
        original_image = transform(original_image)
        test_image = transform(test_image)

    original_image = original_image.unsqueeze(0).to(device)
    test_image = test_image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        features1 = model.forward_features(original_image)
        features2 = model.forward_features(test_image)
    
    euclidean_distance = torch.nn.functional.pairwise_distance(features1, features2).mean().item()
    print(euclidean_distance)

    if euclidean_distance < threshold:
        response = "Original"
    else:
        response = "Forged"
    
    return response