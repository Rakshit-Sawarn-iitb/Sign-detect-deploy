import torch
from PIL import Image

def predict_signature(model, signatures, test_image_path, transform, device, threshold=0.55):
    counter = 0
    model.eval()
    for i in range(len(signatures)):
        original_image = signatures[i]
        original_image = Image.fromarray(original_image).convert("L")
        test_image = Image.open(test_image_path).convert("L")

        if transform is not None:
            original_image = transform(original_image)
            test_image = transform(test_image)

        original_image = original_image.unsqueeze(0).to(device)
        test_image = test_image.unsqueeze(0).to(device)

        with torch.no_grad():
            features1 = model.forward_features(original_image)
            features2 = model.forward_features(test_image)
        
        euclidean_distance = torch.nn.functional.pairwise_distance(features1, features2).mean().item()
        print(euclidean_distance)

        if euclidean_distance < threshold:
            counter = counter+1
    if (counter>4):
        response = "Original"
    else:
        response = "Forged"
    
    return response