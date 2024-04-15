import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import build_model  # This imports the build_model function from your model.py

def evaluate(model_path: str,
             test_dir: str,
             img_height: int = 224,
             img_width: int = 224,
             batch_size: int = 32) -> None:
    """
    Evaluates the trained model on the test dataset using PyTorch.

    Args:
        model_path (str): Path to the saved model.
        test_dir (str): Path to the test directory.
        img_height (int): Height of the input images. Default is 224.
        img_width (int): Width of the input images. Default is 224.
        batch_size (int): Size of the batches of data. Default is 32.
    """
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformations for the test data
    test_transforms = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the test data
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the saved model
    model = build_model(num_classes=525)  # Ensure num_classes matches your specific model configuration
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # Evaluate the model on the test set
    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()  # Change if different loss was used

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_accuracy = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

if __name__ == '__main__':
    evaluate(model_path='models/bird_classification_model.pth', test_dir=os.path.join('data', 'test'))
