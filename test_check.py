import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define Augmentations for Training Data
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to a fixed size
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip images
    transforms.RandomRotation(15),  # Random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

# Define Minimal Transforms for Testing Data (No Augmentation)
test_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to the same size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the entire dataset
dataset = datasets.ImageFolder(root="path_to_your_dataset")

# Define split sizes (e.g., 80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

# Split dataset into training and testing sets
train_data, test_data = random_split(dataset, [train_size, test_size])

# Apply transformations (manually override the `dataset.transform`)
train_data.dataset.transform = train_transform  # Augment train set
test_data.dataset.transform = test_transform  # No augmentation for test set

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)




### Applying Data augmentation