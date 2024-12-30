import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
import tkinter as tkk

labels = None
image_paths =None
var = None
root = None
train_losses = None
valid_losses = None
valid_accuracies = None
image_list = []
label_list = []

class PlantDiseaseDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image and apply transformations
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

def load_images(directory_root):
    global image_list
    global label_list
    image_list, label_list = [], []
    print("[INFO] Loading images...")

    for disease_folder in os.listdir(directory_root):
        disease_folder_path = os.path.join(directory_root, disease_folder)
        if not os.path.isdir(disease_folder_path):
            continue

        for img_name in os.listdir(disease_folder_path):
            if img_name.startswith("."):
                continue
            img_path = os.path.join(disease_folder_path, img_name)
            if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_list.append(img_path)
                label_list.append(disease_folder)

    print("[INFO] Image loading completed")
    return image_list, label_list

# Load images and labels

directory_root = r"C:\Users\M MAGED\Downloads\PlantVillage"
image_paths, labels = load_images(directory_root)

# Encode labels as integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)


# Train, validation, and test splits
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels_encoded, test_size=0.3, random_state=42, stratify=labels_encoded
)
valid_paths, test_paths, valid_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)
def data_info():
    global var
    global image_list
    var.set(f"Total images: {len(image_list)}Training samples: {len(train_paths)} \n Validation samples: {len(valid_paths)} \n Test samples: {len(test_paths)}")


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),           # Data augmentation for training
    transforms.RandomRotation(30),              # Random rotation for variability
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
])

valid_test_transform = transforms.Compose([
    transforms.Resize((256, 256)),              # Consistent resizing for validation/test
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Same normalization as training
])

# Create datasets with appropriate transformations
train_dataset = PlantDiseaseDataset(train_paths, train_labels, transform=train_transform)
valid_dataset = PlantDiseaseDataset(valid_paths, valid_labels, transform=valid_test_transform)
test_dataset = PlantDiseaseDataset(test_paths, test_labels, transform=valid_test_transform)

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle for training
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False) # No shuffle for validation/test
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)   # No shuffle for test

def visualize_samples(dataset, num_samples=5):
    plt.figure(figsize=(22, 5))
    for i in range(num_samples):
        img, label = dataset[i]
        img = img.permute(1, 2, 0).numpy()
        img = (img * 0.5) + 0.5  # De-normalize
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(label_encoder.inverse_transform([label])[0].replace('_',' '))  # Convert label back to class name
        plt.axis('off')
    plt.show()
def visual():
    global test_dataset
    visualize_samples(train_dataset)

class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        # Convolutional Block 1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Output: 32x128x128 (assuming input is 3x256x256)
        )
        # Convolutional Block 2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Output: 64x64x64
        )
        # Convolutional Block 3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Output: 128x32x32
        )
        # Convolutional Block 4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # Output: 256x16x16
        )
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output: 256x1x1
        # Fully Connected Layers
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),  # Adjusted input size after GAP
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.global_avg_pool(x)
        x = self.fc_block(x)
        return x

# Initialize model, loss, optimizer

num_classes = len(label_encoder.classes_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PlantDiseaseModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.002)


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, save_path="best_model.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)  # Save the best model
            print(f"[INFO] Model checkpoint saved to {self.save_path}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("[INFO] Early stopping triggered.")
                return True
        return False


def evaluate_model(model, data_loader, criterion):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= len(data_loader)
    accuracy = correct / total * 100
    return val_loss, accuracy


def train_model(

        model, train_loader, valid_loader, criterion, optimizer, epochs, early_stopping=None):
    train_losses, valid_losses, valid_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", total=len(train_loader))

        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Train Loss": loss.item()})

        # Record training loss
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        val_loss, val_accuracy = evaluate_model(model, valid_loader, criterion)
        valid_losses.append(val_loss)
        valid_accuracies.append(val_accuracy)

        # Print epoch summary
        print(
            f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.2f}%")

        # Early stopping
        if early_stopping and early_stopping(val_loss, model):
            print("[INFO] Early stopping triggered.")
            break

    return train_losses, valid_losses, valid_accuracies

def trainn_model():
    global train_losses
    global valid_losses
    global valid_accuracies
    global model
    global train_loader
    global valid_loader
    global criterion
    global optimizer

    n_epochs = 100
    early_stopping = EarlyStopping(patience=4, min_delta=0.01, save_path="best_model.pth")
    train_losses, valid_losses, valid_accuracies = train_model(
    model, train_loader, valid_loader, criterion, optimizer, epochs=n_epochs, early_stopping=early_stopping
    )

def plot_learning_curve(train_losses, valid_losses, valid_accuracies):
    plt.figure(figsize=(12, 6))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.show()

def plotlearning_curve():
    global train_losses
    global valid_losses
    global valid_accuracies
    plot_learning_curve(train_losses, valid_losses, valid_accuracies)

def print_accuracy():
    global model
    global test_loader
    global criterion
    global var
    final_val_loss, final_val_accuracy = evaluate_model(model, test_loader, criterion)
    var.set(f"Final Evaluation -> Val Loss: {final_val_loss:.4f}, Val Accuracy: {final_val_accuracy:.2f}%")

def exit():
    global root
    root.destroy()

def num_class_and_divice():
    global var
    global num_classes
    global device
    var.set(f"Number of Classes: {num_classes} \n Device Used: {device}")
def addTextLable(root):
    global var
    var =tkk.StringVar()
    var.set("")
    labl = tkk.Label(root,textvariable=var,borderwidth=15)
    labl.pack()

root = tkk.Tk()
a = tkk.Label(root , text="plant_dieases_classification_with_gui" , borderwidth=20)
butt1 = tkk.Button(root, text= "train_the_model",command=trainn_model,borderwidth=15)
butt2 = tkk.Button(root, text= "train_val_test_total",command=data_info,borderwidth=15)
butt3 = tkk.Button(root, text= "visualize_samples",command=visual,borderwidth=15)
butt4 = tkk.Button(root, text= "plot_learning_curve",command=plotlearning_curve,borderwidth=15)
butt5 = tkk.Button(root, text= "print_accuracy",command=print_accuracy,borderwidth=15)
butt6 = tkk.Button(root,text="Exit",command=exit,borderwidth=15)
lab = tkk.Label(root,text="eng:Fahd ashraf \n eng: Rawan islam \n eng: mariam rabea \n eng: zeed ramadan \n eng : kareem hussin \n eng : omar ayman",borderwidth=25)
butt7 = tkk.Button(root,text="num_class_and_divice",command=num_class_and_divice,borderwidth=15)
a.pack()
butt1.pack()
butt2.pack()
butt3.pack()
butt7.pack()
butt4.pack()
butt5.pack()
butt6.pack()
addTextLable(root)
lab.pack()
root.mainloop()