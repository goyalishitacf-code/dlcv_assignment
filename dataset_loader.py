import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import zipfile

# This dictionary handles the translation
ITALIAN_TO_ENGLISH = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}

DATASET_CLASSES = {
    "cifar10":       ["airplane","automobile","bird","cat","deer",
                      "dog","frog","horse","ship","truck"],
    "mnist":         [str(i) for i in range(10)],
    "animals10": ["butterfly","cat","chicken","cow","dog",
              "elephant","horse","sheep","spider","squirrel"],
}

def handle_manual_zip(root_folder):
    extract_to = root_folder 
    zip_path = root_folder + ".zip" 
    if not os.path.exists(extract_to):
        if os.path.exists(zip_path):
            print(f"[System] Found {zip_path}. Extracting now...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(extract_to))
            print(f"[System] Extraction complete.")
        else:
            print(f"!! Error: Please place 'animals10.zip' in the data folder !!")
            return False
    return True

def _to_numpy(dataset, max_samples=5000):
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)
    imgs, lbls = [], []
    for x, y in loader:
        imgs.append(x.numpy())
        lbls.append(y.numpy())
        if sum(len(l) for l in lbls) >= max_samples:
            break
    imgs = np.concatenate(imgs)[:max_samples]
    lbls = np.concatenate(lbls)[:max_samples]
    imgs_hwc = (imgs.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
    return imgs_hwc, lbls

def load_cifar10(root="./data", max_samples=5000):
    tf = transforms.ToTensor()
    ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tf)
    imgs, labels = _to_numpy(ds, max_samples)
    return imgs, labels, DATASET_CLASSES["cifar10"]

def load_mnist(root="./data", max_samples=5000):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])
    ds = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=tf)
    imgs, labels = _to_numpy(ds, max_samples)
    return imgs, labels, DATASET_CLASSES["mnist"]

def load_animals10(root="./data/animals10", max_samples=5000):
    if not handle_manual_zip(root):
        return np.array([]), np.array([]), []

    actual_data_path = root
    for current_root, dirs, files in os.walk(root):
        if len(dirs) >= 10: 
            actual_data_path = current_root
            break

    tf = transforms.Compose([
        transforms.Resize((32, 32)), 
        transforms.ToTensor()
    ])
    
    try:
        ds = torchvision.datasets.ImageFolder(root=actual_data_path, transform=tf)
        imgs, labels = _to_numpy(ds, max_samples)
        
        # --- TRANSLATION LOGIC START ---
        # Convert Italian folder names to English names using our dictionary
        english_classes = [ITALIAN_TO_ENGLISH.get(name.lower(), name) for name in ds.classes]
        # --- TRANSLATION LOGIC END ---
        
        print(f"[Animals-10]      {imgs.shape} ready with English labels.")
        return imgs, labels, english_classes
    except Exception as e:
        print(f"!! Error loading ImageFolder: {e} !!")
        return np.array([]), np.array([]), []

def load_all(root="./data", max_samples=5000):
    return {
        "cifar10":   load_cifar10(root, max_samples),
        "mnist":     load_mnist(root, max_samples),
        "animals10": load_animals10(os.path.join(root, "animals10"), max_samples),
    }