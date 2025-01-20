import argparse
import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models

def arg_parse():
    parser = argparse.ArgumentParser(description='Image Classifier Training')

    parser.add_argument('data_dir', help='Path to the data directory')
    parser.add_argument('--save_dir', help='Set directory to save checkpoints', default="checkpoint.pth")
    parser.add_argument('--learning_rate', help='Set the learning rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', help='Set the number of hidden units', type=int, default=4096)
    parser.add_argument('--output_features', help='Specify the number of output features', type=int, default=102)
    parser.add_argument('--epochs', help='Set the number of epochs', type=int, default=1)
    parser.add_argument('--gpu', help='Use GPU for training', action='store_true')
    parser.add_argument('--arch', help='Choose architecture', default='vgg11')

    return parser.parse_args()

def train_transform(train_dir):
    transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_set = datasets.ImageFolder(train_dir, transform=transform)
    return train_set

def valid_transform(valid_dir):
    transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_set = datasets.ImageFolder(valid_dir, transform=transform)
    return valid_set

def train_loader(data, batch_size=64, shuffle=True):
    return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)

def valid_loader(data, batch_size=64):
    return torch.utils.data.DataLoader(data, batch_size=batch_size)

def check_device(use_gpu):
    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    return device

def load_model(arch):
    if hasattr(models, arch):
        model = getattr(models, arch)(pretrained=True)
    else:
        raise ValueError(f"Model architecture '{arch}' is not supported.")
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def initialize_classifier(model, hidden_units, output_features):
    if hasattr(model, 'classifier'):
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, output_features),
            nn.LogSoftmax(dim=1)
        )
    elif hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, output_features),
            nn.LogSoftmax(dim=1)
        )
    else:
        raise AttributeError("Model does not have 'classifier' or 'fc' attribute.")
    return model

def train_model(model, trainloader, validloader, device, optimizer, criterion, epochs=1, print_every=10):
    steps = 0
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0
        model.train()

        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                test_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        output = model.forward(images)
                        loss = criterion(output, labels)
                        test_loss += loss.item()

                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

def save_checkpoint(model, optimizer, class_to_idx, path, arch, hidden_units, output_features):
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'arch': arch,
        'hidden_units': hidden_units,
        'output_features': output_features,
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)

def main():
    args = arg_parse()

    train_dir = args.data_dir + '/train'
    valid_dir = args.data_dir + '/valid'

    train_set = train_transform(train_dir)
    valid_set = valid_transform(valid_dir)

    trainloader = train_loader(train_set)
    validloader = valid_loader(valid_set)

    device = check_device(args.gpu)

    model = load_model(args.arch)
    model = initialize_classifier(model, args.hidden_units, args.output_features)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() if hasattr(model, 'classifier') else model.fc.parameters(), lr=args.learning_rate)

    model.to(device)

    print_every = 10
    steps = 0

    train_model(model, trainloader, validloader, device, optimizer, criterion, args.epochs, print_every)
    save_checkpoint(model, optimizer, train_set.class_to_idx, args.save_dir, args.arch, args.hidden_units, args.output_features)

if __name__ == '__main__':
    main()