import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from collections import OrderedDict
from hooks import cnn_manual_model_summary
from multiprocessing import Queue
from renderer import start_render


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)

model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if __name__ == "__main__":
    # visualizer = {'visual': 'linear',
    #               'camera_distance': 144}
    visualizer = {'visual': 'ring',
                  'camera_distance': 84}

    render_queue = Queue()
    p = start_render(render_queue, visualizer)
    summary = OrderedDict()

    inputs = next(iter(test_loader))[0].to(device)
    cnn_manual_model_summary(model, inputs, summary, render_queue)

    with torch.no_grad():
        for idx, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device)
            print('Running Model')
            output = model(image)
            pred = output.argmax(dim=1)
            render_queue.put(summary.copy())
            summary.clear()

