import json

def evaluate(model, testloader):
    model.eval()
    val_acc, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:

            outputs, _ = model(images.to(device))
            val_acc += (outputs.argmax(dim=1) == labels.to(device)).sum().item()
            total += labels.shape[0]
            
    val_acc = val_acc / total
    return val_acc


def save_stats(key, stats):
    
    for stats_path, value in stats.items():
        try:
            with open(stats_path, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = {}

        data[key] = value

        with open(stats_path, 'w') as file:
            json.dump(data, file)