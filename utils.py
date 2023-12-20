import torch
import json
from time import time
from tqdm import tqdm

def evaluate(model, testloader):
    model.eval()
    val_acc, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:

            outputs, _ = model(images.to("cuda") )
            val_acc += (outputs.argmax(dim=1) == labels.to("cuda") ).sum().item()
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
            

def train(path, model, criterion, optimizer1, num_epochs, trainloader, testloader, optimizer2=None):
    torch.manual_seed(0)
    
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer1, max_lr=1e-2, steps_per_epoch=len(trainloader), epochs=num_epochs)

    best_val_acc = 0
    epochs_no_improve, max_patience = 0, 20
    early_stop = False
    step = -1

    pbar = tqdm(range(num_epochs))
    
    start = time()
    for epoch in pbar:

        epoch_acc, epoch_loss, total = 0.0, 0.0, 0
        model.train()
        for inputs, labels in trainloader:
            model.zero_grad()
            step += 1

            outputs, attentions = model(inputs.to("cuda") )
            if optimizer2:
                ce_loss, decatt_loss = criterion(outputs, attentions, labels.to("cuda") )
            else:
                ce_loss = criterion(outputs, labels.to("cuda") )
                decatt_loss = torch.tensor(0)
            
            ce_loss.backward(retain_graph=True)
            optimizer1.step()
            
            if optimizer2:
                decatt_loss.backward()
                optimizer2.step()

            epoch_acc += (outputs.argmax(dim=1) == labels.to("cuda") ).sum().item()
            epoch_loss += ce_loss.item() + decatt_loss.item()
            total += labels.shape[0]
            
#             scheduler.step()
        
        epoch_loss = epoch_loss / len(trainloader)
        epoch_acc = epoch_acc / total
        val_acc = evaluate(model, testloader)
        
        save_stats(epoch, {
            f"stats/{path}_valacc.txt": val_acc,
            f"stats/{path}_trainacc.txt": epoch_acc,
            f"stats/{path}_trainloss.txt": epoch_loss,
            f"stats/{path}_traintime.txt": time() - start
        })
        
        pbar.set_postfix({"Epoch": epoch+1, "Train Accuracy": epoch_acc*100, "Training Loss": epoch_loss, "Validation Accuracy": val_acc*100})

        # Save the best model
        if val_acc > best_val_acc:
            epochs_no_improve = 0
            best_val_acc = val_acc
            tta = time() - start
            
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer1': optimizer1,
                'optimizer2': optimizer2,
            },  f'saved_models/{path}.pth')

        else:
            epochs_no_improve += 1

        if epoch > 100 and epochs_no_improve >= max_patience:
            print('Early stopping!')
            early_stop = True
            break
    
    print(f"Best Validation Accuracy: {best_val_acc*100:.3f}%")
    print(f"Time to Max Val Accuracy: {tta / 60:.3f} mins")