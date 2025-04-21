import torch
import time
import copy

def train_model(model, criterion, optimizer, scheduler,train_loader, val_loader, device, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start_time = time.time()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-'*50)

        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

        training_loss = running_loss / len(train_loader.dataset)
        train_acc = correct_predictions.double() / total_predictions

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {training_loss:.4f} | Train Acc: {train_acc:.2%}")

        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_predictions.double() / total_predictions
        scheduler.step()
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2%}")
        print()

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        history['train_loss'].append(training_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.2%}')

    model.load_state_dict(best_model_wts)
    return model, history
