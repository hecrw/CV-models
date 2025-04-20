import torch
import time
import copy

def train_model(model, criterion, optimizer, train_loader, val_loader, device,num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-'*10)

        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_predictions += labels.size(0)

            training_loss = running_loss/len(train_loader.dataset)
            train_acc = correct_predictions.double() / total_predictions
            print(f'Training Loss: {training_loss:.4f}, Training Accuracy: {train_acc:.4f}')

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
                    correct_predictions += torch.sum(preds == labels.data)
                    total_preds += labels.size(0)

            val_loss = val_loss / len(val_loader.datasets)
            val_acc = correct_predictions.double() / total_preds
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

            if val_acc > best_acc:
                best_acc = val_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            print()

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best validation accuracy: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model