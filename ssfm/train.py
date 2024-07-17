import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device=torch.device('cuda')):
    train_losses = []
    val_losses = []
    val_accuracies = []
    for epoch in range(num_epochs):
        model.train()
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Ensure outputs are of type torch.FloatTensor and labels are torch.LongTensor
            outputs = outputs.float()
            labels = labels.long()

            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        train_losses.append(loss.item())
        # Evaluation on validation data
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                outputs = outputs.float()
                labels = labels.long()
                
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%')
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
    return model, train_losses, val_losses, val_accuracies