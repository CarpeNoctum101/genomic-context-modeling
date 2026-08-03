import torch
from sklearn.metrics import roc_auc_score

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        sequences = batch["sequences"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(sequences)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * sequences.size(0)

    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            
            sequences = batch["sequences"].to(device)
            labels = batch["labels"].to(device)

            logits = model(sequences)

            loss = criterion(logits, labels)
            total_loss += loss.detach().item() * sequences.size(0)

            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).long()

            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total
    accuracy = correct / total

    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, accuracy, auc

def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs
):
    model.to(device)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_auc": []
    }

    best_val_auc = 0
    best_state = None

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history, best_val_auc