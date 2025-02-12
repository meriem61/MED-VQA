from torch import nn
import warnings
import torch
warnings.filterwarnings("ignore")
from model import VQAClassifier
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torchvision.models as models
from MED_VQA_Data import MED_VQA_Data
from tqdm import tqdm


# Hyperparameters
input_size = 768
num_classes = 1742  
learning_rate = 0.0001
num_epochs = 30
batch_size = 32
dropout_prob = 0.2
num_heads = 8 


# Load dataset
train_df = pd.read_csv("../datasets/clef2019/train/traindf_labeled.csv")
val_df = pd.read_csv("../datasets/clef2019/valid/valdf_labeled.csv")
test_df = pd.read_csv("../datasets/clef2019/test/testdf_labeled.csv")

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Create DataLoader
train_dataset = MED_VQA_Data(train_df)
val_dataset = MED_VQA_Data(val_df)
test_dataset = MED_VQA_Data(test_df)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#define model
model = VQAClassifier(embed_dim=input_size, num_classes=num_classes, dropout=dropout_prob, num_heads=num_heads)

# to device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def calculate_metrics(loader, model):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    predicted_labels = []
    true_labels = []
    type_correct = {'CLOSED': 0, 'OPEN': 0}
    type_total = {'CLOSED': 0, 'OPEN': 0}
    with torch.no_grad():
        for data in loader:
            image_emb = data['img_emb'].to(device)
            text_emb = data['text_emb'].to(device)
            labels = data['label'].to(device)
            type = data['type']
            outputs = model(image_emb, text_emb)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            for i in range(len(labels)):
                if type[i] == 'CLOSED':
                    type_correct['CLOSED'] += (predicted[i] == labels[i]).item()
                    type_total['CLOSED'] += 1
                elif type[i] == 'OPEN':
                    type_correct['OPEN'] += (predicted[i] == labels[i]).item()
                    type_total['OPEN'] += 1
    accuracy = correct / total
    avg_loss = total_loss / total
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    type_accuracy = {t: type_correct[t] / type_total[t] for t in type_correct}
    return accuracy, avg_loss,precision,recall,f1,type_accuracy

results_df = pd.DataFrame(columns=["Epoch", "Training loss","Training accuracy", "Validation loss", "Validation accuracy", "Test accuracy", "Test loss","Precision","Recall","F1","Open Accuracy","Closed Accuracy"])

for epoch in range(num_epochs):
    model.train()
    train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
    for batch_idx, data in enumerate(train_loader_tqdm):
        image_emb = data['img_emb'].to(device)
        text_emb = data['text_emb'].to(device)
        labels = data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(image_emb, text_emb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loader_tqdm.set_postfix(loss=loss.item())

    train_acc, train_loss,_,_,_,_= calculate_metrics(train_loader, model)
    val_acc, val_loss,_,_,_,_ = calculate_metrics(val_loader, model)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    test_acc, test_loss,pre_score,rec_score,f1,type_acc = calculate_metrics(test_loader, model)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Precision: {pre_score:.4f}, Recall: {rec_score:.4f}, F1: {f1:.4f},open_acc: {type_acc["OPEN"]:.4f}, closed_acc: {type_acc["CLOSED"]:.4f}')
    results_df = pd.concat([results_df, pd.DataFrame([[epoch, train_loss,train_acc, val_loss, val_acc, test_acc, test_loss,pre_score,rec_score,f1,type_acc['OPEN'],type_acc['CLOSED']]], columns=["Epoch", "Training loss","Training accuracy", "Validation loss", "Validation accuracy", "Test accuracy", "Test loss","Precision","Recall","F1","Open Accuracy","Closed Accuracy"])])
    results_df.to_csv(f"Results/results_VIT_imgClef_{num_heads}h_{dropout_prob}d.csv", index=False)
    check_point = epoch+1
    if check_point % 5 == 0:
        torch.save(model.state_dict(), f'combined_model_VIT_imgClef_{check_point}.pth')
        print(f"Model saved at epoch {check_point}")

print("Training complete.")
torch.save(model.state_dict(), 'combined_model_VIT_imgClef.pth')
print("Model saved successfully")