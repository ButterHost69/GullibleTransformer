import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split,Dataset
from torch.nn.utils.rnn import pad_sequence

from tokenizer import MyBPETokenizer
from model import Encoder
from embedding import CompleteEmbedding

# --------------------------------------------------------------
#                       Data Preprocessing
# --------------------------------------------------------------

print("Processing & Concating Data ...")
# 1. Load the Datasets
df_fake = pd.read_csv('E:\\TimePass\\GullibleTransformer\\data\\kaggle\\news1_Fake.csv')
df_true = pd.read_csv('E:\\TimePass\\GullibleTransformer\\data\\kaggle\\news1_True.csv')

# 2. Assign Labels (True: 0, Fake: 1)
df_true['label'] = 0
df_fake['label'] = 1

df_combined = pd.concat([df_true, df_fake], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)



# --------------------------------------------------------------
#                        Tokenize Tokens
# --------------------------------------------------------------
tokenizer = MyBPETokenizer()
tokenizer.load("E:\\TimePass\\GullibleTransformer\\data\\#synthetic\\tokenizer_v0")

input_ids = []
labels = []

print("Tokenizing Data...")
for index, row in df_combined.iterrows():
    text = row['title']    
    encoded_sent = tokenizer.encode(text)
    input_ids.append(encoded_sent)
    labels.append(row['label'])



# --------------------------------------------------------------
#                      Data Transformation
# --------------------------------------------------------------
batch_size = 16

# Pad the sequences so they are all the same length
input_tensor_list = [torch.tensor(seq) for seq in input_ids]
data_tensor = pad_sequence(input_tensor_list, batch_first=True, padding_value=0)
label_tensor = torch.tensor(labels)

# Enforce a Max Length (Truncation)
# Cut off extremely long titles
MAX_LEN = 1024
if data_tensor.shape[1] > MAX_LEN:
    data_tensor = data_tensor[:, :MAX_LEN]

train_len = int(0.8 * len(input_ids))
test_len = len(input_ids) - train_len

print(f"Data Shape: {data_tensor.shape}")   # (N, max_len)
print(f"Label Shape: {label_tensor.shape}") # (N,)

train_dataset = torch.utils.data.TensorDataset(data_tensor[:train_len,:], label_tensor[:train_len])
test_dataset = torch.utils.data.TensorDataset(data_tensor[train_len:,:], label_tensor[train_len:])


train_loder = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loder = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)



# -----------------------------------------------------------------
#                          Train Model
# -----------------------------------------------------------------
block_size = 1024
number_heads = 12
n_classes = 2

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # For pretraining: 0 is good ; For finetuning try 0.1+
bias = False # For LayerNorm and Linear layers

# adamw optimizer
learning_rate = 6e-4    # max learning rate
epochs = 15             # total number of training iterations

print("Initializing a new model from scratch")

vocab_size = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# embbed_layer = CompleteEmbedding(
#     embedding_dim=n_embd, 
#     vocab_size=vocab_size, 
#     context_len=vocab_size
# ).to(device)

model = Encoder(
    n_layers=n_layer, 
    vocab_size=vocab_size,
    embedding_dim=n_embd,
    bias=bias, 
    num_heads=n_head, 
    output_size=n_classes,
    dropout=dropout,
).to(device)


# all_params = list(model.parameters()) + list(embbed_layer.parameters())
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)


def train(epochs: int, train_dataloader: DataLoader, optimizer: optim.Optimizer, model: nn.Module):
    print("----------------  Training  ----------------")
    correct = 0
    total_loss = 0
    size = len(train_dataloader.dataset)
    model.train()
    
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        
        pred, loss = model(X, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred_labels = pred.argmax(dim=1)
        correct += (pred_labels == y.squeeze()).sum().item()

        if batch % 10 == 0:
            current = (batch + 1) * len(X)
            print(f"batch: {batch:>3d}  |  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    avg_loss = total_loss / len(train_dataloader)
    avg_accuracy = 100 * (correct / size)
    print(f"\tAvg Loss: {avg_loss:.6f}  |  Avg Accuracy: {avg_accuracy:.2f}%")
    return avg_loss, avg_accuracy



def test(epochs:int, test_dataloader:DataLoader, model:nn.Module):
    print("----------------  Testing  ----------------")
    correct = 0
    total_loss = 0
    size = len(test_dataloader.dataset)
    model.eval()
    
    with torch.no_grad():
        for (X, y) in test_dataloader:
            X, y = X.to(device), y.to(device)

            pred, loss = model(X, y)
            
            total_loss += loss.item()
            pred_labels = pred.argmax(dim=1)
            correct += (pred_labels == y.squeeze()).sum().item()
            
    avg_loss = total_loss/len(test_dataloader)
    avg_accuracy = 100 * (correct/size)
    print(f"\tAvg Loss: {avg_loss:>4f}  |  Avg Accuracy: {int(avg_accuracy)}%")
    return avg_loss, avg_accuracy


epochs = 5
best_val_loss = float("inf")
best_val_accuracy = 0

# train_writer = SummaryWriter(log_dir='./classifier_tensorboard/run2/training')
# test_writer = SummaryWriter(log_dir='./classifier_tensorboard/run2/test')

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

learning_rate = []

loss_model_path = "best_loss.pth"
acc_model_path = "best_acc.pth"
best_loss_epoch = 0
best_accuracy_epoch = 0

for i in range(epochs):
    print(f"Epoch {i+1} | Current LR: {optimizer.param_groups[0]['lr']}")
    learning_rate.append(optimizer.param_groups[0]['lr'])
    train_loss, train_acc = train(epochs=i + 1, train_dataloader=train_loder, model=model, optimizer=optimizer)
    # train_writer.add_scalar("Loss/train", train_loss, epochs)
    # train_writer.add_scalar("Accurary/Epoch", train_acc, epochs)
    # train_writer.flush()
    
    val_loss, val_acc = test(epochs=i + 1, test_dataloader=test_loder, model=model)
    # test_writer.add_scalar("Loss/train", val_loss, epochs)
    # test_writer.add_scalar("Accurary/Epoch", val_acc, epochs)
    # test_writer.flush()
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    if val_loss < best_val_loss:
        best_loss_epoch = i
        best_val_loss = val_loss
        # loss_model_path = f"best_loss_model_{best_val_loss:.3f}.pth"
        torch.save(model.state_dict(), loss_model_path)
        print("✅ Saved Best Loss Model")
    
    if val_acc > best_val_accuracy:
        # acc_model_path = f"best_acc_model_{best_val_accuracy}.pth"
        best_val_accuracy = val_acc
        best_accuracy_epoch = i
        torch.save(model.state_dict(), acc_model_path)
        print("✅ Saved Best Accuracy Model")
    

print("Highest Testing Accuracy: ", best_val_accuracy, " | epoch: ", best_accuracy_epoch)
print("Least Testing Loss: ", best_val_loss, " | epoch: ", best_loss_epoch)
print("Done !!")


import matplotlib.pyplot as plt
# Plot Loss
plt.figure(figsize=(10, 4))
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', marker='o')
plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss', marker='o')
plt.axvline(x=best_accuracy_epoch+1, color='red', linestyle='--', linewidth=2, label=f'Min Val Accuracy ({best_val_accuracy:.2f})')
plt.axvline(x=best_loss_epoch+1, color='green', linestyle='--', linewidth=1.5, label=f'Min Val Loss ({best_val_loss:.2f})')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 4))
plt.plot(range(1, epochs + 1), train_accuracies, label='Train Accuracy', marker='o')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy', marker='o')
plt.axvline(x=best_accuracy_epoch+1, color='red', linestyle='--', linewidth=2, label=f'Min Val Accuracy ({best_val_accuracy:.2f})')
plt.axvline(x=best_loss_epoch+1, color='green', linestyle='--', linewidth=1.5, label=f'Min Val Loss ({best_val_loss:.2f})')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
