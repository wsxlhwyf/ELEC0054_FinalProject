import os
os.environ['KMP_DUPLICATE_LIB_OK']="TRUE"
import torch.optim as optim
import torch
from modeling import ShareBottomCls
from dataloader import ModelNet40
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

# read data
train_dataset = ModelNet40(2048)  # read training data
test_dataset = ModelNet40(2048, 'test')   # read testing data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True) # packaging dataloader
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ShareBottomCls(num_classes=40) # initialize the model
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001) # optimizer
criterion = torch.nn.NLLLoss() # loss

# training process
def train(epoch, train_loss):
    model.train() # train
    for batch_idx, (data, target) in enumerate(train_loader):  
        data, target = data.to(device), target.reshape(-1,).to(device)
        optimizer.zero_grad() # gradient reset
        output = model(data) 
        loss = criterion(output, target) #  calculate loss
        loss.backward() # backward propogation
        optimizer.step() # optimize
        if batch_idx % 10 == 0: 
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    train_loss.append(loss.item())
    with open('train_loss.txt', 'a', encoding='utf-8') as file:
        file.write(str(loss.item()) + '\n')



# The following is the test function, which only lacks the backpropagation update compared to train()
def valid(test_loss_, test_acc, best):
    model.eval()
    test_loss = 0
    correct = 0
    pred_list, target_list = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.reshape(-1,).to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            pred_list += pred.detach().cpu().numpy().flatten().tolist()
            target_list += target.detach().cpu().numpy().flatten().tolist()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    test_loss_.append(test_loss)
    test_acc_ = correct / len(test_loader.dataset)
    test_acc.append(test_acc_)
    with open('test_loss.txt', 'a', encoding='utf-8') as file:
        file.write(str(test_loss) + '\n')
    with open('test_acc.txt', 'a', encoding='utf-8') as file:
        file.write(str(test_acc_) + '\n')
    if best < test_acc_:
        best = test_acc_
        cm = pd.DataFrame(confusion_matrix(target_list, pred_list))
        cm.to_csv('./confusion_matrix.csv', index=False)

    return best


if __name__ == '__main__':
    print('start training!')
    epochs = 10000000000000
    train_loss = []
    test_loss, test_acc = [], []
    best = 0
    for epoch in range(1, epochs):
        train(epoch, train_loss)
        best = valid(test_loss, test_acc, best)
    
    plt.figure(figsize = (5,5))
    plt.plot(range(1, epochs), train_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train loss')
    # plt.savefig('./image/train_loss.png')
    plt.show()
    
    plt.figure(figsize = (5,5))
    plt.plot(range(1, epochs), test_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('test loss')
    # plt.savefig('./image/test_loss.png')
    plt.show()
    
    plt.figure(figsize = (5,5))
    plt.plot(range(1, epochs), test_acc)
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.title('test acc')
    # plt.savefig('./image/test_acc.png')
    plt.show()