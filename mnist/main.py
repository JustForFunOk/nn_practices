import torch
import torchvision
import matplotlib.pyplot as plt
from cov_net import Net
import torch.optim as optim
import torch.nn.functional as F

batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
n_epochs = 3  # 将所有的数据训练3次
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Dataset
# https://pytorch.org/vision/0.17/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
training_data = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
)

print(type(training_data))  # <class 'torchvision.datasets.mnist.MNIST'>
print(len(training_data))  # 60000


test_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
)

print(type(test_data))  # # <class 'torchvision.datasets.mnist.MNIST'>
print(len(test_data))  # 10000

# DataLoader
# https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
train_loader = torch.utils.data.DataLoader(
    dataset=training_data,
    batch_size=batch_size_train,
    shuffle=True)  # 每个epoch中打乱数据顺序

test_loader = torch.utils.data.DataLoader(
    dataset=test_data,
    batch_size=batch_size_test,
    shuffle=True)  # 每个epoch中打乱数据顺序

print(type(test_loader))

# Visualize
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)  # torch.Size([1000, 1, 28, 28])
print(example_targets.shape)  # torch.Size([1000])

plt.figure()
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.savefig("dataset_demo.png")  # 保存到文件
plt.close()

# Optimizer
network = Net()
optimizer = optim.SGD(
    network.parameters(),
    lr=learning_rate,  # 学习率，默认0.001
    momentum=momentum  # 动量
)  # 随机梯度下降


train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 每个batch之前清理梯度以确保batch之间的独立性
        output = network(data)  # 前向传播，得到预测值
        # negative log likelihood，若网络最后一层用了LogSoftmax则损失函数用这个
        loss = F.nll_loss(output, target)
        loss.backward()  # 链式求导法则，对每个参数求偏导
        optimizer.step()  # 根据计算出的梯度更新模型参数
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), 'model.pth')
            torch.save(optimizer.state_dict(), 'optimizer.pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def export_last_to_onnx():
    network.eval()
    x = torch.randn(1, 1, 28, 28)
    torch.onnx.export(network, x, "mnist_conv2d_lr0.01_batch64_epoch3.onnx", input_names=[
                      'input'], output_names=['output'])


test()
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
export_last_to_onnx()

# Visualize loss
plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.savefig("train_and_test_loss.png")  # 保存到文件
plt.close()


# Visualize outputs
with torch.no_grad():  # 临时关闭梯度计算，提高计算效率
    output = network(example_data)

plt.figure()
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Prediction: {}".format(
        output.data.max(1, keepdim=True)[1][i].item()))
    plt.xticks([])
    plt.yticks([])
plt.savefig("predict_result_demo.png")
plt.close()
