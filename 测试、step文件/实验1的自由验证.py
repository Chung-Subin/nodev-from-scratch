import torch
from torch import nn
from torchvision import transforms
from PIL import Image
img="number8.jpg"
device = "cuda" if torch.cuda.is_available() else "cpu"
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
def cnn():
    return CNN().to(device)
def preprocess_image(image_path):

    # 定义转换
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
        transforms.Resize((28, 28)),  # 调整为28x28
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化
    ])

    # 加载图片
    image = Image.open(image_path)

    # 应用转换
    image_tensor = transform(image)

    # 添加batch维度 [1, 1, 28, 28]
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor, image
model = cnn()
model.load_state_dict(torch.load("实验1/test_1.pth", weights_only=True))
model.eval()
image_tensor, image=preprocess_image(img)
with torch.no_grad():
    image_tensor = image_tensor.to(device)
    output = model(image_tensor)

        # 获取预测概率和类别
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item()
print(predicted_class,confidence)


