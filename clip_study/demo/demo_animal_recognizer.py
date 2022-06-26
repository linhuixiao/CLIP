import torch
import clip
from PIL import Image
import requests

# 1.导入CLIP, 加载模型和图像预处理
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# 2.提取图像嵌入
# 设置图片的URL
#image_name = "pexels-photo-1485637.jpeg"
#image_url = f"https://images.pexels.com/photos/1485637/{image_name}?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940"
#image = Image.open(requests.get(image_url, stream=True).raw)
# 加载图片
image = Image.open("test3.jpg")
print("Image to be processed")
# display(image)  # TODO 有问题
# 预处理图像
image = preprocess(image).unsqueeze(0).to(device)  # 将图片预处理成 1 * 224 * 224 * 3
print("\n\nTensor shape:")
print(image.shape)
#“encode_image”方法提取图像特征
with torch.no_grad():
    image_features = model.encode_image(image)
print(image_features.shape)  # encoder 图片成 1 * 512 向量

#3.提取文本嵌入
text_snippets = ["a photo of a dog", "a photo of a cat", "a photo of a tiger"]
# 预处理文本
text = clip.tokenize(text_snippets).to(device)
print(text.shape)            # 文本 encoder 成 3 * 77
#调用“encode_text”方法来提取文本特征
with torch.no_grad():
    text_features = model.encode_text(text)
print(text_features.shape)   # 文本特征 3 * 512

#4.比较图像嵌入和文本嵌入
with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)


