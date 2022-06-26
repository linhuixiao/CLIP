import torch
import clip
from PIL import Image

import torch.onnx
import netron

export_onnx_graph = False

if export_onnx_graph:
    device = "cpu"
else:
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

print('cuda is: ', torch.cuda.is_available())

print('Model name listed by clip: ', clip.available_models())
# prints: clip.available_models() =  ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']

print('Begin to load model ===>')
# model, preprocess = clip.load("ViT-B/16", device=device)
model, preprocess = clip.load("ViT-B/32", device=device)
# model, preprocess = clip.load("RN50", device=device)
# model, preprocess = clip.load("RN101", device=device)
# model, preprocess = clip.load("RN50x4", device=device)
# model, preprocess = clip.load("RN50x16", device=device)
print('End')
model.eval()



# print(model.state_dict)
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())


# print("Total number of param in CLIP's is ", sum(x.numel() for x in model.state_dict))

# sum = 0
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
#     sum += model.state_dict()[param_tensor].parameters().numel()
# print('sum = ', sum)

# for param_tensor in model.state_dict():
#     print(param_tensor)


print("Total number of param in CLIP's is ", sum(x.numel() for x in model.parameters()))

# saved_models = torch.jit.load("/home/lhxiao/.cache/clip/ViT-B-32.pt", map_location="cpu")
# print("Total number of param in CLIP's is ", sum(x.numel() for x in saved_models['embedding'].state_dict().image_encoder.parameters()))


# print(model)

# saved_models = torch.jit.load("/home/lhxiao/.cache/clip/ViT-B-32.pt", map_location="cpu")
# embedding_net = torch.nn.Embedding
# embedding_net.load_state_dict(saved_models['embedding'])
# embedding_net.eval()
# print("Total number of param in CLIP's image_encoder is ", sum(x.numel() for x in embedding_net.image_encoder.parameters()))
# print("Total number of param in CLIP's text_encoder is ", sum(x.numel() for x in embedding_net.text_encoder.parameters()))

# preprocess 调用 torchvision 的 transform 预处理函数，将读取的图片转成 3 * 224 * 224 格式， ViT 模型只能处理 224 * 224 格式的图片
# 再将 3 维图片展开成 4 维，并放到 device 上
image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# image = preprocess(Image.open("test1.png")).unsqueeze(0).to(device)
# image = preprocess(Image.open("test2.png")).unsqueeze(0).to(device)
print(image.size())
# print("Total number of param in CLIP's image_encoder is ", sum(x.numel() for x in model.encode_image(image).parameters()))

# text 是 一个 3 * 77 维度的张量，而 77 维度中只有 前 4 项是有数值的。
# 进行分词， 最长长度 77
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
# print(text)

with torch.no_grad():
    # print(image.type())  # torch.cuda.FloatTensor
    # 调用 resnet 或者是 ViT 的 image encode 作为 encode编码器对图像编码，Transformer 作文文本 编码，为什么下面的 没有用上？（重复做了，演示）
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print(image_features.type())  # torch.cuda.HalfTensor
    print(image_features.size())  # 1 * 512
    print(text_features.size())   # 3 * 512

    # model 里面做了一次 图像、文本 编码
    # 最终输出是 1 * 3
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    if export_onnx_graph:
        onnx_path = "clip_onnx_model.onnx"
        torch.onnx.export(model, (image, text), onnx_path)
        netron.start(onnx_path)

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]






# ======================================================
'''
def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, osp.expanduser("/hdd/pengf/CLIP/models"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

saved_models = torch.load("~/.cache/clip/ViT-B-32.pt", map_location="cpu")
model =
embedding_net.load_state_dict(saved_models['embedding'])
embedding_net.eval()=

# x.numel()

print("Total number of param in CLIP's image_encoder is ", sum(x.numel() for x in embedding_net.image_encoder.parameters()))
print("Total number of param in CLIP's text_encoder is ", sum(x.numel() for x in embedding_net.text_encoder.parameters()))

model_path = '~/.cache/clip/ViT-B-32.pt'
saved_models = clip.load()
model, preprocess = clip.load("ViT-B/32", device=device)
'''

