import torch
import clip
from PIL import Image
import torch.onnx
import netron
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

export_onnx_graph = False

if export_onnx_graph:
    device = "cpu"
else:
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

print('cuda is: ', torch.cuda.is_available())

print('Model name listed by clip: ', clip.available_models())
# prints: clip.available_models() =  ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']

print('Begin to load model ===>')
model, preprocess = clip.load("ViT-B/32", device=device)
print('End')
model.eval()

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
# image = preprocess(Image.open("test1.png")).unsqueeze(0).to(device)
# image = preprocess(Image.open("test2.png")).unsqueeze(0).to(device)
print(image.size())

# text 是 一个 3 * 77 维度的张量，而 77 维度中只有 前 4 项是有数值的。进行分词， 最长长度 77
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
# print(text)  # 是一个 3*77 的 张量，每个1*77的张量里面，只有前 4 位有数值，因为是 <Star token> + a + dog + <end token>
# text_decode = [_Tokenizer.decode(i) for i in text]
# 有bug： text_decode = _Tokenizer.decode(text[1].to('cpu'))
# print(text_decode)

with torch.no_grad():
    # print(image.type())  # torch.cuda.FloatTensor
    # 调用 resnet 或者是 ViT 的 image encode 作为 encode编码器对图像编码，Transformer 作文文本 编码，为什么下面的 没有用上？（重复做了，演示）
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print(image_features.type())  # torch.cuda.HalfTensor， N C H W
    print(image_features.size())  # 1 * 512
    print(text_features.size())   # 3 * 512

    # model 里面做了一次 图像、文本 编码, 最终输出是 1 * 3
    logits_per_image, logits_per_text = model(image, text)
    # 输出的logits和最终输出的probs是同维度的，唯一区别在于，1、未做softmax，2、在设备端，3、采用pytorch内存，未转换成正常数组
    # print(logits_per_image)  # tensor([[25.5625, 20.0938, 19.7500]], device='cuda:1', dtype=torch.float16)
    # 文本 logits 和 图片 logits 刚好是对称关系
    # print(logits_per_text)
    # 需要对最后一个维度做 softmax
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    # print("logit_scale: ", model.logit_scale)  # tensor(4.6052, device='cuda:1', requires_grad=True), 其实就是ln(100)
    # print("logit_scale: ", model.logit_scale.exp())  # tensor(100., device='cuda:1')

    if export_onnx_graph:
        onnx_path = "clip_onnx_model.onnx"
        torch.onnx.export(model, (image, text), onnx_path)
        netron.start(onnx_path)

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]



