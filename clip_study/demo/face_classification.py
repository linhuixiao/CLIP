import os
from torch.utils.data import DataLoader
import clip
import torch
import torchvision
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

def model_load(model_name):
    # 加载模型
    model, preprocess = clip.load(model_name, device) #ViT-B/32 RN50x16
    return model, preprocess

def data_load(data_path):
    #加载数据集和文字描述
    celeba = torchvision.datasets.CelebA(root='/hdd/xiaolinhui/clip/CELEBA', split='test', download=True)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in celeba.attr_names]).to(device)
    return celeba, text_inputs


def test_model(start, end, celeba, text_inputs, model, preprocess):
    #测试模型
    length = end - start + 1
    face_accuracy = 0
    face_score = 0

    for i, data in enumerate(celeba):
        face_result = 0
        if i < start:
            continue
        image, target = data
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_score, top_label = text_probs.topk(6, dim=-1)
        for k, score in zip(top_label[0], top_score[0]):
            if k.item() < 40 and target[k.item()] == 1:
                face_result = 1
                face_score += score.item()
                print('Predict right! The predicted is {}'.format(celeba.attr_names[k.item()]))
            else:
                print('Predict flase! The predicted is {}'.format(celeba.attr_names[k.item()]))
        face_accuracy += face_result

        if i == end:
            break
    face_score = face_score / length
    face_accuracy = face_accuracy / length

    return face_score, face_accuracy

def main():
    start = 0
    end = 1000
    model_name = 'ViT-B/32' #ViT-B/32 RN50x16
    data_path = 'CELEBA'

    time_start = time.time()
    print("Beigin to load model ===>")
    model, preprocess = model_load(model_name)
    print("load model end.")
    print("Beigin to load data ===>")
    celeba, text_inputs = data_load(data_path)
    print("load data end.")
    print("Beigin to perform face classification ===>")
    face_score, face_accuracy = test_model(start, end, celeba, text_inputs, model, preprocess)
    time_end = time.time()
    print("End.")

    print('The prediction:')
    print('face_accuracy: {:.2f} face_score: {}%'.format(face_accuracy, face_score*100))
    print('runing time: %.4f'%(time_end - time_start))

if __name__ == '__main__':
    main()

