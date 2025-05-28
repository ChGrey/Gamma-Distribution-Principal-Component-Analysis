import os
import json
import numpy
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from vit_model import vit_base_patch16_224_in21k as create_model
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # whether to use Gamma-PCA
    is_gpca = False
    kernel_dir = './mat/0_90'

    if is_gpca:
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    else:
        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
             ])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # load image
    path = "./test/all"
    assert os.path.exists(path), "file: '{}' dose not exist.".format(path)
    class_list = os.listdir(path)
    class_list.sort()

    # create model
    model = create_model(num_classes=10, has_logits=False).to(device)
    # load model weights
    weight_path = "./model/0_90/org1.pth"
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    n_total = 0
    n_true = 0
    n_acc_total = 0.0

    true_labels = []
    pred_labels = []

    for i in range(len(class_list)):
        nc_true = 0
        class_path = os.path.join(path, class_list[i])
        images = os.listdir(class_path)
        nc_total = len(images)

        for image in images:
            img_path = os.path.join(class_path, image)
            img = Image.open(img_path)
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                output = torch.squeeze(model(img.to(device), is_gpca, kernel_dir)).cpu()
                predict = torch.softmax(output, dim=0)
                final_predict = torch.argmax(predict).numpy()

            predict_label = class_indict[str(final_predict)]
            true_label = str(class_list[i])

            true_labels.append(true_label)
            pred_labels.append(predict_label)

            if predict_label == true_label:
                nc_true += 1

        nc_acc = round(nc_true / nc_total, 4)
        print(f"{class_list[i]}: {nc_true}/{nc_total} {nc_acc}")
        n_true += nc_true
        n_total += nc_total
        n_acc_total += nc_acc

    print(f"OA: {n_true}/{n_total} {round(n_true / n_total, 4)}")
    print(f"AA: {round(n_acc_total / len(class_list), 4)}")

    cm = confusion_matrix(true_labels, pred_labels, labels=class_list)
    cm = np.array(cm)
    total_samples = np.sum(cm)

    plt.figure(figsize=(10, 8))
    df_cm = pd.DataFrame(cm, index=class_list, columns=class_list)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

    print("\nClass-wise Performance Metrics:")
    print("{:<15} {:<10} {:<10} {:<10} {:<10}".format(
        "Class", "TPR", "FPR", "Precision", "F1-score"))
    print("-" * 55)

    macro_TPR = 0
    macro_FPR = 0
    macro_Precision = 0
    macro_F1 = 0

    micro_TP = 0
    micro_FP = 0
    micro_FN = 0
    micro_TN = 0

    for i, class_name in enumerate(class_list):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = total_samples - TP - FP - FN

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        F1 = 2 * (Precision * TPR) / (Precision + TPR) if (Precision + TPR) > 0 else 0

        print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
            class_name, TPR, FPR, Precision, F1))

        macro_TPR += TPR
        macro_FPR += FPR
        macro_Precision += Precision
        macro_F1 += F1

        micro_TP += TP
        micro_FP += FP
        micro_FN += FN
        micro_TN += TN

    macro_TPR /= len(class_list)
    macro_FPR /= len(class_list)
    macro_Precision /= len(class_list)
    macro_F1 /= len(class_list)

    micro_TPR = micro_TP / (micro_TP + micro_FN) if (micro_TP + micro_FN) > 0 else 0
    micro_FPR = micro_FP / (micro_FP + micro_TN) if (micro_FP + micro_TN) > 0 else 0
    micro_Precision = micro_TP / (micro_TP + micro_FP) if (micro_TP + micro_FP) > 0 else 0
    micro_F1 = 2 * (micro_Precision * micro_TPR) / (micro_Precision + micro_TPR) if (micro_Precision + micro_TPR) > 0 else 0

    print("\nAverage Performance Metrics:")
    print("{:<15} {:<10} {:<10} {:<10} {:<10}".format(
        "Average Type", "TPR", "FPR", "Precision", "F1-score"))
    print("-" * 55)
    print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        "Macro", macro_TPR, macro_FPR, macro_Precision, macro_F1))
    print("{:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        "Micro", micro_TPR, micro_FPR, micro_Precision, micro_F1))


if __name__ == '__main__':
    main()