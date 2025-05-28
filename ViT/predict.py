import os
import json

import torch
from PIL import Image
from torchvision import transforms
from vit_model import vit_base_patch16_224_in21k as create_model


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # whether to use Gamma-PCA
    is_gpca = True
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
    weight_path = "./weights/best.pth"
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
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device), is_gpca, kernel_dir)).cpu()
                predict = torch.softmax(output, dim=0)
                final_predict = torch.argmax(predict).numpy()

            predict_label = class_indict[str(final_predict)]
            true_label = str(class_list[i])

            true_labels.append(true_label)
            pred_labels.append(predict_label)

            if predict_label == true_label:
                nc_true = nc_true + 1

        nc_acc = round(nc_true / nc_total, 4)
        print(str(class_list[i]) + ":", str(nc_true) + "/" + str(nc_total), str(nc_acc))
        n_true = n_true + nc_true
        n_total = n_total + nc_total
        n_acc_total = nc_acc + n_acc_total

    print("OA:", str(n_true) + "/" + str(n_total), str(round(n_true / n_total, 4)))
    print("AA:", str(round(n_acc_total / len(class_list), 4)))


if __name__ == '__main__':
    main()