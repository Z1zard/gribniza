import os
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import PySimpleGUI as sg

file_types = [("All files (*.*)", "*.*")]


#определим устройство, где проводить вычисления (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ("больной", "здоровый")

# зададим какие преобразования необходимо сделать с каждым изображением
resnet18_transform = transforms.Compose(
    [transforms.Resize((512,512)),  #изменим размер изображений
     transforms.ToTensor(),   #переведем в формат который необходим нейронной сети - тензор
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]) # проведем нормализацию изображения

resnet34_transform = transforms.Compose(
    [transforms.Resize((324,324)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def image_loader(image_path, model):
    """load image, returns cuda tensor"""
    image = Image.open(image_path)
    if model == 'resnet18':
        transformed_image = resnet18_transform(image).permute(1, 2, 0)
        transformed_image = transformed_image / 2 + 0.5  # денормализация
        image = resnet18_transform(image).float()
    elif model == 'resnet34':
        transformed_image = resnet34_transform(image).permute(1, 2, 0)
        transformed_image = transformed_image / 2 + 0.5  # денормализация
        image = resnet34_transform(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image.to(device)

def prediction(model_path, image_path, mdl):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.to(device)
    model.eval()
    # Проверка одного изображения
    image = image_loader(image_path, mdl)
    output = model(image)
    _, predicted = torch.max(output, 1)
    return classes[predicted]

# Интерфейс
elements = [
    [sg.Image(key="IMAGE")],
    [sg.Text("Выбор модели:")],
    [
        sg.Radio("ResNet18", "RADIO", default= True, key= "RADIO18"),
        sg.Radio("ResNet34", "RADIO", default= False, key= "RADIO34")
    ],
    [
        sg.Text("Изображение"),
        sg.Input(size=(25, 1), enable_events=True, key="FILE"),
        sg.FileBrowse("Найти", file_types=file_types),
    ],
    [sg.Button("Выполнить", key="LOAD")]
]
window = sg.Window("Eye Predict", elements)

while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    if event == "LOAD":
        filename = values["FILE"]
        if os.path.exists(filename):
            if values["RADIO18"] == True:
                model_path = os.path.abspath("resnet18.pth")
                mdl = 'resnet18'
            elif values["RADIO34"] == True:
                model_path = os.path.abspath("resnet34.pth")
                mdl = 'resnet34'
            image = Image.open(values["FILE"])
            image.thumbnail((400, 400))
            # bio = io.BytesIO()
            # image.save(bio, format="PNG")
            # window["-IMAGE-"].update(data=bio.getvalue())
            plt.imshow(image)
            plt.title(prediction(model_path, values["FILE"], mdl), fontsize=20)
            plt.show()

window.close()