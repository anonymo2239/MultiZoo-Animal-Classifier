import tkinter as tk
from tkinter import Label, filedialog, ttk
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from transformers import AutoModelForImageClassification, AutoConfig

class_names = [
    "antilop", "porsuk", "yarasa", "ayı", "arı", "böcek", "bizon", "yaban domuzu",
    "kelebek", "kedi", "tırtıl", "şempanze", "hamam böceği", "inek", "çakal", "yengeç", 
    "karga", "geyik", "köpek", "yunus", "eşek", "helikopter böceği", "ördek", "kartal",
    "fil", "flamingo", "sinek", "tilki", "keçi", "japon balığı", "kaz", "goril", "çekirge",
    "hamster", "yaban tavşanı", "kirpi", "su aygırı", "kasklı guguk kuşu", "at", "sinek kuşu",
    "sırtlan", "denizanası", "kanguru", "koala", "uğur böceği", "leopar", "aslan", "kertenkele",
    "ıstakoz", "sivrisinek", "güve", "fare", "ahtapot", "okapi", "orangutan", "su samuru", "baykuş",
    "öküz", "istiridye", "panda", "papağan", "pelikan", "penguen", "domuz", "güvercin", "dikenli kirpi",
    "keseli sıçan", "rakun", "sıçan", "ren geyiği", "gergedan", "kıyı kuşu", "denizatı", "fok",
    "köpekbalığı", "koyun", "yılan", "serçe", "mürekkep balığı", "sincap", "deniz yıldızı", "kuğu",
    "kaplan", "hindi", "kaplumbağa", "balina", "kurt", "vombat", "ağaçkakan", "zebra"
]

model_options = {
    "DeiT (facebook/deit-base)": "facebook/deit-base-patch16-224",
    "SwinV2 (microsoft/swinv2-tiny)": "microsoft/swinv2-tiny-patch4-window16-256"
}

model_paths = {
    "facebook/deit-base-patch16-224": "src/model/deit/deit_base_model.pt",
    "microsoft/swinv2-tiny-patch4-window16-256": "src/model/swin/swin_model.pt"
}

image_paths = {
    "facebook/deit-base-patch16-224": "src/images/deit_info.png",
    "microsoft/swinv2-tiny-patch4-window16-256": "src/images/swin_info.png"
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

selected_model_name = list(model_options.values())[1]
model = None
image_path = None

def load_model(model_name):
    config = AutoConfig.from_pretrained(model_name)
    config.num_labels = len(class_names)
    model = AutoModelForImageClassification.from_config(config)
    model.load_state_dict(torch.load(model_paths[model_name], map_location="cpu"))
    model.eval()
    return model

def change_model(event):
    global model
    selected = model_combobox.get()
    selected_model = model_options[selected]
    model = load_model(selected_model)
    prediction_label.config(text=f"{selected} modeli yüklendi.")

def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f'{width}x{height}+{x}+{y}')

def run_model():
    global model
    if image_path:
        img = Image.open(image_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor).logits
            probabilities = torch.softmax(outputs, dim=1)
            top3 = torch.topk(probabilities, 3)

            top_indices = top3.indices[0].tolist()
            top_scores = top3.values[0].tolist()

        result_lines = []
        for i in range(3):
            class_name = class_names[top_indices[i]].capitalize()
            confidence = top_scores[i] * 100
            result_lines.append(f"{i+1}. {class_name}: %{confidence:.2f}")

        result_text = "Tahminler:\n" + "\n".join(result_lines)
        prediction_label.config(text=result_text, font=("Montserrat", 16))
    else:
        prediction_label.config(
            text="Lütfen önce bir görsel yükleyin.",
            font=("Montserrat", 14)
        )

def infoPage():
    selected = model_combobox.get()
    selected_model = model_options[selected]
    image_path_info = image_paths[selected_model]

    info_window = tk.Toplevel(window)
    info_window.title("Model Bilgisi")
    info_window.geometry("530x620")
    info_window.config(bg=bek)
    center_window(info_window, 530, 620)

    info_img = Image.open(image_path_info).resize((528, 615), Image.LANCZOS)
    info_img = ImageTk.PhotoImage(info_img)

    info_label = Label(info_window, image=info_img, bg=bek)
    info_label.image = info_img
    info_label.pack(pady=10)

def imageUploader():
    global image_path
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = filedialog.askopenfilename(filetypes=fileTypes)

    if len(path):
        image_path = path
        img = Image.open(path).resize((300, 300))
        pic = ImageTk.PhotoImage(img)
        uploaded_label.config(image=pic)
        uploaded_label.image = pic
        uploaded_label.place(x=(window_width - 300) // 2, y=60)
        window.update()
    else:
        print("No file is chosen !! Please choose a file.")


# GUI Başlat
window = tk.Tk()
bek = '#0E1821'
window.title('MultiZoo Animal Classifier')
window_width = 865
window_height = 490
center_window(window, window_width, window_height)
window.config(bg=bek)

# Logo
original_img = Image.open('src/images/yazlablogo.png')
resized_img = original_img.resize((500, 400), Image.LANCZOS)
img = ImageTk.PhotoImage(resized_img)
center_x = (window_width - 500) // 2 - 10
center_y = (window_height - 400) // 2 - 30
imgLabel = Label(window, image=img, bg=bek)
imgLabel.image = img
imgLabel.place(x=center_x, y=center_y)

# Combobox
model_combobox = ttk.Combobox(
    window,
    values=list(model_options.keys()),
    font=("Montserrat", 12),
    state="readonly",
    width=20
)
model_combobox.current(1)
model_combobox.place(x=window_width - 260, y=20)
model_combobox.bind("<<ComboboxSelected>>", change_model)

# İlk modeli yükle
model = load_model(selected_model_name)

# Görsel kutusu
uploaded_label = Label(window, bg="white", bd=2, relief="solid")
uploaded_label.place_forget()

# Tahmin kutusu
prediction_label = Label(window, text="", fg="white", bg=bek, justify="left")
prediction_label.place(x=20, y=200)

# Butonlar
window.option_add("*Label*Background", bek)
window.option_add("*Button*Background", "#FD6915")

uploadButton = tk.Button(window, text="Yeni Hayvan Resmi Yükle", command=imageUploader)
uploadButton.place(x=100, y=420, anchor=tk.CENTER)

infoButton = tk.Button(window, text="Model Bilgisi", font=("Montserrat", 14), command=infoPage)
infoButton.place(x=window_width - 155, y=50)

run_button = tk.Button(text="Hayvanı Tahmin Et", font=("Montserrat", 14), command=run_model)
run_button.place(x=420, y=420, anchor=tk.CENTER)

window.mainloop()
