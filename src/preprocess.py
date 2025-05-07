import tkinter as tk
from tkinter import messagebox

def hello():
    messagebox.showinfo("Merhaba", "Hello, world!")

# Pencere oluştur
pencere = tk.Tk()
pencere.title("Demo Arayüz")
pencere.geometry("300x200")

# Buton ekle
buton = tk.Button(pencere, text="Tıkla!", command=hello)
buton.pack(pady=50)

# Arayüzü başlat
pencere.mainloop()
