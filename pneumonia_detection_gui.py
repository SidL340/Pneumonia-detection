import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
MODEL_PATH = "pneumonia_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess the uploaded image
def preprocess_image(image_path):
    img = Image.open(image_path)
    
    # Convert the image to RGB if it's not
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img = img.resize((120, 120))  # Resize to match model input (120, 120)
    img_array = np.array(img) / 255.0  # Normalize the image
    return np.expand_dims(img_array, axis=0)

# Function to predict pneumonia
def predict_pneumonia():
    image_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if image_path:
        try:
            img_array = preprocess_image(image_path)
            print(f"Image shape after preprocessing: {img_array.shape}")  # Debugging line

            prediction = model.predict(img_array)
            print(f"Model prediction: {prediction}")  # Debugging line
            
            probability = prediction[0][0] * 100  # Convert to percentage
            print(f"Pneumonia Probability: {probability:.2f}%")  # Debugging line

            # Update result label with probability and health quote
            result_label.config(
                text=f"Pneumonia Probability: {probability:.2f}%\n"
                     "“The greatest wealth is health.” – Virgil"
            )

            # Display the uploaded image
            display_image(image_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

# Function to display the uploaded image
def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((250, 250), Image.LANCZOS)  # Resize image for display
    img_tk = ImageTk.PhotoImage(img)
    
    # Show image in label
    image_label.config(image=img_tk)
    image_label.image = img_tk  # Keep a reference

# Function to reset the GUI elements
def reset_gui():
    # Clear the result label and image label
    result_label.config(text="")
    image_label.config(image='')
    image_label.image = None  # Clear the reference

# Set up the main application window
app = tk.Tk()
app.title("Pneumonia Detection")
app.geometry("400x500")

# Load background image
background_image = Image.open("image.png")
background_image = background_image.resize((400, 500), Image.LANCZOS)  # Resize to match window size
background_tk = ImageTk.PhotoImage(background_image)

# Create a canvas to hold the background
canvas = tk.Canvas(app, width=400, height=500)
canvas.pack()

# Set the background image
canvas.create_image(0, 0, anchor="nw", image=background_tk)

# Create a label for the hospital name at the top
hospital_label = tk.Label(app, text="XID Hospital", font=("Helvetica", 16, "bold"), bg="#e0f7fa")
canvas.create_window(200, 20, window=hospital_label)  # Position hospital name

# Create a button to upload image
upload_button = ttk.Button(app, text="Upload Chest X-Ray Image", command=predict_pneumonia)
canvas.create_window(200, 150, window=upload_button)  # Position button

# Create a label to show the uploaded image
image_label = tk.Label(app, bg="#e0f7fa")
canvas.create_window(200, 220, window=image_label)  # Position image label

# Create a label to display the result (probability)
result_label = tk.Label(app, text="", font=("Helvetica", 14), bg="#e0f7fa")
canvas.create_window(200, 350, window=result_label)  # Position result label at the bottom

# Create a reset button to allow for another prediction
reset_button = ttk.Button(app, text="Reset", command=reset_gui)
canvas.create_window(200, 400, window=reset_button)  # Position reset button

# Start the GUI event loop
app.mainloop()
