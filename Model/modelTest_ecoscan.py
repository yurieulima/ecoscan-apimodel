import numpy as np
from keras.api.models import load_model
from keras.api.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Caminho da imagem de teste
image_path = "/Users/yuri/EcoScan/ThrashTest2.jpeg"

# Carregar o modelo treinado
modelo_carregado = load_model("EcoScanModelV1.keras")

# Carregar e processar a imagem
img = load_img(image_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = img_array / 127.5 - 1  # Normalizar
img_array = np.expand_dims(img_array, axis=0)

plt.figure(figsize=(5,5))
plt.imshow(img)
plt.title("Imagem Original")
plt.axis("off")
plt.show()

# Fazer a predição com o modelo carregado
predictions = modelo_carregado.predict(img_array)
predicted_class = np.argmax(predictions)

# Mapear a classe para o nome correspondente (substitua pelos seus rótulos)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
print(f"Classe prevista: {class_names[predicted_class]} - Confiança: {predictions[0][predicted_class]*100:.2f}%")
