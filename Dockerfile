# Usa una imagen base de Python 3.12
FROM python:3.12-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . /app/

# Instala las dependencias del proyecto
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto si estás ejecutando una aplicación web (por ejemplo, Gradio)
EXPOSE 8080

# Comando por defecto para ejecutar la aplicación (ajusta el comando según tu archivo principal)
CMD ["python", "gradio_app/main.py"]
