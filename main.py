import threading
import os

def run_script(script_path):
    os.system(f"python {script_path}")

# Ruta de tus scripts
script1_path = "deteccion_personas.py"
#script2_path = "face_recognition.py"

# Crear los hilos
thread1 = threading.Thread(target=run_script, args=(script1_path,))
#thread2 = threading.Thread(target=run_script, args=(script2_path,))

# Iniciar los hilos
thread1.start()
#thread2.start()

# Esperar a que los hilos terminen
thread1.join()
#thread2.join()
