import numba
from numba import cuda
import traceback

print("--- Проверка инициализации Numba CUDA ---")
try:
    print("Версия Numba:", numba.__version__)
    print("Пытаюсь получить список устройств...")
    devices = cuda.gpus
    print("Объекты устройств:", devices)
    print("Количество устройств:", len(devices))

    if len(devices) > 0:
        print("\nПытаюсь выбрать и активировать устройство 0...")
        cuda.select_device(0)
        device = cuda.get_current_device()
        print("Устройство успешно выбрано!")
        print("Имя устройства:", device.name.decode('UTF-8'))
        
        # --- ИСПРАВЛЕННЫЙ КОД ---
        # Старый, неработающий код:
        # print("Compute Capability:", f"{device.COMPUTE_CAPABILITY[0]}.{device.COMPUTE_CAPABILITY[1]}")
        
        # Новый код для современных версий Numba:
        major = device.COMPUTE_CAPABILITY_MAJOR
        minor = device.COMPUTE_CAPABILITY_MINOR
        print(f"Compute Capability: {major}.{minor}")
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        print("\n--- Инициализация прошла успешно! ---")
    else:
        print("\n--- Устройства не найдены Numba. ---")

except Exception as e:
    print("\n--- ПРОИЗОШЛА ОШИБКА ИНИЦИАЛИЗАЦИИ ---")
    traceback.print_exc()