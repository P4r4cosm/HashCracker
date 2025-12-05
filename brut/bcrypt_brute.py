import time
import rust_cracker  # Импортируем нашу скомпилированную библиотеку
from base_cracker import CHARSET_PY

class BCryptBruteCracker:
    def __init__(self, target_hash):
        self.target_hash = target_hash

    def brute(self, length):
        print(f"Запуск CPU (Rust + Rayon) для bCrypt, длина: {length}")
        
        start_time = time.time()
        
        # Rust принимает строку (utf-8), а у тебя в base_cracker байты (b"abc...").
        # Нужно декодировать.
        charset_str = CHARSET_PY.decode('utf-8')
        
        # Вызываем функцию из Rust. Она сама распараллелит задачу на все ядра CPU.
        result = rust_cracker.brute_bcrypt(self.target_hash, length, charset_str)
        
        end_time = time.time()
        
        if result:
            print(f"\n[+] ПАРОЛЬ НАЙДЕН: {result}")
            print(f"Время: {end_time - start_time:.2f}s")
            return result
        else:
            return None