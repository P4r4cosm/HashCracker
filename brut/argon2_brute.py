import time
import rust_cracker
from base_cracker import CHARSET_PY

class Argon2BruteCracker:
    def __init__(self, target_hash):
        self.target_hash = target_hash

    def brute(self, length):
        print(f"Запуск CPU (Rust + Rayon) для Argon2, длина: {length}")
        start_time = time.time()
        
        charset_str = CHARSET_PY.decode('utf-8')
        
        # Вызываем Rust функцию
        result = rust_cracker.brute_argon2(self.target_hash, length, charset_str)
        
        if result:
            print(f"\n[+] ПАРОЛЬ НАЙДЕН: {result}")
            print(f"Время: {time.time() - start_time:.2f}s")
            return result
        return None