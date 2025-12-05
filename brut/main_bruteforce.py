import sys
import multiprocessing
import time
# Импортируем старые GPU модули
from md5_brute import MD5BruteCracker
from sha1_brute import SHA1BruteCracker
# Импортируем новые CPU/Rust модули
from bcrypt_brute import BCryptBruteCracker
from argon2_brute import Argon2BruteCracker

def main():
    print("=== UNIVERSAL BRUTEFORCE (MD5 / SHA-1 / BCRYPT / ARGON2) ===")
    
    target_hash = input("Введите хеш: ").strip()
    if not target_hash:
        print("Нужен хеш!")
        return

    hash_len = len(target_hash)
    cracker = None

    # --- ЛОГИКА ОПРЕДЕЛЕНИЯ АЛГОРИТМА ---
    
    # Argon2 обычно начинается с $argon2...
    if target_hash.startswith("$argon2"):
        print("Обнаружен Argon2 (используем CPU/Rust)")
        cracker = Argon2BruteCracker(target_hash)
        
    # BCrypt начинается с $2a$, $2b$, $2y$ и имеет длину 60 (иногда 59)
    elif target_hash.startswith("$2") and hash_len in [59, 60]:
        print("Обнаружен bCrypt (используем CPU/Rust)")
        cracker = BCryptBruteCracker(target_hash)
        
    # MD5 (hex) - 32 символа
    elif hash_len == 32:
        print("Обнаружен MD5 (используем GPU/CUDA)")
        cracker = MD5BruteCracker(target_hash)
        
    # SHA-1 (hex) - 40 символов
    elif hash_len == 40:
        print("Обнаружен SHA-1 (используем GPU/CUDA)")
        cracker = SHA1BruteCracker(target_hash)
        
    else:
        print(f"Неизвестный формат или длина. Длина: {hash_len}")
        return

    try:
        length_str = input("Максимальная длина пароля: ").strip()
        max_length = int(length_str) if length_str else 6
    except ValueError:
        print("Неверная длина")
        return

    print(f"Запуск перебора от 1 до {max_length} символов...")

    start_global = time.time()
    
    for length in range(1, max_length + 1):
        print(f"\n--- Проверка длины: {length} ---")
        
        # Вызываем унифицированный метод brute
        result = cracker.brute(length=length)
        
        if result:
            print(f"Пароль успешно найден на длине {length}!")
            break
    else:
        print("\nПеребор завершен. Пароль не найден.")
        
    print(f"Общее время работы: {time.time() - start_global:.2f}s")

if __name__ == "__main__":
    # Для Windows обязателен freeze_support при multiprocessing
    multiprocessing.freeze_support()
    main()