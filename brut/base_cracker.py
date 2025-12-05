import time
import sys
import numpy as np
from numba import cuda

# Общий алфавит для всех крекеров
CHARSET_PY = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHARSET_LEN = len(CHARSET_PY)
CHARSET_ARR = np.ascontiguousarray(np.frombuffer(CHARSET_PY, dtype=np.uint8))

class BaseGPUCracker:
    def __init__(self, target_hash):
        self.target_hash_raw = target_hash
        self.target_hash_gpu = self._prepare_hash(target_hash)
        
        # Настройки производительности
        self.BATCH_SIZE = 100_000_000
        self.THREADS_PER_BLOCK = 256
        self.BLOCKS_PER_GRID = (self.BATCH_SIZE + self.THREADS_PER_BLOCK - 1) // self.THREADS_PER_BLOCK
        
        # Ссылка на ядро (должна быть определена в наследнике)
        self.kernel = None 

    def _prepare_hash(self, hex_hash):
        """
        Преобразует hex-строку в формат, удобный для GPU.
        Должен быть переопределен в наследнике.
        """
        raise NotImplementedError("Метод _prepare_hash должен быть переопределен")

    def brute(self, length):
        if self.kernel is None:
            raise ValueError("CUDA ядро не определено!")

        print(f"Запуск GPU Brute-Force для длины {length}...")
        
        # Массив для результата: [найдено (0/1), относительный_индекс]
        found_info_arr = np.zeros(3, dtype=np.uint64)
        found_info_gpu = cuda.to_device(found_info_arr)

        total_space = CHARSET_LEN ** length
        print(f"Всего комбинаций: {total_space:,}")

        start_time = time.time()
        current_idx = 0
        
        while current_idx < total_space:
            batch_start = time.time()
            
            # ЗАПУСК ЯДРА
            # Мы передаем kernel только специфичные данные, общие логические параметры делает Base
            self.kernel[self.BLOCKS_PER_GRID, self.THREADS_PER_BLOCK](
                current_idx, 
                self.target_hash_gpu, 
                found_info_gpu,
                length
            )
            
            # Ждем завершения вычислений на видеокарте
            cuda.synchronize()

            # Проверяем флаг успеха (копируем маленький массив обратно на CPU)
            found_info = found_info_gpu.copy_to_host()
            
            if found_info[0] == 1:
                relative_idx = found_info[1]
                global_idx = current_idx + relative_idx
                password = self._idx_to_pass(global_idx, length)
                total_time = time.time() - start_time
                print(f"\n\n[+] ПАРОЛЬ НАЙДЕН: {password}")
                print(f"Время: {total_time:.2f}s")
                return password

            current_idx += self.BATCH_SIZE
            batch_dur = time.time() - batch_start
            
            # Вывод прогресса
            if batch_dur > 0:
                speed = self.BATCH_SIZE / batch_dur
                percent = min(100, (current_idx / total_space) * 100)
                sys.stdout.write(f"\rProgress: {percent:5.2f}% | Speed: {speed/1_000_000:6.2f} MH/s | Checked: {current_idx:,}")
                sys.stdout.flush()

        print(f"\n[-] Пароль длины {length} не найден.")
        return None

    def _idx_to_pass(self, idx, length):
        """Восстанавливает строку пароля по глобальному индексу"""
        chars = []
        temp = int(idx)
        for _ in range(length):
            char_idx = temp % CHARSET_LEN
            chars.append(chr(CHARSET_ARR[char_idx]))
            temp //= CHARSET_LEN
        return "".join(reversed(chars))