import numpy as np
import sys
import time
from numba import cuda, uint32, uint64, uint8
from base_cracker import BaseGPUCracker, CHARSET_ARR, CHARSET_LEN

# --- НАСТРОЙКИ ПОД RTX 3070 Ti ---
# Увеличиваем количество итераций внутри ядра.
# Видеокарта мощная, она может "крутиться" внутри долго, не возвращая управление CPU.
# Это снижает overhead на вызов ядра.
INNER_LOOPS = 128 

# Сдвиги для MD5 (hardcoded for speed)
S11, S12, S13, S14 = 7, 12, 17, 22
S21, S22, S23, S24 = 5,  9, 14, 20
S31, S32, S33, S34 = 4, 11, 16, 23
S41, S42, S43, S44 = 6, 10, 15, 21

@cuda.jit(device=True, inline=True)
def left_rotate(x, c):
    return (x << uint32(c)) | (x >> uint32(32 - c))

@cuda.jit(device=True, inline=True)
def f_func(x, y, z): return (x & y) | (~x & z)
@cuda.jit(device=True, inline=True)
def g_func(x, y, z): return (x & z) | (y & ~z)
@cuda.jit(device=True, inline=True)
def h_func(x, y, z): return x ^ y ^ z
@cuda.jit(device=True, inline=True)
def i_func(x, y, z): return y ^ (x | ~z)

@cuda.jit(device=True, inline=True)
def md5_step(a, b, c, d, k, s, x, func_type):
    """Один шаг MD5. func_type: 1=F, 2=G, 3=H, 4=I"""
    if func_type == 1: temp = f_func(b, c, d)
    elif func_type == 2: temp = g_func(b, c, d)
    elif func_type == 3: temp = h_func(b, c, d)
    else: temp = i_func(b, c, d)
    
    res = uint32(a + temp + x + k)
    return uint32(b + left_rotate(res, s))

@cuda.jit(device=True, inline=True)
def check_md5_fast(w0, w1, w2, w3, w14, target_hash):
    """
    Оптимизированная трансформация MD5.
    Принимает только изменяемые части буфера w0-w3 и длину w14.
    Остальные части буфера (w4-w13, w15) считаются нулями или статичным паддингом,
    который мы учитываем внутри (для упрощения считаем, что длина пароля < 32 байт,
    что верно для брутфорса).
    """
    a, b, c, d = uint32(0x67452301), uint32(0xefcdab89), uint32(0x98badcfe), uint32(0x10325476)

    # --- ROUND 1 ---
    a = md5_step(a, b, c, d, 0xd76aa478, S11, w0, 1)
    d = md5_step(d, a, b, c, 0xe8c7b756, S12, w1, 1)
    c = md5_step(c, d, a, b, 0x242070db, S13, w2, 1)
    b = md5_step(b, c, d, a, 0xc1bdceee, S14, w3, 1) # w3
    a = md5_step(a, b, c, d, 0xf57c0faf, S11, 0, 1)  # w4=0 assumed (pad logic separate)
    d = md5_step(d, a, b, c, 0x4787c62a, S12, 0, 1)  # w5=0
    c = md5_step(c, d, a, b, 0xa8304613, S13, 0, 1)
    b = md5_step(b, c, d, a, 0xfd469501, S14, 0, 1)
    a = md5_step(a, b, c, d, 0x698098d8, S11, 0, 1)
    d = md5_step(d, a, b, c, 0x8b44f7af, S12, 0, 1)
    c = md5_step(c, d, a, b, 0xffff5bb1, S13, 0, 1)
    b = md5_step(b, c, d, a, 0x895cd7be, S14, 0, 1)
    a = md5_step(a, b, c, d, 0x6b901122, S11, 0, 1)
    d = md5_step(d, a, b, c, 0xfd987193, S12, 0, 1)
    c = md5_step(c, d, a, b, 0xa679438e, S13, w14, 1) # Length
    b = md5_step(b, c, d, a, 0x49b40821, S14, 0, 1)

    # --- ROUND 2 (G func) ---
    # Индексы смешиваются: 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12
    a = md5_step(a, b, c, d, 0xf61e2562, S21, w1, 2)
    d = md5_step(d, a, b, c, 0xc040b340, S22, 0, 2)   # w6
    c = md5_step(c, d, a, b, 0x265e5a51, S23, 0, 2)   # w11
    b = md5_step(b, c, d, a, 0xe9b6c7aa, S24, w0, 2)  # w0
    a = md5_step(a, b, c, d, 0xd62f105d, S21, 0, 2)   # w5
    d = md5_step(d, a, b, c, 0x02441453, S22, 0, 2)   # w10
    c = md5_step(c, d, a, b, 0xd8a1e681, S23, 0, 2)   # w15=0
    b = md5_step(b, c, d, a, 0xe7d3fbc8, S24, 0, 2)   # w4
    a = md5_step(a, b, c, d, 0x21e1cde6, S21, 0, 2)   # w9
    d = md5_step(d, a, b, c, 0xc33707d6, S22, w14, 2) # w14
    c = md5_step(c, d, a, b, 0xf4d50d87, S23, w3, 2)  # w3
    b = md5_step(b, c, d, a, 0x455a14ed, S24, 0, 2)   # w8
    a = md5_step(a, b, c, d, 0xa9e3e905, S21, 0, 2)   # w13
    d = md5_step(d, a, b, c, 0xfcefa3f8, S22, w2, 2)  # w2
    c = md5_step(c, d, a, b, 0x676f02d9, S23, 0, 2)   # w7
    b = md5_step(b, c, d, a, 0x8d2a4c8a, S24, 0, 2)   # w12

    # --- ROUND 3 (H func) ---
    # Индексы: 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2
    a = md5_step(a, b, c, d, 0xfffa3942, S31, 0, 3)   # w5
    d = md5_step(d, a, b, c, 0x8771f681, S32, 0, 3)   # w8
    c = md5_step(c, d, a, b, 0x6d9d6122, S33, 0, 3)   # w11
    b = md5_step(b, c, d, a, 0xfde5380c, S34, w14, 3) # w14
    a = md5_step(a, b, c, d, 0xa4beea44, S31, w1, 3)  # w1
    d = md5_step(d, a, b, c, 0x4bdecfa9, S32, 0, 3)   # w4
    c = md5_step(c, d, a, b, 0xf6bb4b60, S33, 0, 3)   # w7
    b = md5_step(b, c, d, a, 0xbebfbc70, S34, 0, 3)   # w10
    a = md5_step(a, b, c, d, 0x289b7ec6, S31, 0, 3)   # w13
    d = md5_step(d, a, b, c, 0xeaa127fa, S32, w0, 3)  # w0
    c = md5_step(c, d, a, b, 0xd4ef3085, S33, w3, 3)  # w3
    b = md5_step(b, c, d, a, 0x04881d05, S34, 0, 3)   # w6
    a = md5_step(a, b, c, d, 0xd9d4d039, S31, 0, 3)   # w9
    d = md5_step(d, a, b, c, 0xe6db99e5, S32, 0, 3)   # w12
    c = md5_step(c, d, a, b, 0x1fa27cf8, S33, 0, 3)   # w15
    b = md5_step(b, c, d, a, 0xc4ac5665, S34, w2, 3)  # w2

    # --- ROUND 4 (I func) ---
    # Индексы: 0, 7, 14, 5, 12, 3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9
    a = md5_step(a, b, c, d, 0xf4292244, S41, w0, 4)  # w0
    d = md5_step(d, a, b, c, 0x432aff97, S42, 0, 4)   # w7
    c = md5_step(c, d, a, b, 0xab9423a7, S43, w14, 4) # w14
    b = md5_step(b, c, d, a, 0xfc93a039, S44, 0, 4)   # w5
    a = md5_step(a, b, c, d, 0x655b59c3, S41, 0, 4)   # w12
    d = md5_step(d, a, b, c, 0x8f0ccc92, S42, w3, 4)  # w3
    c = md5_step(c, d, a, b, 0xffeff47d, S43, 0, 4)   # w10
    b = md5_step(b, c, d, a, 0x85845dd1, S44, w1, 4)  # w1
    a = md5_step(a, b, c, d, 0x6fa87e4f, S41, 0, 4)   # w8
    d = md5_step(d, a, b, c, 0xfe2ce6e0, S42, 0, 4)   # w15
    c = md5_step(c, d, a, b, 0xa3014314, S43, 0, 4)   # w6
    b = md5_step(b, c, d, a, 0x4e0811a1, S44, 0, 4)   # w13
    a = md5_step(a, b, c, d, 0xf7537e82, S41, 0, 4)   # w4
    d = md5_step(d, a, b, c, 0xbd3af235, S42, 0, 4)   # w11
    c = md5_step(c, d, a, b, 0x2ad7d2bb, S43, w2, 4)  # w2
    b = md5_step(b, c, d, a, 0xeb86d391, S44, 0, 4)   # w9

    if (uint32(a + 0x67452301) == target_hash[0] and 
        uint32(b + 0xefcdab89) == target_hash[1] and 
        uint32(c + 0x98badcfe) == target_hash[2] and 
        uint32(d + 0x10325476) == target_hash[3]):
        return True
    return False

@cuda.jit(fastmath=True)
def md5_kernel_ampere(start_index, target_hash, found_info, word_len):
    """
    Оптимизированный кернел для Ampere. 
    - Использует регистры вместо локального массива для w0-w3.
    - Считает, что пароль <= 16 символов (иначе нужно w4+).
    - Если пароль > 16, код нужно дополнить w4.
    """
    idx = cuda.grid(1)
    if found_info[0] == 1: return

    CONST_CHARSET = cuda.const.array_like(CHARSET_ARR)
    charset_len_u64 = uint64(CHARSET_LEN)
    
    # Глобальный индекс потока
    current_iter_idx = uint64(start_index) + (uint64(idx) * uint64(INNER_LOOPS))

    # --- Инициализация (перевод индекса в строку/буфер) ---
    # Храним индексы символов в локальном массиве (это быстро, т.к. доступ редкий при инкременте)
    char_indices = cuda.local.array(16, dtype=uint8)
    
    temp_val = current_iter_idx
    for i in range(word_len):
        char_pos = word_len - 1 - i
        rem = uint32(temp_val % charset_len_u64)
        temp_val = temp_val // charset_len_u64
        char_indices[char_pos] = rem

    # --- Подготовка регистров w0, w1, w2, w3 ---
    w0, w1, w2, w3 = uint32(0), uint32(0), uint32(0), uint32(0)
    
    # Заполняем регистры из char_indices
    # Мы делаем это один раз перед циклом
    for pos in range(word_len):
        val = uint32(CONST_CHARSET[char_indices[pos]])
        word_idx = pos >> 2
        shift = (pos & 3) * 8
        if word_idx == 0: w0 |= (val << shift)
        elif word_idx == 1: w1 |= (val << shift)
        elif word_idx == 2: w2 |= (val << shift)
        elif word_idx == 3: w3 |= (val << shift)

    # Добавляем Padding (0x80)
    # Бит 0x80 ставится сразу после последнего символа
    pad_word_idx = word_len >> 2
    pad_shift = (word_len & 3) * 8
    if pad_word_idx == 0: w0 |= (uint32(0x80) << pad_shift)
    elif pad_word_idx == 1: w1 |= (uint32(0x80) << pad_shift)
    elif pad_word_idx == 2: w2 |= (uint32(0x80) << pad_shift)
    elif pad_word_idx == 3: w3 |= (uint32(0x80) << pad_shift)
    
    # Длина в битах (для w14)
    w14 = uint32(word_len * 8)

    # --- ГЛАВНЫЙ ЦИКЛ ---
    for loop_i in range(INNER_LOOPS):
        # 1. Проверка хеша (все w0...w3 уже в регистрах)
        if check_md5_fast(w0, w1, w2, w3, w14, target_hash):
            found_info[0] = 1
            found_info[1] = (uint64(idx) * uint64(INNER_LOOPS)) + uint64(loop_i)
            return

        # 2. Инкремент пароля (Odometer)
        # Обновляем char_indices и сразу же обновляем w0-w3
        pos = word_len - 1
        while pos >= 0:
            c_idx = char_indices[pos] + 1
            if c_idx < CHARSET_LEN:
                char_indices[pos] = c_idx
                new_char = uint32(CONST_CHARSET[c_idx])
                
                # Обновляем нужный байт в регистрах w
                word_idx = pos >> 2
                shift = (pos & 3) * 8
                mask = ~(uint32(0xFF) << shift)
                
                if word_idx == 0: w0 = (w0 & mask) | (new_char << shift)
                elif word_idx == 1: w1 = (w1 & mask) | (new_char << shift)
                elif word_idx == 2: w2 = (w2 & mask) | (new_char << shift)
                elif word_idx == 3: w3 = (w3 & mask) | (new_char << shift)
                
                break
            else:
                # Перенос (Carry)
                char_indices[pos] = 0
                new_char = uint32(CONST_CHARSET[0]) # 'a' или '0'
                
                word_idx = pos >> 2
                shift = (pos & 3) * 8
                mask = ~(uint32(0xFF) << shift)
                
                if word_idx == 0: w0 = (w0 & mask) | (new_char << shift)
                elif word_idx == 1: w1 = (w1 & mask) | (new_char << shift)
                elif word_idx == 2: w2 = (w2 & mask) | (new_char << shift)
                elif word_idx == 3: w3 = (w3 & mask) | (new_char << shift)

                pos -= 1

class MD5BruteCracker(BaseGPUCracker):
    def __init__(self, target_hash):
        super().__init__(target_hash)
        self.kernel = md5_kernel_ampere
        self.INNER_LOOPS = INNER_LOOPS
        # Увеличим батч еще сильнее, так как скорость вырастет
        self.BATCH_SIZE = 200_000_000 

    def _prepare_hash(self, hex_hash):
        h = []
        for i in range(0, 32, 8):
            chunk = hex_hash[i:i+8]
            # MD5 использует little-endian для слов, переворачиваем байты
            val = int(chunk[6:8] + chunk[4:6] + chunk[2:4] + chunk[0:2], 16)
            h.append(val)
        return cuda.to_device(np.array(h, dtype=np.uint32))

    def brute(self, length):
        # Быстрый фикс: данный оптимизированный кернел рассчитан на длину < 16 байт.
        # Если нужно > 16, придется добавить w4..w15 в логику ядра, что замедлит его.
        if length >= 16:
            print("Предупреждение: Данный оптимизированный кернел поддерживает длину < 16.")
            return None

        if self.kernel is None: raise ValueError("Kernel needed")
        
        print(f"Запуск Optimized Ampere MD5 для длины {length}...")
        print(f"Inner Loops: {self.INNER_LOOPS}")
        
        found_info_arr = np.zeros(3, dtype=np.uint64)
        found_info_gpu = cuda.to_device(found_info_arr)
        total_space = CHARSET_LEN ** length
        
        # Подбираем размеры блоков под Ampere (SMs = 48 на 3070 Ti)
        block_size = 128
        # Нам нужно забить GPU работой.
        # (BATCH_SIZE / INNER_LOOPS) = количество потоков, которые нужно запустить
        total_threads_needed = (self.BATCH_SIZE + self.INNER_LOOPS - 1) // self.INNER_LOOPS
        blocks = (total_threads_needed + block_size - 1) // block_size
        
        # Ограничим количество блоков разумным пределом, чтобы не крашнуть драйвер (TDR),
        # хотя с маленьким BATCH временем выполнения это не так страшно.
        # Для 3070 Ti можно смело ставить много блоков.
        
        real_batch_size = blocks * block_size * self.INNER_LOOPS

        current_idx = 0
        start_time = time.time()

        while current_idx < total_space:
            batch_start = time.time()
            
            self.kernel[blocks, block_size](
                current_idx, 
                self.target_hash_gpu, 
                found_info_gpu,
                length
            )
            cuda.synchronize()

            found_info = found_info_gpu.copy_to_host()
            if found_info[0] == 1:
                relative_idx = found_info[1] 
                global_idx = current_idx + relative_idx
                password = self._idx_to_pass(global_idx, length)
                print(f"\n\n[+] ПАРОЛЬ НАЙДЕН: {password}")
                print(f"Время: {time.time() - start_time:.2f}s")
                return password

            current_idx += real_batch_size
            
            batch_dur = time.time() - batch_start
            if batch_dur > 0:
                speed = real_batch_size / batch_dur
                percent = min(100, (current_idx / total_space) * 100)
                sys.stdout.write(f"\rProgress: {percent:5.2f}% | Speed: {speed/1_000_000:6.2f} MH/s")
                sys.stdout.flush()
        
        print("\nНе найдено.")
        return None