import numpy as np
import sys
import time
from numba import cuda, uint32, uint64, uint8
from base_cracker import BaseGPUCracker, CHARSET_ARR, CHARSET_LEN

# --- CONFIG ---
# Увеличиваем loops, так как инициализация SHA-1 дорогая
INNER_LOOPS = 512 
BLOCK_SIZE = 128

@cuda.jit(device=True, inline=True)
def rotl(x, n):
    return (x << uint32(n)) | (x >> uint32(32 - n))

@cuda.jit(device=True, inline=True)
def check_sha1_circular(w0, w1, w2, w3, w15, target_hash):
    """
    Супер-оптимизированная версия с кольцевым буфером (Circular Buffer).
    Использует всего 16 слов памяти вместо 80.
    """
    # Initial State
    a = uint32(0x67452301)
    b = uint32(0xEFCDAB89)
    c = uint32(0x98BADCFE)
    d = uint32(0x10325476)
    e = uint32(0xC3D2E1F0)

    # Экономим память: массив всего на 16 элементов!
    W = cuda.local.array(16, dtype=uint32)

    # 1. Загрузка начальных данных
    W[0], W[1], W[2], W[3] = w0, w1, w2, w3
    # Заполнение нулями w4..w14 (развернуто для скорости)
    W[4] = 0; W[5] = 0; W[6] = 0; W[7] = 0
    W[8] = 0; W[9] = 0; W[10]= 0; W[11]= 0
    W[12]= 0; W[13]= 0; W[14]= 0
    W[15] = w15

    # 2. Главный цикл на 80 шагов
    # Мы вычисляем W[t] "на лету" и сразу используем.
    
    # --- ROUND 1 (0-19) ---
    # Logic: d ^ (b & (c ^ d))  <-- Оптимизация "Selection" вместо (b&c)|(~b&d)
    # K = 0x5A827999
    for t in range(0, 20):
        s = t & 0xF # index mod 16
        if t >= 16:
            # Message Schedule: W[t] = S1(W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16])
            # В кольцевом буфере:
            # t-3  -> (t + 13) % 16
            # t-8  -> (t + 8)  % 16
            # t-14 -> (t + 2)  % 16
            # t-16 -> t % 16 (текущая ячейка, которую перезаписываем)
            val = W[(t + 13) & 0xF] ^ W[(t + 8) & 0xF] ^ W[(t + 2) & 0xF] ^ W[s]
            W[s] = rotl(val, 1)
        
        w_val = W[s]
        # Func F: (b & c) | ((~b) & d) эквивалентно d ^ (b & (c ^ d))
        f = d ^ (b & (c ^ d))
        
        temp = uint32(rotl(a, 5) + f + e + 0x5A827999 + w_val)
        e = d
        d = c
        c = rotl(b, 30)
        b = a
        a = temp

    # --- ROUND 2 (20-39) ---
    # Logic: b ^ c ^ d
    # K = 0x6ED9EBA1
    for t in range(20, 40):
        s = t & 0xF
        # Schedule (можно не проверять t >= 16, тут t всегда >= 20)
        val = W[(t + 13) & 0xF] ^ W[(t + 8) & 0xF] ^ W[(t + 2) & 0xF] ^ W[s]
        W[s] = rotl(val, 1)
        
        f = b ^ c ^ d
        temp = uint32(rotl(a, 5) + f + e + 0x6ED9EBA1 + W[s])
        e = d
        d = c
        c = rotl(b, 30)
        b = a
        a = temp

    # --- ROUND 3 (40-59) ---
    # Logic: (b & c) | (b & d) | (c & d) <-- Majority
    # K = 0x8F1BBCDC
    for t in range(40, 60):
        s = t & 0xF
        val = W[(t + 13) & 0xF] ^ W[(t + 8) & 0xF] ^ W[(t + 2) & 0xF] ^ W[s]
        W[s] = rotl(val, 1)

        f = (b & c) | (b & d) | (c & d)
        temp = uint32(rotl(a, 5) + f + e + 0x8F1BBCDC + W[s])
        e = d
        d = c
        c = rotl(b, 30)
        b = a
        a = temp

    # --- ROUND 4 (60-79) ---
    # Logic: b ^ c ^ d
    # K = 0xCA62C1D6
    for t in range(60, 80):
        s = t & 0xF
        val = W[(t + 13) & 0xF] ^ W[(t + 8) & 0xF] ^ W[(t + 2) & 0xF] ^ W[s]
        W[s] = rotl(val, 1)

        f = b ^ c ^ d
        temp = uint32(rotl(a, 5) + f + e + 0xCA62C1D6 + W[s])
        e = d
        d = c
        c = rotl(b, 30)
        b = a
        a = temp

    # Check
    if (uint32(a + 0x67452301) == target_hash[0] and
        uint32(b + 0xEFCDAB89) == target_hash[1] and
        uint32(c + 0x98BADCFE) == target_hash[2] and
        uint32(d + 0x10325476) == target_hash[3] and
        uint32(e + 0xC3D2E1F0) == target_hash[4]):
        return True
    return False

@cuda.jit(fastmath=True)
def sha1_kernel_opt(start_index, target_hash, found_info, word_len):
    idx = cuda.grid(1)
    if found_info[0] == 1: return

    CONST_CHARSET = cuda.const.array_like(CHARSET_ARR)
    charset_len_u64 = uint64(CHARSET_LEN)
    
    current_iter_idx = uint64(start_index) + (uint64(idx) * uint64(INNER_LOOPS))

    char_indices = cuda.local.array(16, dtype=uint8)
    temp_val = current_iter_idx
    for i in range(word_len):
        char_pos = word_len - 1 - i
        rem = uint32(temp_val % charset_len_u64)
        temp_val = temp_val // charset_len_u64
        char_indices[char_pos] = rem

    # BIG ENDIAN PACKING
    w0, w1, w2, w3 = uint32(0), uint32(0), uint32(0), uint32(0)
    for pos in range(word_len):
        val = uint32(CONST_CHARSET[char_indices[pos]])
        word_idx = pos >> 2
        shift = (3 - (pos & 3)) * 8
        if word_idx == 0: w0 |= (val << shift)
        elif word_idx == 1: w1 |= (val << shift)
        elif word_idx == 2: w2 |= (val << shift)
        elif word_idx == 3: w3 |= (val << shift)

    pad_word_idx = word_len >> 2
    pad_shift = (3 - (word_len & 3)) * 8
    if pad_word_idx == 0: w0 |= (uint32(0x80) << pad_shift)
    elif pad_word_idx == 1: w1 |= (uint32(0x80) << pad_shift)
    elif pad_word_idx == 2: w2 |= (uint32(0x80) << pad_shift)
    elif pad_word_idx == 3: w3 |= (uint32(0x80) << pad_shift)
    
    w15 = uint32(word_len * 8)

    # LOOP
    for loop_i in range(INNER_LOOPS):
        if check_sha1_circular(w0, w1, w2, w3, w15, target_hash):
            found_info[0] = 1
            found_info[1] = (uint64(idx) * uint64(INNER_LOOPS)) + uint64(loop_i)
            return

        # Odometer Update
        pos = word_len - 1
        while pos >= 0:
            c_idx = char_indices[pos] + 1
            if c_idx < CHARSET_LEN:
                char_indices[pos] = c_idx
                new_char = uint32(CONST_CHARSET[c_idx])
                word_idx = pos >> 2
                shift = (3 - (pos & 3)) * 8
                mask = ~(uint32(0xFF) << shift)
                if word_idx == 0: w0 = (w0 & mask) | (new_char << shift)
                elif word_idx == 1: w1 = (w1 & mask) | (new_char << shift)
                elif word_idx == 2: w2 = (w2 & mask) | (new_char << shift)
                elif word_idx == 3: w3 = (w3 & mask) | (new_char << shift)
                break
            else:
                char_indices[pos] = 0
                new_char = uint32(CONST_CHARSET[0])
                word_idx = pos >> 2
                shift = (3 - (pos & 3)) * 8
                mask = ~(uint32(0xFF) << shift)
                if word_idx == 0: w0 = (w0 & mask) | (new_char << shift)
                elif word_idx == 1: w1 = (w1 & mask) | (new_char << shift)
                elif word_idx == 2: w2 = (w2 & mask) | (new_char << shift)
                elif word_idx == 3: w3 = (w3 & mask) | (new_char << shift)
                pos -= 1

class SHA1BruteCracker(BaseGPUCracker):
    def __init__(self, target_hash):
        super().__init__(target_hash)
        self.kernel = sha1_kernel_opt
        self.INNER_LOOPS = INNER_LOOPS
        self.BATCH_SIZE = 150_000_000 
        self.BLOCK_SIZE = BLOCK_SIZE

    def _prepare_hash(self, hex_hash):
        h = []
        if len(hex_hash) != 40: raise ValueError("Invalid SHA-1 length")
        for i in range(0, 40, 8):
            chunk = hex_hash[i:i+8]
            val = int(chunk, 16)
            h.append(val)
        return cuda.to_device(np.array(h, dtype=np.uint32))

    def brute(self, length):
        if length >= 14:
            print("Warning: Length >= 14 not supported by fast SHA-1 kernel.")
            return None

        if self.kernel is None: raise ValueError("Kernel needed")
        
        print(f"GPU SHA-1 Config: Loops={self.INNER_LOOPS}, BlockSize={self.BLOCK_SIZE} (Circular Buffer)")
        
        found_info_arr = np.zeros(3, dtype=np.uint64)
        found_info_gpu = cuda.to_device(found_info_arr)
        total_space = CHARSET_LEN ** length
        
        total_threads_needed = (self.BATCH_SIZE + self.INNER_LOOPS - 1) // self.INNER_LOOPS
        blocks = (total_threads_needed + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        real_batch_size = blocks * self.BLOCK_SIZE * self.INNER_LOOPS

        current_idx = 0
        start_time = time.time()

        while current_idx < total_space:
            batch_start = time.time()
            
            self.kernel[blocks, self.BLOCK_SIZE](
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
                sys.stdout.write(f"\r{percent:5.2f}% | {speed/1_000_000:6.2f} MH/s | Grid: {blocks}x{self.BLOCK_SIZE}")
                sys.stdout.flush()
        
        print("\nНе найдено.")
        return None