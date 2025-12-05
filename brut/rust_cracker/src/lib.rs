use pyo3::prelude::*;
use rayon::prelude::*;
use argon2::{
    password_hash::{PasswordHash, PasswordVerifier},
    Argon2
};
use std::sync::{Arc, atomic::{AtomicBool, AtomicU64, Ordering}};
use std::thread;
use std::time::{Duration, Instant};
use std::io::{self, Write};

// #[inline(always)] подсказывает компилятору встраивать функцию везде, избегая накладных расходов на вызов
#[inline(always)]
fn fill_pass_buffer(mut idx: u64, length: usize, charset: &[u8], buffer: &mut [u8]) {
    let len_char = charset.len() as u64;
    // Заполняем с конца, так обычно удобнее для понимания (как числа), но порядок не важен для брутфорса
    for i in 0..length {
        // Используем unchecked доступ, если уверены в индексах, но безопасный Rust лучше для универа
        let char_idx = (idx % len_char) as usize;
        buffer[length - 1 - i] = charset[char_idx];
        idx /= len_char;
    }
}

fn spawn_monitor(counter: Arc<AtomicU64>, done: Arc<AtomicBool>, total: u64) {
    thread::spawn(move || {
        let start_time = Instant::now();
        let mut last_count = 0;
        
        while !done.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(1000));
            
            // Если работа завершена, прерываем цикл монитора сразу
            if done.load(Ordering::Relaxed) { break; }

            let current = counter.load(Ordering::Relaxed);
            let speed = current - last_count;
            last_count = current;
            
            let elapsed = start_time.elapsed().as_secs_f64();
            let avg_speed = if elapsed > 0.0 { current as f64 / elapsed } else { 0.0 };
            
            let percent = if total > 0 { (current as f64 / total as f64) * 100.0 } else { 0.0 };

            print!(
                "\r[Rust] {:.2}% | Checked: {}/{} | Speed: {} H/s | Avg: {:.0} H/s", 
                percent, current, total, speed, avg_speed
            );
            io::stdout().flush().unwrap();
        }
    });
}

#[pyfunction]
fn brute_bcrypt(target_hash: String, length: usize, charset: String) -> Option<String> {
    let charset_bytes = charset.as_bytes();
    let charset_len = charset_bytes.len() as u64;
    let total_combinations = charset_len.pow(length as u32);

    let counter = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicBool::new(false));

    spawn_monitor(counter.clone(), done.clone(), total_combinations);

    // ОПТИМИЗАЦИЯ: map_init инициализирует состояние ОДИН РАЗ для каждого потока Rayon.
    // Мы выделяем буфер памяти один раз и переиспользуем его.
    let result = (0..total_combinations)
        .into_par_iter()
        .map_init(
            || vec![0u8; length], // Init: выполняется 1 раз на поток (thread)
            |buffer, i| {         // Map: выполняется на каждой итерации
                // Если флаг завершения поднят другим потоком, пропускаем работу (легкая проверка)
                if done.load(Ordering::Relaxed) {
                    return None;
                }
                
                // Перезаписываем тот же буфер, избегая malloc/free
                fill_pass_buffer(i, length, charset_bytes, buffer);
                
                // bcrypt::verify делает всю тяжелую работу
                let valid = match bcrypt::verify(&buffer, &target_hash) {
                    Ok(v) => v,
                    Err(_) => false,
                };

                // Ordering::Relaxed достаточно для счетчика статистики
                counter.fetch_add(1, Ordering::Relaxed);

                if valid {
                    done.store(true, Ordering::Relaxed); // Сигнализируем всем стоп
                    Some(String::from_utf8(buffer.clone()).unwrap())
                } else {
                    None
                }
            }
        )
        .find_map_any(|res| res); // Находим первый Some(...)

    // Убеждаемся, что монитор остановится
    done.store(true, Ordering::Relaxed);
    println!(); 
    
    result
}

#[pyfunction]
fn brute_argon2(target_hash: String, length: usize, charset: String) -> Option<String> {
    let charset_bytes = charset.as_bytes();
    let charset_len = charset_bytes.len() as u64;
    let total_combinations = charset_len.pow(length as u32);
    
    // Предварительный парсинг хеша (уже было у вас, это правильно)
    let parsed_hash = match PasswordHash::new(&target_hash) {
        Ok(h) => h,
        Err(_) => return None, 
    };
    
    // Argon2 объект создается один раз
    let argon2 = Argon2::default();

    let counter = Arc::new(AtomicU64::new(0));
    let done = Arc::new(AtomicBool::new(false));

    spawn_monitor(counter.clone(), done.clone(), total_combinations);

    let result = (0..total_combinations)
        .into_par_iter()
        .map_init(
            || vec![0u8; length], 
            |buffer, i| {
                if done.load(Ordering::Relaxed) {
                    return None;
                }

                fill_pass_buffer(i, length, charset_bytes, buffer);
                
                let valid = match argon2.verify_password(buffer, &parsed_hash) {
                    Ok(_) => true,
                    Err(_) => false,
                };

                counter.fetch_add(1, Ordering::Relaxed);

                if valid {
                    done.store(true, Ordering::Relaxed);
                    Some(String::from_utf8(buffer.clone()).unwrap())
                } else {
                    None
                }
            }
        )
        .find_map_any(|res| res);

    done.store(true, Ordering::Relaxed);
    println!();

    result
}

#[pymodule]
fn rust_cracker(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(brute_bcrypt, m)?)?;
    m.add_function(wrap_pyfunction!(brute_argon2, m)?)?;
    Ok(())
}