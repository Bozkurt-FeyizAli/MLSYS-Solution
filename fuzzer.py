import json
import random

def generate_random_problem(filename, num_ops=10):
    op_types_pool = ["MatMul", "Pointwise"]
    dims_pool = [128, 256, 512, 1024, 2048]
    
    tensors_count = 0
    inputs = []
    outputs = []
    op_types = []
    base_costs = []
    
    # Başlangıç tensörü
    available_tensors = [tensors_count]
    tensors_count += 1
    
    for i in range(num_ops):
        op_type = random.choice(op_types_pool)
        op_types.append(op_type)
        base_costs.append(random.choice([200, 500, 1000, 2000]))
        
        # Girdi seçimi (Döngüyü önlemek için sadece önceki tensörlerden)
        num_inputs = 2 if op_type == "MatMul" else 1
        current_inputs = random.sample(available_tensors, min(num_inputs, len(available_tensors)))
        
        # MatMul için her zaman 2 girdi sağla
        while len(current_inputs) < num_inputs:
            current_inputs.append(random.choice(available_tensors))
            
        inputs.append(current_inputs)
        
        # Yeni çıktı tensörü üret ve havuza ekle
        outputs.append([tensors_count])
        available_tensors.append(tensors_count)
        tensors_count += 1

    # Tensör boyutlarını ata
    widths = [random.choice(dims_pool) for _ in range(tensors_count)]
    heights = [random.choice(dims_pool) for _ in range(tensors_count)]

    problem = {
        "widths": widths,
        "heights": heights,
        "inputs": inputs,
        "outputs": outputs,
        "base_costs": base_costs,
        "op_types": op_types,
        "fast_memory_capacity": random.choice([50000, 100000, 500000]),
        "slow_memory_bandwidth": random.choice([20, 50, 100]),
        "native_granularity": [128, 128]
    }

    with open(filename, 'w') as f:
        json.dump(problem, f, indent=2)
    print(f"Başarıyla üretildi: {filename} | Toplam İşlem: {num_ops}")

if __name__ == "__main__":
    # Zorluğu artırmak için num_ops değerini değiştirebilirsin
    generate_random_problem("random_test.json", num_ops=20)