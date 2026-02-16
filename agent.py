import os
import sys
import json
import google.generativeai as genai
from src.hardware import ProblemSpec, HardwareSimulator

# API Key Kontrolü
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    print("UYARI: GOOGLE_API_KEY environment variable olarak bulunamadı.")
    API_KEY = "AIzaSyC01q02r-YmLRlnPc_pY_J_HojFO2PgDL8" 

genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = """
You are an expert AI accelerator scheduler.
Your goal is to MINIMIZE LATENCY by maximizing DATA REUSE.

CRITICAL STRATEGIES:
1. **KERNEL FUSION (Most Important):** - Always try to group connected operations (e.g., Op0 -> Op1) into a single subgraph `[0, 1]`.
   - This eliminates the need to write Op0's output to Slow Memory and read it back for Op1.
   - Intermediate data between fused ops becomes "Ephemeral" (Costs 0 time, 0 memory).

2. **TILING (Granularity):**
   - Choose the LARGEST granularity [w, h, k] that fits in Fast Memory.
   - Formula: (Input_Tile_Size + Output_Tile_Size) <= Fast_Memory_Capacity.
   - If it fits, use [128, 128]. If not, try [128, 64], then [64, 64].

3. **RESIDENCY:**
   - Use 'tensors_to_retain' to keep data in Fast Memory between subgraphs if it is used immediately in the next step.

OUTPUT FORMAT:
Return valid JSON only.
"""

def generate_schedule_with_retry(problem_path: str, output_path: str):
    # 1. Problemi Yükle
    with open(problem_path, 'r') as f:
        raw_data = json.load(f)
    
    spec = ProblemSpec.from_json(raw_data)
    sim = HardwareSimulator(spec)

    # 2. LLM için Context Oluştur
    tensor_info = "\n".join([f"Tensor {k}: {v.width}x{v.height}" for k, v in spec.tensors.items()])
    op_info = "\n".join([
        f"Op {k}: {v.type}, Inputs:{v.input_ids}, Out:{v.output_ids}, Cost:{v.base_cost}" 
        for k, v in spec.ops.items()
    ])

    user_prompt = f"""
    PROBLEM SPEC:
    Fast Memory: {spec.fast_mem_cap}
    Bandwidth: {spec.bandwidth}
    Native Granularity: {spec.native_granularity}

    TENSORS:
    {tensor_info}

    OPERATIONS:
    {op_info}

    Task: Create a schedule. CAUTION: Check if tensors fit in memory. 
    If a tensor is larger than {spec.fast_mem_cap}, use tiling (smaller granularity).
    """

    # 3. Model Başlatma
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json"}
    )

    # 4. Döngü (Retry Logic)
    max_retries = 3
    chat = model.start_chat(history=[])
    
    current_prompt = user_prompt

    for attempt in range(max_retries):
        print(f"--- Attempt {attempt + 1} ---")
        response = chat.send_message(current_prompt)
        
        try:
            schedule = json.loads(response.text)
            
            # --- VALIDASYON VE HESAPLAMA ---
            is_valid = True
            error_msg = ""
            calculated_latencies = []
            resident_tensors = set() # Başlangıçta Fast Memory boş

            if "subgraphs" not in schedule or "granularities" not in schedule:
                raise ValueError("JSON missing keys")

            # Adım adım simülasyon
            for i, (subgraph, gran) in enumerate(zip(schedule['subgraphs'], schedule['granularities'])):
                # 1. Hafıza Kontrolü
                ok, msg = sim.validate_step(subgraph, gran)
                if not ok:
                    is_valid = False
                    error_msg = f"Step {i} Failed: {msg}"
                    break
                
                # 2. Retain (Saklanacaklar) Listesini Al
                # Eğer LLM retain listesi vermediyse boş kabul et
                retain_list = []
                if "tensors_to_retain" in schedule and i < len(schedule["tensors_to_retain"]):
                    retain_list = schedule["tensors_to_retain"][i]
                
                # 3. Latency Hesapla (Python Yapıyor!)
                latency = sim.calculate_latency(subgraph, gran, resident_tensors, retain_list)
                calculated_latencies.append(latency)
                
                # 4. Hafıza Durumunu Güncelle (Bir sonraki adım için)
                # Yeni resident seti = retain edilenler
                resident_tensors = set(retain_list)

            if is_valid:
                print("Valid schedule found! Overwriting latencies with calculated values.")
                
                # LLM'in uydurduğu sayıları sil, gerçek hesaplananları yaz
                schedule['subgraph_latencies'] = calculated_latencies
                
                with open(output_path, 'w') as f:
                    json.dump(schedule, f, indent=2)
                
                print(f"Total Latency: {sum(calculated_latencies)}")
                return
            else:
                print(f"Invalid schedule: {error_msg}")
                current_prompt = f"Your previous solution was invalid. Error: {error_msg}. Please fix tiling."

        except json.JSONDecodeError:
            print("Invalid JSON received.")
            current_prompt = "You returned invalid JSON. Please return ONLY valid JSON."
        except Exception as e:
            print(f"Error: {e}")
            current_prompt = f"An error occurred: {e}. Fix the format."

    print("Failed to find solution after retries.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python agent.py <input.json> <output.json>")
    else:
        generate_schedule_with_retry(sys.argv[1], sys.argv[2])