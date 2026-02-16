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
You are a compiler optimization agent for a Tensor Processing Unit (TPU).
Your goal: Generate a valid execution schedule JSON.

RULES:
1. 'granularities' [w, h, k] define the tile size. Smaller tiles use less memory but increase loop overhead.
2. Memory Constraint: (Tile Inputs + Tile Output) MUST be <= Fast Memory Capacity.
3. If a tensor is large, you MUST decompose it using smaller granularity.

OUTPUT FORMAT:
Return ONLY raw JSON matching this structure:
{
  "subgraphs": [[0], [1]], 
  "granularities": [[64, 64, 128], [128, 128, 1]],
  "tensors_to_retain": [[], []],
  "traversal_orders": [null, null],
  "subgraph_latencies": [100.0, 200.0]
}
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
            
            # --- VALIDASYON (Python Logic) ---
            is_valid = True
            error_msg = ""
            
            # Adım adım kontrol et
            if "subgraphs" not in schedule or "granularities" not in schedule:
                raise ValueError("JSON missing keys")

            for i, (subgraph, gran) in enumerate(zip(schedule['subgraphs'], schedule['granularities'])):
                ok, msg = sim.validate_step(subgraph, gran)
                if not ok:
                    is_valid = False
                    error_msg = f"Step {i} Failed: {msg}"
                    break
            
            if is_valid:
                print("Valid schedule found!")
                with open(output_path, 'w') as f:
                    json.dump(schedule, f, indent=2)
                return
            else:
                print(f"Invalid schedule: {error_msg}")
                # Hatayı LLM'e geri besle
                current_prompt = f"Your previous solution was invalid. Error: {error_msg}. Please fix the granularity and try again."
                
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