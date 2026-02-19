import os
import sys
import json
import re  # Regex ekledik (Temizleme için)
import google.generativeai as genai
from src.hardware import ProblemSpec, HardwareSimulator

# API Key Kontrolü
API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    # Güvenlik için buraya hardcode yapma, terminalden export et.
    print("HATA: GOOGLE_API_KEY environment variable bulunamadı!")
    sys.exit(1)

genai.configure(api_key=API_KEY)

# --- JSON TEMİZLEYİCİ ---
def extract_json(text):
    """LLM çıktısındaki Markdown bloklarını ve gereksiz metinleri temizler."""
    text = text.strip()
    # ```json ... ``` bloklarını bul
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)
    # Eğer blok yoksa, sadece süslü parantez arasını bulmaya çalış
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text

SYSTEM_PROMPT = """
You are an elite AI compiler engineer.
Your ABSOLUTE GOAL is to MINIMIZE LATENCY. You will fail if you do not use FUSION.

### STRICT RULES (Follow in Order)

1.  **AGGRESSIVE FUSION (NO SINGLE OPS):**
    * NEVER output single-operation subgraphs like `[[0], [1], [2]]`. This is a failure.
    * You MUST group operations into large chunks (e.g., `[[0, 1, 2, 3], [4, 5, 6]]`).
    * Follow the topology: Group producers and consumers together.

2.  **TILING OVER SEPARATION (OOM Handling):**
    * If a large fused chunk causes Out of Memory (OOM), DO NOT break the chunk apart.
    * INSTEAD, aggressively reduce the granularity (Tiling).
    * Try `[64, 64, 64]`, `[32, 32, 32]`, or `[16, 16, 16]`.

3.  **DATA RESIDENCY:**
    * Use `tensors_to_retain` to keep intermediate tensors in Fast Memory between steps.

### REQUIRED EXACT OUTPUT FORMAT
You MUST return a single JSON object with EXACTLY these three keys at the root level. Do NOT wrap them in a "schedule" object. Do NOT use custom keys like "fused_ops".

Example:
{
  "subgraphs": [[0, 1, 2, 3], [4, 5]], 
  "granularities": [[64, 64, 64], [128, 128, 32]], 
  "tensors_to_retain": [[3], []] 
}
"""

def auto_optimize_residency(schedule, problem_spec, simulator):
    """
    LLM'in unuttuğu 'tensors_to_retain' listesini otomatik doldurur.
    Eğer bir tensor, Adım N'de üretilip Adım N+1'de kullanılıyorsa ve belleğe sığıyorsa, onu SAKLAR.
    """
    subgraphs = schedule['subgraphs']
    granularities = schedule['granularities']
    
    # Yeni retain listesi (Boş başlat)
    new_retains = [[] for _ in range(len(subgraphs))]
    
    for i in range(len(subgraphs) - 1):
        # Şu anki adımın ürettiği çıktılar
        current_ops = subgraphs[i]
        current_outputs = set()
        for oid in current_ops:
            current_outputs.update(problem_spec.ops[oid].output_ids)
            
        # Bir sonraki adımın ihtiyaç duyduğu girdiler
        next_ops = subgraphs[i+1]
        next_inputs = set()
        for oid in next_ops:
            next_inputs.update(problem_spec.ops[oid].input_ids)
            
        # Kesişim: Hem üretilen hem de sonraki adımda lazım olanlar
        candidates = current_outputs.intersection(next_inputs)
        
        # Hafıza Kontrolü: Saklarsak sığar mı?
        # Basitlik için: Eğer aday varsa direkt ekleyelim, HardwareSimulator zaten OOM kontrolü yapacak.
        if candidates:
            print(f"⚡ Auto-Optimization: Keeping Tensor {list(candidates)} resident between Step {i} -> {i+1}")
            new_retains[i] = list(candidates)
            
    return new_retains

def generate_schedule_with_retry(problem_path: str, output_path: str):
    print(f"Loading problem from: {problem_path}")
    with open(problem_path, 'r') as f:
        raw_data = json.load(f)
    
    spec = ProblemSpec.from_json(raw_data)
    sim = HardwareSimulator(spec)

    tensor_info = "\n".join([f"Tensor {k}: {v.width}x{v.height}" for k, v in spec.tensors.items()])
    op_info = "\n".join([
        f"Op {k}: {v.type}, Inputs:{v.input_ids}, Out:{v.output_ids}, Cost:{v.base_cost}" 
        for k, v in spec.ops.items()
    ])

    user_prompt = f"""
    PROBLEM SPEC:
    Fast Memory: {spec.fast_mem_cap}
    Bandwidth: {spec.bandwidth}
    
    TENSORS:
    {tensor_info}
    OPERATIONS:
    {op_info}

    Task: Generate the JSON schedule.
    """

    # Model İsmini Listene Göre Ayarladık
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash", # Senin listendeki model
        system_instruction=SYSTEM_PROMPT,
        generation_config={
            "response_mime_type": "application/json", 
            "temperature": 0.4
        }
    )

    chat = model.start_chat(history=[])
    current_prompt = user_prompt
    
    max_retries = 3
    for attempt in range(max_retries):
        print(f"\n--- Attempt {attempt + 1}/{max_retries} ---")
        try:
            response = chat.send_message(current_prompt)
            
            # --- DEBUG: Model ne cevap verdi görelim ---
            # Eğer hata alırsan bu çıktıyı bana at
            print(f"DEBUG (Raw Response): {response.text[:100]}...") 

            # Temizleme ve Parse
            clean_text = extract_json(response.text)
            schedule = json.loads(clean_text)

            print("🔍 Optimizing Residency...")
            optimized_retains = auto_optimize_residency(schedule, spec, sim)
            schedule['tensors_to_retain'] = optimized_retains
            
            # --- VALIDASYON ---
            is_valid = True
            error_msg = ""
            calculated_latencies = []
            resident_tensors = set() 

            if "subgraphs" not in schedule or "granularities" not in schedule:
                # Model bazen anahtarları yanlış isimlendiriyor, yakalayalım
                print(f"DEBUG (Keys Found): {schedule.keys()}")
                raise ValueError("JSON missing keys 'subgraphs' or 'granularities'")

            for i, (subgraph, gran) in enumerate(zip(schedule['subgraphs'], schedule['granularities'])):
                ok, msg = sim.validate_step(subgraph, gran)
                if not ok:
                    is_valid = False
                    error_msg = f"Step {i} Failed (OOM): {msg}"
                    break
                
                retain_list = []
                if "tensors_to_retain" in schedule and i < len(schedule["tensors_to_retain"]):
                    retain_list = schedule["tensors_to_retain"][i]
                
                latency = sim.calculate_latency(subgraph, gran, resident_tensors, retain_list)
                calculated_latencies.append(latency)
                resident_tensors = set(retain_list)

            if is_valid:
                print("✅ Valid schedule found!")
                schedule['subgraph_latencies'] = calculated_latencies
                total_latency = sum(calculated_latencies)
                
                with open(output_path, 'w') as f:
                    json.dump(schedule, f, indent=2)
                
                print(f"🎉 Success! Output saved to {output_path}")
                print(f"🚀 Total Latency: {total_latency}")
                return
            else:
                print(f"❌ Invalid schedule: {error_msg}")
                current_prompt = f"Previous solution INVALID: {error_msg}. Fix granularity/fusion and return JSON."

        except json.JSONDecodeError:
            print("❌ Invalid JSON structure.")
            current_prompt = "Return ONLY raw JSON, no markdown."
        except Exception as e:
            print(f"❌ Error: {e}")
            current_prompt = f"Error: {e}. Regenerate JSON."

    print("💀 Failed after retries.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python agent.py <input.json> <output.json>")
    else:
        generate_schedule_with_retry(sys.argv[1], sys.argv[2])