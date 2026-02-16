import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

@dataclass
class Tensor:
    id: int
    width: int
    height: int

@dataclass
class Operation:
    id: int
    type: str  # "MatMul" or "Pointwise"
    input_ids: List[int]
    output_ids: List[int]
    base_cost: float

@dataclass
class ProblemSpec:
    tensors: Dict[int, Tensor]
    ops: Dict[int, Operation]
    fast_mem_cap: int
    bandwidth: int
    native_granularity: Tuple[int, int]  # (w, h)

    @staticmethod
    def from_json(data: Dict) -> 'ProblemSpec':
        tensors = {}
        for i, (w, h) in enumerate(zip(data['widths'], data['heights'])):
            tensors[i] = Tensor(id=i, width=w, height=h)
        
        ops = {}
        for i, cost in enumerate(data['base_costs']):
            ops[i] = Operation(
                id=i,
                type=data['op_types'][i],
                input_ids=data['inputs'][i],
                output_ids=data['outputs'][i],
                base_cost=cost
            )
            
        return ProblemSpec(
            tensors=tensors,
            ops=ops,
            fast_mem_cap=data['fast_memory_capacity'],
            bandwidth=data['slow_memory_bandwidth'],
            native_granularity=tuple(data['native_granularity'])
        )

class HardwareSimulator:
    def __init__(self, problem: ProblemSpec):
        self.p = problem

    def calculate_tile_memory(self, op_id: int, gran: List[int]) -> int:
        """
        Verilen bir Granularity [w, h, k] için o operasyonun
        Fast Memory'de kaplayacağı anlık alanı (Working Set) hesaplar.
        """
        w, h, k = gran
        op = self.p.ops[op_id]
        
        mem_usage = 0
        
        # 1. Output Tensor Boyutu (Her zaman w * h)
        # Not: Yarışma kurallarına göre çıktı tile'ı hafızada yer tutmalı.
        mem_usage += w * h
        
        # 2. Input Tensor Boyutları
        if op.type == "Pointwise":
            # Pointwise işlemde input tile boyutu output ile aynıdır
            # k boyutu yoksayılır (1 kabul edilir)
            for _ in op.input_ids:
                mem_usage += w * h
                
        elif op.type == "MatMul":
            # MatMul Inputs: [LHS, RHS]
            # LHS (Sol Matris): row=h, col=k -> h * k
            # RHS (Sağ Matris): row=k, col=w -> k * w
            if len(op.input_ids) >= 1: # LHS
                mem_usage += h * k
            if len(op.input_ids) >= 2: # RHS
                mem_usage += k * w
                
        return mem_usage

    def validate_step(self, op_ids: List[int], granularity: List[int]) -> Tuple[bool, str]:
        """
        Gemini'nin önerdiği bir adımın (Subgraph) geçerli olup olmadığını kontrol eder.
        """
        # 1. Native Granularity Kontrolü (İsteğe bağlı, padding uyarısı yapılabilir)
        # Şimdilik sadece hard memory limitine bakıyoruz.
        
        total_working_set = 0
        
        # Subgraph içindeki her operasyon için en kötü durum (peak memory) senaryosuna bakmalıyız.
        # Basitleştirme: Subgraph içindeki max tile gereksinimini alıyoruz.
        # (Gerçekte fused op'larda intermediate buffer yoktur, sadece input+output vardır)
        
        # Eğer tek operasyon varsa direkt hesapla
        if len(op_ids) == 1:
            usage = self.calculate_tile_memory(op_ids[0], granularity)
            if usage > self.p.fast_mem_cap:
                return False, f"OOM: Op {op_ids[0]} needs {usage} bytes, cap is {self.p.fast_mem_cap}"
        else:
            # Fused operasyonlar için basitleştirilmiş mantık:
            # İlk Op'un inputları + Son Op'un outputları bellekte olmalı.
            # Ara tensörler (ephemeral) bellekte yer kaplamaz (0 size).
            pass # TODO: Burası gelişmiş fusion mantığı için güncellenecek.
            # Şimdilik "Worst Case" olarak her operasyonu ayrı ayrı sığıyor mu diye kontrol edelim.
            for oid in op_ids:
                usage = self.calculate_tile_memory(oid, granularity)
                if usage > self.p.fast_mem_cap:
                    return False, f"OOM inside fusion: Op {oid} needs {usage} bytes"

        return True, "OK"

    def get_tensor_dims(self, tid: int) -> str:
        t = self.p.tensors[tid]
        return f"{t.width}x{t.height}"