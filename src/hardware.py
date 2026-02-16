import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set

@dataclass
class Tensor:
    id: int
    width: int
    height: int

@dataclass
class Operation:
    id: int
    type: str
    input_ids: List[int]
    output_ids: List[int]
    base_cost: float

@dataclass
class ProblemSpec:
    tensors: Dict[int, Tensor]
    ops: Dict[int, Operation]
    fast_mem_cap: int
    bandwidth: int
    native_granularity: Tuple[int, int]

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

    def get_tile_dims(self, tensor_id: int, granularity: List[int], op_type: str, is_output: bool = False, input_idx: int = 0) -> int:
        """Bir tile (parça) için gerekli veri boyutunu hesaplar."""
        w_tile, h_tile, k_tile = granularity
        
        # Eğer bu bir output ise boyut her zaman w * h
        if is_output:
            return w_tile * h_tile

        # Input ise Op tipine göre değişir
        if op_type == "Pointwise":
            return w_tile * h_tile
        
        elif op_type == "MatMul":
            # MatMul Inputs: [LHS (Sol), RHS (Sağ)]
            # LHS: [h, k] -> h_tile * k_tile
            # RHS: [k, w] -> k_tile * w_tile
            if input_idx == 0: # LHS
                return h_tile * k_tile
            else: # RHS
                return k_tile * w_tile
        
        return 0

    def calculate_latency(self, 
                          op_ids: List[int], 
                          granularity: List[int], 
                          resident_tensors: Set[int], 
                          retain_next: List[int]) -> float:
        """
        Bir Subgraph için kesin Latency değerini hesaplar.
        Model: max(Compute, Memory_In + Memory_Out)
        """
        
        w_tile, h_tile, k_tile = granularity
        
        # --- calculate_latency Fonksiyonunun Başı ---
        
        # 1. Toplam Tile Sayısını Bul (Loop Count)
        ref_op = self.p.ops[op_ids[0]]
        
        # Output boyutuna göre W ve H döngüleri
        # (Çıktı tensörünün ID'si output_ids[0] kabul edilir)
        out_tensor_id = ref_op.output_ids[0]
        ref_tensor = self.p.tensors[out_tensor_id]
        
        num_tiles_w = math.ceil(ref_tensor.width / w_tile)
        num_tiles_h = math.ceil(ref_tensor.height / h_tile)
        
        # --- KRİTİK GÜNCELLEME: K (Reduction) Döngüsü ---
        num_tiles_k = 1
        
        if ref_op.type == "MatMul":
            # MatMul için reduction boyutu (K), Input 0'ın genişliğidir (veya Input 1'in yüksekliği).
            # Inputs: [LHS, RHS]. LHS boyutu [Height, K].
            lhs_id = ref_op.input_ids[0]
            k_dim_size = self.p.tensors[lhs_id].width 
            
            # Eğer granülarite [w, h, k] ise, K döngüsü = Tensor_K / k_tile
            num_tiles_k = math.ceil(k_dim_size / k_tile)

        # Toplam Döngü Sayısı
        total_tiles = num_tiles_w * num_tiles_h * num_tiles_k
        
        # ... kodun geri kalanı (compute ve memory hesabı) aynı ...
        
        # 2. Her bir Tile (Parça) için Maliyet Hesabı
        
        # A) Compute Time (İşlem Gücü)
        # Basit Model: Base Cost / Toplam Parça Sayısı
        total_base_cost = sum(self.p.ops[oid].base_cost for oid in op_ids)
        compute_time_per_tile = total_base_cost / total_tiles

        # B) Memory Time (Veri Taşıma)
        memory_load_bytes = 0
        memory_store_bytes = 0
        
        # Girdileri Yükle
        loaded_inputs = set()
        for oid in op_ids:
            op = self.p.ops[oid]
            for idx, inp_id in enumerate(op.input_ids):
                # Eğer input zaten hafızadaysa (resident) yükleme maliyeti 0
                if inp_id in resident_tensors:
                    continue
                # Aynı subgraph içinde bir önceki op'un çıktısıysa (Fusion) maliyet 0
                if inp_id in [out for prev_op in op_ids for out in self.p.ops[prev_op].output_ids]:
                    continue
                
                # Inputu yükle
                if inp_id not in loaded_inputs:
                    size = self.get_tile_dims(inp_id, granularity, op.type, is_output=False, input_idx=idx)
                    memory_load_bytes += size
                    loaded_inputs.add(inp_id)


        # Çıktıları Yaz (YENİ)
        # Kural: Eğer output 'retain_next' listesindeyse (Fast Memory'de kalacaksa),
        # Ana Belleğe (Slow Memory) yazma maliyeti ÖDENMEZ.
        # --- GÜNCELLENMİŞ ÇIKTI YAZMA MANTIĞI ---
        
        # 1. Bu subgraph içinde tüketilen tensörleri bul (Ephemeral Tensors)
        # Eğer bir çıktı, aynı subgraph içindeki başka bir op tarafından girdi olarak kullanılıyorsa,
        # o veri Fast Memory içinde akar, Ana Belleğe gitmesine gerek yoktur.
        consumed_within_subgraph = set()
        for oid in op_ids:
            consumed_within_subgraph.update(self.p.ops[oid].input_ids)

        stored_outputs = set()
        for oid in op_ids:
            op = self.p.ops[oid]
            for out_id in op.output_ids:
                if out_id in stored_outputs:
                    continue
                
                # KURAL 1: Bir sonraki adım için saklanıyorsa (Retain) -> Yazma Maliyeti 0.
                if out_id in retain_next:
                    continue 

                # KURAL 2 (YENİ): Subgraph içinde tüketiliyorsa (Ephemeral) -> Yazma Maliyeti 0.
                # (Not: Eğer bu tensör aynı zamanda grafın en son çıktısıysa yazılmalı ama
                # bu yarışma özelinde ara tensörler genelde çıktı değildir.)
                if out_id in consumed_within_subgraph:
                    continue

                # Diğer durumlarda Ana Belleğe (Slow Memory) yazılır.
                size = self.get_tile_dims(out_id, granularity, op.type, is_output=True)
                memory_store_bytes += size
                stored_outputs.add(out_id)

        # Bandwidth'e böl
        memory_time_per_tile = (memory_load_bytes + memory_store_bytes) / self.p.bandwidth

        # 3. Roofline Modeli: Hangisi yavaşsa onu al
        step_latency = max(compute_time_per_tile, memory_time_per_tile)
        
        # 4. Toplam Süre = Tek Adım * Adım Sayısı
        return step_latency * total_tiles

    def validate_step(self, op_ids: List[int], granularity: List[int]) -> Tuple[bool, str]:
        """Working Set (Anlık Hafıza) Kontrolü"""
        total_mem = 0
        
        # Çok basit worst-case hesabı
        # Subgraph içindeki tüm unique input ve output tile'larını topla
        seen_tensors = set()
        
        for oid in op_ids:
            op = self.p.ops[oid]
            # Inputs
            for idx, inp in enumerate(op.input_ids):
                if inp not in seen_tensors:
                    total_mem += self.get_tile_dims(inp, granularity, op.type, is_output=False, input_idx=idx)
                    seen_tensors.add(inp)
            # Outputs
            for out in op.output_ids:
                if out not in seen_tensors:
                    total_mem += self.get_tile_dims(out, granularity, op.type, is_output=True)
                    seen_tensors.add(out)
                    
        if total_mem > self.p.fast_mem_cap:
            return False, f"OOM: Requires {total_mem}, Cap {self.p.fast_mem_cap}"
            
        return True, "OK"