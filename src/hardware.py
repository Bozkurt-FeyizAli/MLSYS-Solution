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
        Final Versiyon: Split-K ve Output Stationary destekli Latency Hesabı.
        """
        w_tile, h_tile, k_tile = granularity
        
        # 1. Döngü Sayılarını Hesapla
        ref_op = self.p.ops[op_ids[0]]
        ref_tensor = self.p.tensors[ref_op.output_ids[0]]
        
        num_tiles_w = math.ceil(ref_tensor.width / w_tile)
        num_tiles_h = math.ceil(ref_tensor.height / h_tile)
        
        # MatMul K (Derinlik) Döngüsü
        num_tiles_k = 1
        if ref_op.type == "MatMul":
            lhs_id = ref_op.input_ids[0]
            k_full_dim = self.p.tensors[lhs_id].width
            num_tiles_k = math.ceil(k_full_dim / k_tile)

        total_spatial_tiles = num_tiles_w * num_tiles_h
        total_loop_tiles = total_spatial_tiles * num_tiles_k # Toplam kaç 'tik' atacak
        
        # 2. MALİYET A: Inner Loop (Her adımda yapılan işler)
        # Bu kısımda Inputs yüklenir ve Compute yapılır. Output buraya dahil edilmez!
        
        # A1. Compute Time (Her adımda)
        total_base_cost = sum(self.p.ops[oid].base_cost for oid in op_ids)
        # Base cost tüm işlemi kapsar, bunu toplam tile sayısına böleriz
        compute_time_per_tile = total_base_cost

        # A2. Input Load Time (Her adımda tekrarlanır)
        memory_load_bytes = 0
        loaded_inputs = set()
        consumed_within_subgraph = set() # Fusion için
        
        for oid in op_ids:
            op = self.p.ops[oid]
            consumed_within_subgraph.update(op.input_ids)
            for idx, inp_id in enumerate(op.input_ids):
                # Residency Kontrolü
                if inp_id in resident_tensors: continue
                # Fusion Kontrolü (Önceki op'un çıktısıysa yükleme)
                if inp_id in [out for prev_op in op_ids for out in self.p.ops[prev_op].output_ids]: continue
                
                # Inputu Yükle
                # Dikkat: Set kullanmıyoruz çünkü Split-K'da inputlar her K diliminde farklıdır!
                # Ancak aynı op içindeki tekrarları önlemek için basit kontrol:
                # (Basitleştirme: Her tile için input kesin yüklenir varsayıyoruz)
                size = self.get_tile_dims(inp_id, granularity, op.type, is_output=False, input_idx=idx)
                memory_load_bytes += size

        input_time_per_tile = memory_load_bytes / self.p.bandwidth

        # Inner Loop Latency (Roofline: Compute vs Memory Load)
        inner_step_latency = max(compute_time_per_tile, input_time_per_tile)
        total_inner_latency = inner_step_latency * total_loop_tiles


        # 3. MALİYET B: Epilog (Sadece en sonda yapılan işler)
        # Output sadece 1 kere yazılır (Accumulation bittikten sonra)
        # Bu yüzden num_tiles_k ile ÇARPILMAZ.
        
        memory_store_bytes = 0
        stored_outputs = set()
        
        for oid in op_ids:
            op = self.p.ops[oid]
            for out_id in op.output_ids:
                if out_id in stored_outputs: continue
                
                # Retain ediliyorsa yazma maliyeti 0
                if out_id in retain_next: continue
                # Ara tensörse (ephemeral) yazma maliyeti 0
                if out_id in consumed_within_subgraph: continue

                # Sadece Spatial Tile sayısı kadar yazılır (K döngüsünden bağımsız)
                size = self.get_tile_dims(out_id, granularity, op.type, is_output=True)
                memory_store_bytes += size
                stored_outputs.add(out_id)

        # Output yazma süresi (Sadece Spatial Tile sayısı kadar)
        output_time_total = (memory_store_bytes * total_spatial_tiles) / self.p.bandwidth

        # 4. TOPLAM SÜRE
        return total_inner_latency + output_time_total

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