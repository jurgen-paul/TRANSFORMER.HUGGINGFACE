from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, TypeVar

import torch


T = TypeVar("T")


class Cache(ABC):
    def __init__(self) -> None:
        self.key_cache: Dict[int, Tuple[torch.Tensor]] = {}
        self.value_cache: Dict[int, Tuple[torch.Tensor]] = {}

    @abstractmethod
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self.key_cache:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        return (
            tuple(self.key_cache[layer_idx] for layer_idx in range(len(self.key_cache))),
            tuple(self.value_cache[layer_idx] for layer_idx in range(len(self.value_cache))),
        )

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[List[torch.FloatTensor]]) -> "DynamicCache":
        if past_key_values is None:
            return cls()
        cache = cls()
        for layer_idx, (key_states, value_states) in enumerate(zip(*past_key_values)):
            cache.update(key_states, value_states, layer_idx)
        return cache


class DynamicCache(Cache):
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self.key_cache:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class PagedAttentionCache(Cache):
    def __init__(self, num_blocks: int = 8, block_size: int = 16) -> None:
        super().__init__()
        # cache config & kv_cache
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.key_cache: Dict[int, Tuple[torch.Tensor]] = {}  # layer_idx -> key_cache
        self.value_cache: Dict[int, Tuple[torch.Tensor]] = {}  # layer_idx -> value_cache
        self.cache_initialized = False

        # cache runtime management information
        self.free_blocks = list(range(num_blocks))  # free blocks
        self.block_ref_count = [0] * self.num_blocks  # init the reference count for each physical block
        self.block_tables = {}  # mapping logical block to physical blocks for each sequence
        self.context_lens = {}  # context length for each sequence

        # The follow two states are shared accross layer but only for the current decode step. Need to update for every decode step.
        self.batch2seq = {}  # mapping batch index to {seq_id0, seq_id1, ...} to enable prompt sharing.
        self.slots_mapping = []  # mapping logical slots to physical slots.
    
    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[List[torch.FloatTensor]]) -> "PageAttentionCache":
        if past_key_values is None:
            return cls()
        cache = cls()
        for layer_idx, (key_states, value_states) in enumerate(zip(*past_key_values)):
            cache.update(key_states, value_states, layer_idx)
        return cache
    
    def copy_on_write(self, src_block_idx: int, dst_block_idx: int):
        """
        Copy the content of src_block_idx to dst_block_idx.

        Args:
            src_block_idx (int): The index of the source block.
            dst_block_idx (int): The index of the destination block.
        """
        for layer_idx in range(len(self.key_cache)):
            self.key_cache[layer_idx][dst_block_idx] = self.key_cache[layer_idx][src_block_idx].clone()
            self.value_cache[layer_idx][dst_block_idx] = self.value_cache[layer_idx][src_block_idx].clone()
        
    def allocate(self, seq_idx: int, key_len: int, context_len: int) -> List[int]:
        """
        Allocate physical slots for a given sequence index, key length and context length.

        Args:
        - seq_idx (int): The index of the sequence.
        - key_len (int): The length of the key.
        - context_len (int): The length of the context.

        Returns:
        - slots (list): The physical slots for the given sequence, where 1 slot is allocated for 1 token state.
        """
        slots = []
        if seq_idx not in self.block_tables:
            # allocate blocks for this sequence
            assert context_len == 0
            needed_blocks = (key_len + self.block_size - 1) // self.block_size
            assert needed_blocks <= len(self.free_blocks)
            blocks = self.free_blocks[:needed_blocks]
            self.free_blocks = self.free_blocks[needed_blocks:]
            self.block_tables[seq_idx] = blocks
            for block_idx in blocks:
                self.block_ref_count[block_idx] += 1
        else:
            # find free slots in the allocated blocks or allocate new blocks
            seq_len = key_len + context_len
            target_blocks = (seq_len + self.block_size - 1) // self.block_size
            new_blocks = target_blocks - len(self.block_tables[seq_idx])
            assert new_blocks <= len(self.free_blocks)
            if new_blocks > 0:  # allocate new blocks
                candidate_blocks = self.free_blocks[:new_blocks]
                self.block_tables[seq_idx].extend(self.free_blocks[:new_blocks])
                self.free_blocks = self.free_blocks[new_blocks:]
                for block_idx in candidate_blocks:
                    self.block_ref_count[block_idx] += 1
            else:
                last_block = self.block_tables[seq_idx][-1]
                #sharing the last block with other sequences, need to allocate a new block and copy the last block
                if self.block_ref_count[last_block] > 1:
                    assert len(self.free_blocks) > 0
                    new_block = self.free_blocks.pop()
                    self.block_tables[seq_idx][-1] = new_block
                    self.block_ref_count[new_block] += 1
                    self.block_ref_count[last_block] -= 1
                    self.copy_on_write(last_block, new_block)
        # return the slots for this sequence
        for i in range(key_len):
            token_id = i + context_len
            block_idx = token_id // self.block_size
            block_offset = token_id % self.block_size
            slots.append(self.block_tables[seq_idx][block_idx] * self.block_size + block_offset)
        return slots

    def free(self, seq_idx: int):
        """
        Frees the blocks allocated for the given sequence index.

        Args:
            seq_idx (int): The index of the sequence whose blocks are to be freed.

        Raises:
            AssertionError: If the given sequence index is not present in the block tables.
        """
        assert seq_idx in self.block_tables
        for block_idx in self.block_tables[seq_idx]:
            self.block_ref_count[block_idx] -= 1
            if self.block_ref_count[block_idx] == 0:
                self.free_blocks.append(block_idx)

    def fork(self, seq_idx: int, new_seq_idx: int):
        """
        Forks the blocks allocated for seq_idx to new_seq_idx.

        Args:
            seq_idx (int): The index of the sequence to be forked.
            new_seq_idx (int): The index of the new sequence.

        Raises:
            AssertionError: If seq_idx is not in block_tables or if new_seq_idx is already in block_tables.
        """
        assert seq_idx in self.block_tables
        assert new_seq_idx not in self.block_tables
        self.block_tables[new_seq_idx] = self.block_tables[seq_idx]
        for block_idx in self.block_tables[seq_idx]:
            self.block_ref_count[block_idx] += 1

    def set_batch2seq(self, batch2seq: Dict[int, int]):
        self.batch2seq = batch2seq

    def reshape_and_cache(
        self, slot_mapping: List[List[int]], key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ):
        """
        Reshapes and caches the key and value states based on the given slot mapping.

        Args:
            slot_mapping (List[List[int]]): A list of lists representing the slot mapping.
            key_states (torch.Tensor): The key states tensor.
            value_states (torch.Tensor): The value states tensor.
            layer_idx (int): The index of the layer.

        Returns:
            None
        """
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.long, device="cpu")
        block_indicies = torch.div(slot_mapping, self.block_size, rounding_mode="floor")
        block_indicies = block_indicies.cpu().tolist()
        block_offsets = slot_mapping % self.block_size
        block_offsets = block_offsets.cpu().tolist()
        batch = len(slot_mapping)
        assert batch == key_states.shape[0]
        seq_len = key_states.shape[-2]
        key = key_states.transpose(1, 2)
        value = value_states.transpose(1, 2)
        for bi in range(batch):
            for ti in range(seq_len):
                block_idx = block_indicies[bi][ti]
                block_offset = block_offsets[bi][ti]
                self.key_cache[layer_idx][block_idx][block_offset] = key[bi][ti]
                self.value_cache[layer_idx][block_idx][block_offset] = value[bi][ti]

    def is_last_layer(self, layer_idx: int) -> bool:
        return layer_idx + 1 == len(self.key_cache) and self.cache_initialized
    
    def has_context(self, layer_idx: int, seq_id: int) -> bool:
        return seq_id in self.context_lens and layer_idx < len(self.context_lens[seq_id]) and self.context_lens[seq_id][layer_idx] != 0
      
    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self.key_cache:
            return 0
        return self.context_lens[0][layer_idx] #current assume that padding batch to same length

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with key and value states for a specific layer.

        Args:
            key_states (torch.Tensor): The key states tensor of shape [batch, head, seq, dim].
            value_states (torch.Tensor): The value states tensor of shape [batch, head, seq, dim].
            layer_idx (int): The index of the layer.
            cos (Optional[torch.Tensor]): Optional tensor of shape [batch, head, seq, dim] representing cosine values.
            sin (Optional[torch.Tensor]): Optional tensor of shape [batch, head, seq, dim] representing sine values.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the updated key states and value states tensors.

        Raises:
            AssertionError: If the batch size is inconsistent with the existing cache.
        """
        batch_size = key_states.shape[0]  # [batch, head, seq, dim]
        kv_head = key_states.shape[1]
        head_size = key_states.shape[-1]
        original_key_states = key_states       
        # self.batch2seq is only for the current decode step, need to clear in the last layer and init in the first layer or setup externally
        if layer_idx == 0 and self.batch2seq == {}:
            assert len(self.block_tables) == 0
            self.batch2seq = {i: [i] for i in range(batch_size)}
            self.slots_mapping = []
        elif layer_idx == 0 and self.batch2seq != {}:
            assert len(self.batch2seq) == batch_size        

        if layer_idx not in self.key_cache:  # init the cache
            self.key_cache[layer_idx] = torch.zeros(
                (self.num_blocks, self.block_size, kv_head, head_size),
                dtype=key_states.dtype,
                device=key_states.device,
            )
            self.value_cache[layer_idx] = torch.zeros(
                (self.num_blocks, self.block_size, kv_head, head_size),
                dtype=value_states.dtype,
                device=value_states.device,
            )
            self.cache_initialized = False
        else:
            self.cache_initialized = True
        # step 1): allocate slots to store token states for each sequence in the batch, only need run in the first layer
        if layer_idx == 0:
            self.slots_mapping = []
            # only allocate the slots for the first sequence in the batch to enable prompt sharing
            for batch_idx in range(batch_size):
                seq_id = self.batch2seq[batch_idx][0]
                key_len = key_states[batch_idx].shape[-2]                 
                past_context_len = self.context_lens[seq_id][layer_idx] if self.has_context(layer_idx, seq_id) else 0
                slots = self.allocate(seq_id, key_len, past_context_len)
                self.slots_mapping.append(slots)
        assert len(self.slots_mapping) == batch_size

        # step 2): cache key_states & value states
        self.reshape_and_cache(self.slots_mapping, key_states, value_states, layer_idx)

        # step 3): fork new sequences to enable prompt sharing, only need run in the first layer
        if layer_idx == 0:
            for batch_idx in range(batch_size):
                seq_ids = self.batch2seq[batch_idx]
                # fork the blocks allocated for the first sequence to other sequences in the batch
                for seq_id in seq_ids[1:]:
                    self.fork(seq_ids[0], seq_id)                 
                          
        context_len =  self.context_lens[0][layer_idx] if self.has_context(layer_idx, 0) else 0        
        # step 4): update the key_states & value_states for each sequence in the batch
        if context_len != 0:
            context_len = context_len + key_states.shape[-2]
            key = torch.zeros((batch_size, context_len, kv_head, head_size), dtype=key_states.dtype, device=key_states.device)
            value = torch.zeros((batch_size, context_len, kv_head, head_size), dtype=value_states.dtype, device=value_states.device)
            for batch_idx in range(batch_size):
                for i in range(context_len):
                    block_idx = i // self.block_size
                    block_offset = i % self.block_size
                    key[batch_idx][i] = self.key_cache[layer_idx][block_idx][block_offset]
                    value[batch_idx][i] = self.value_cache[layer_idx][block_idx][block_offset]
            key_states = key.transpose(1,2).contiguous()
            value_states = value.transpose(1,2).contiguous()   
        
        #update the context length for each sequence in the batch
        for batch_idx in range(batch_size):
            seq_ids = self.batch2seq[batch_idx]
            # fork the blocks allocated for the first sequence to other sequences in the batch
            for seq_id in seq_ids:
                key_len = original_key_states[batch_idx].shape[-2]
                if seq_id not in self.context_lens:
                    self.context_lens[seq_id] = []
                if layer_idx == len(self.context_lens[seq_id]):
                    self.context_lens[seq_id].append(0)
                self.context_lens[seq_id][layer_idx] += key_len
        return key_states, value_states

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """
        Reorder the cache according to the beam index. The beam index is a tensor of shape (batch_size,)
        and the sequence id can be get from the self.batch2seq.
        """
        print("reorder_cache")
        print("beam_idx", beam_idx)
        print("self.block_tables", self.block_tables)
        print("self.block_ref_count", self.block_ref_count)
        print("self.free_blocks", self.free_blocks)
        freed_seqs = []
        new_block_tables = self.block_tables.copy()
        for batch_idx, target_batch_idx in enumerate(beam_idx.tolist()):
            target_seq_id = self.batch2seq[target_batch_idx][0]
            seq_id = self.batch2seq[batch_idx][0]
            freed_seqs.append(seq_id)
            new_block_tables[seq_id] = []
            for block in self.block_tables[target_seq_id]:
                self.block_ref_count[block] += 1
                new_block_tables[seq_id].append(block)
        for seq_idx in freed_seqs:
            self.free(seq_idx)
        self.block_tables = new_block_tables
        print("self.free_blocks", self.free_blocks)
        print("self.block_tables", self.block_tables)
        print("self.block_ref_count", self.block_ref_count)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_single(
    key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: Optional[torch.IntTensor] = None
) -> torch.Tensor:
    if position_ids:
        cos = cos[position_ids].unsqueeze(1)  # [seq_len, dim] -> [batch_size, 1, seq_len, head_dim]
        sin = sin[position_ids].unsqueeze(1)
    rotated_key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    return rotated_key_states


class SinkCache(Cache):
    def __init__(self, window_length: int, num_sink_tokens: int) -> None:
        super().__init__()
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self.cos_sin_cache = {}

    def get_rerotation_cos_sin(
        self, key_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if key_states.shape[-2] not in self.cos_sin_cache:
            # Upcast to float32 temporarily for better accuracy
            cos = cos.to(torch.float32)
            sin = sin.to(torch.float32)

            # Compute the cos and sin required for back- and forward-rotating to one position earlier in the sequence
            original_cos = cos[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_cos = cos[self.num_sink_tokens : -key_states.shape[-2]]
            original_sin = sin[self.num_sink_tokens + key_states.shape[-2] :]
            shifted_sin = sin[self.num_sink_tokens : -key_states.shape[-2]]
            rerotation_cos = original_cos * shifted_cos + original_sin * shifted_sin
            rerotation_sin = -original_sin * shifted_cos + original_cos * shifted_sin

            self.cos_sin_cache[key_states.shape[-2]] = (
                rerotation_cos.to(key_states.dtype).unsqueeze(0),
                rerotation_sin.to(key_states.dtype).unsqueeze(0),
            )
        return self.cos_sin_cache[key_states.shape[-2]]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        return min(super().get_seq_length(layer_idx), self.window_length - 1)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # [bsz, num_heads, seq_len, head_dim]
        if layer_idx not in self.key_cache:
            # Empty cache
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states

        elif key_states.shape[-2] + self.get_seq_length(layer_idx) < self.window_length:
            # Growing cache
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        else:
            # Shifting cache
            rotated_keys = self.key_cache[layer_idx][
                :, :, -self.window_length + self.num_sink_tokens + key_states.shape[-2] :
            ]
            rerotation_cos, rerotation_sin = self.get_rerotation_cos_sin(key_states, cos, sin)
            rerotated_keys = apply_rotary_pos_emb_single(rotated_keys, rerotation_cos, rerotation_sin)

            # Concatenate sink tokens, shifted & rotated tokens, and new tokens
            self.key_cache[layer_idx] = torch.cat(
                [self.key_cache[layer_idx][:, :, : self.num_sink_tokens], rerotated_keys, key_states], dim=-2
            )
            self.value_cache[layer_idx] = torch.cat(
                [
                    self.value_cache[layer_idx][:, :, : self.num_sink_tokens],
                    self.value_cache[layer_idx][
                        :, :, -self.window_length + self.num_sink_tokens + value_states.shape[-2] :
                    ],
                    value_states,
                ],
                dim=-2,
            )
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
