import unittest

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers.generation.candidate_generator import UniversalSpeculativeDecodingGenerator


device = "cuda" if torch.cuda.is_available() else "cpu"

class TestUniversalSpeculativeDecoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Setup main and assistant models
        cls.main_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct").to(device)
        cls.assistant_model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-gpt2").to(device)
        cls.main_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct")
        cls.assistant_tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-gpt2")
        cls.generation_config = GenerationConfig()

        # Ensure required tokens exist
        if cls.main_tokenizer.pad_token_id is None:
            cls.main_tokenizer.pad_token_id = cls.main_tokenizer.eos_token_id
        if cls.main_tokenizer.bos_token_id is None:
            cls.main_tokenizer.bos_token_id = cls.main_tokenizer.eos_token_id

    def setUp(self):
        self.input_ids = torch.tensor([[1, 2, 3]]).to(device)
        self.model_kwargs = {
            "attention_mask": torch.ones_like(self.input_ids).to(device),
        }
        self.generator = UniversalSpeculativeDecodingGenerator(
            input_ids=self.input_ids,
            assistant_model=self.assistant_model,
            target_tokenizer=self.main_tokenizer,
            assistant_tokenizer=self.assistant_tokenizer,
            generation_config=self.generation_config,
            model_kwargs=self.model_kwargs,
            target_vocab_size=self.main_tokenizer.vocab_size,
        )

    def test_basic_generation(self):
        """Test basic speculative decoding works"""
        input_text = "The quick brown fox"
        input_ids = self.main_tokenizer.encode(input_text, return_tensors="pt")
        self.generator.input_ids = input_ids
        candidates, scores = self.generator.get_candidates(input_ids)

        self.assertIsNotNone(candidates)
        self.assertIsNotNone(scores)
        self.assertTrue(torch.is_tensor(candidates))
        self.assertTrue(torch.is_tensor(scores))

    def test_mismatched_vocabularies(self):
        """Test handling of mismatched vocabularies between models"""
        # Create input with tokens present in main but not assistant vocab
        # Find a token that is not in the assistant tokenizer but in 
        # the main tokenizer.
        missing_token = next(
            token for token in self.main_tokenizer.get_vocab()
            if token not in self.assistant_tokenizer.get_vocab() and
               token not in self.main_tokenizer.all_special_tokens and
               "reserved_" not in token
        )
        input_ids = torch.tensor([[self.main_tokenizer.convert_tokens_to_ids(missing_token)]]) 
        self.generator.input_ids = input_ids
        candidates, scores = self.generator.get_candidates(input_ids)
        self.assertIsNotNone(candidates)

    def test_speculation_depth(self):
        """Test different speculation depths"""
        input_ids = self.main_tokenizer.encode("Test text", return_tensors="pt")
        self.generator.input_ids = input_ids

        for depth in [1, 8, 17]:
            self.generator.num_assistant_tokens = depth
            candidates, scores = self.generator.get_candidates(input_ids)
            self.assertLessEqual(
                candidates.shape[1] - input_ids.shape[1], depth
            )

    def test_device_consistency(self):
        """Test handling of inputs on different devices"""
        if torch.cuda.is_available():
            input_ids = torch.tensor([[1, 2, 3]]).to(
                self.generator.assistant_model.device)
            self.generator.input_ids = input_ids
            candidates, scores = self.generator.get_candidates(input_ids)
            self.assertEqual(candidates.device, input_ids.device)


if __name__ == '__main__':
    unittest.main()