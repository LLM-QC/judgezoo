"""Tests for the utility functions."""

import os
import sys
from unittest.mock import patch

import pytest
import torch

# Add the parent directory to the path to import judgezoo
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judgezoo._utils import free_vram, with_max_batchsize


class TestWithMaxBatchsize:
    """Test the with_max_batchsize utility function."""

    def test_basic_functionality(self):
        """Test basic functionality with a simple function."""
        def simple_function(inputs):
            return [x * 2 for x in inputs]

        inputs = [1, 2, 3, 4]
        result = with_max_batchsize(simple_function, inputs, initial_batch_size=2)

        assert result == [2, 4, 6, 8]

    def test_tensor_concatenation(self):
        """Test tensor concatenation functionality."""
        def tensor_function(inputs):
            return torch.tensor([[x] for x in inputs])

        inputs = [1, 2, 3, 4]
        result = with_max_batchsize(tensor_function, inputs, initial_batch_size=2)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 1)
        assert torch.equal(result, torch.tensor([[1], [2], [3], [4]]))

    def test_dict_concatenation(self):
        """Test dictionary concatenation functionality."""
        def dict_function(inputs):
            return {"values": inputs, "doubled": [x * 2 for x in inputs]}

        inputs = [1, 2, 3, 4]
        result = with_max_batchsize(dict_function, inputs, initial_batch_size=2)

        assert isinstance(result, dict)
        assert "values" in result
        assert "doubled" in result
        assert result["values"] == [1, 2, 3, 4]
        assert result["doubled"] == [2, 4, 6, 8]

    def test_tuple_concatenation(self):
        """Test tuple concatenation functionality."""
        def tuple_function(inputs):
            tensors = torch.tensor([[x] for x in inputs])
            lists = [x * 2 for x in inputs]
            return (tensors, lists)

        inputs = [1, 2, 3, 4]
        result = with_max_batchsize(tuple_function, inputs, initial_batch_size=2)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], list)
        assert torch.equal(result[0], torch.tensor([[1], [2], [3], [4]]))
        assert result[1] == [2, 4, 6, 8]

    def test_multiple_inputs(self):
        """Test with multiple input arguments."""
        def multi_input_function(inputs1, inputs2):
            return [x + y for x, y in zip(inputs1, inputs2)]

        inputs1 = [1, 2, 3, 4]
        inputs2 = [10, 20, 30, 40]
        result = with_max_batchsize(multi_input_function, inputs1, inputs2, initial_batch_size=2)

        assert result == [11, 22, 33, 44]

    def test_mismatched_input_lengths(self):
        """Test error handling for mismatched input lengths."""
        def simple_function(inputs1, inputs2):
            return inputs1

        inputs1 = [1, 2, 3]
        inputs2 = [10, 20]  # Different length

        with pytest.raises(ValueError, match="All inputs must have the same length"):
            with_max_batchsize(simple_function, inputs1, inputs2)

    def test_no_inputs_error(self):
        """Test error when no inputs are provided."""
        def simple_function():
            return []

        with pytest.raises(ValueError, match="At least one input must be provided"):
            with_max_batchsize(simple_function)

    def test_oom_handling(self, mock_torch_cuda):
        """Test OOM error handling and batch size reduction."""
        call_count = 0

        def oom_function(inputs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2 and len(inputs) > 2:  # Fail for large batches first two times
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return [x * 2 for x in inputs]

        inputs = [1, 2, 3, 4, 5, 6, 7, 8]

        with patch('judgezoo._utils.free_vram') as mock_free:
            result = with_max_batchsize(oom_function, inputs, initial_batch_size=8)

        assert result == [2, 4, 6, 8, 10, 12, 14, 16]
        assert call_count > 1  # Should have retried with smaller batches
        assert mock_free.call_count > 0  # Should have called free_vram

    def test_oom_batch_size_one_error(self):
        """Test error when OOM occurs even with batch size 1."""
        def always_oom_function(inputs):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")

        inputs = [1, 2]

        with patch('judgezoo._utils.free_vram'):
            with pytest.raises(RuntimeError, match="OOM even with batch_size=1"):
                with_max_batchsize(always_oom_function, inputs, initial_batch_size=4)

    def test_next_power_of_two(self):
        """Test the next power of two calculation."""
        def simple_function(inputs):
            return inputs

        # Test various input lengths and verify initial batch size
        test_cases = [
            (1, 1),    # 2^0 = 1
            (2, 2),    # 2^1 = 2
            (3, 4),    # 2^2 = 4
            (5, 8),    # 2^3 = 8
            (9, 16),   # 2^4 = 16
            (100, 128) # 2^7 = 128
        ]

        for input_length, expected_batch_size in test_cases:
            inputs = list(range(input_length))

            # Mock the function to capture the batch sizes used
            batch_sizes_used = []

            def mock_function(batch):
                batch_sizes_used.append(len(batch))
                return batch

            with_max_batchsize(mock_function, inputs)

            # The first batch size should be the next power of two (or input length if smaller)
            first_batch_size = batch_sizes_used[0]
            assert first_batch_size == min(expected_batch_size, input_length)

    def test_verbose_output(self, capsys):
        """Test verbose output functionality."""
        def simple_function(inputs):
            return inputs

        inputs = [1, 2, 3, 4]

        with_max_batchsize(simple_function, inputs, initial_batch_size=2, verbose=True)

        captured = capsys.readouterr()
        # Should have progress output
        assert "Running with batch size" in captured.out or captured.err

    def test_tuple_with_mixed_types_error(self):
        """Test error handling for tuples with mixed types."""
        def mixed_tuple_function(inputs):
            return (torch.tensor([1, 2]), "string")  # Mixed tensor and string

        inputs = [1, 2]

        with pytest.raises(TypeError, match="Wrapped functions may only return Tensors or lists"):
            with_max_batchsize(mixed_tuple_function, inputs, initial_batch_size=1)

    def test_single_batch_processing(self):
        """Test when all data fits in a single batch."""
        def simple_function(inputs):
            return torch.tensor(inputs)

        inputs = [1, 2, 3]
        result = with_max_batchsize(simple_function, inputs, initial_batch_size=10)

        assert torch.equal(result, torch.tensor([1, 2, 3]))

    def test_empty_input(self):
        """Test handling of empty input."""
        def simple_function(inputs):
            return inputs

        inputs = []
        result = with_max_batchsize(simple_function, inputs)

        assert result == []

    def test_assertion_checks(self):
        """Test that length assertions work correctly."""
        def tensor_function(inputs):
            return torch.tensor([[x] for x in inputs])

        inputs = [1, 2, 3, 4]
        result = with_max_batchsize(tensor_function, inputs, initial_batch_size=2)

        # The function should ensure output length matches input length
        assert len(result) == len(inputs)


class TestFreeVram:
    """Test the free_vram utility function."""

    @patch('gc.collect')
    @patch('torch.cuda.empty_cache')
    def test_free_vram_calls(self, mock_empty_cache, mock_gc_collect):
        """Test that free_vram calls the correct functions."""
        free_vram()

        # Should call gc.collect 3 times
        assert mock_gc_collect.call_count == 3

        # Should call torch.cuda.empty_cache 3 times
        assert mock_empty_cache.call_count == 3

    @patch('gc.collect')
    @patch('torch.cuda.empty_cache')
    def test_free_vram_multiple_calls(self, mock_empty_cache, mock_gc_collect):
        """Test multiple calls to free_vram."""
        free_vram()
        free_vram()

        # Should call gc.collect 6 times total
        assert mock_gc_collect.call_count == 6

        # Should call torch.cuda.empty_cache 6 times total
        assert mock_empty_cache.call_count == 6

    def test_free_vram_no_error(self):
        """Test that free_vram doesn't raise errors even if CUDA isn't available."""
        # This should not raise an error even on CPU-only systems
        try:
            free_vram()
        except Exception as e:
            pytest.fail(f"free_vram() raised an unexpected exception: {e}")


class TestIntegration:
    """Integration tests for utility functions."""

    def test_with_oom_simulation(self):
        """Test realistic OOM simulation scenario."""
        oom_triggered = False

        def simulate_model_inference(batch):
            nonlocal oom_triggered
            if len(batch) > 3 and not oom_triggered:
                oom_triggered = True
                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            # Simulate model output
            return {"scores": [0.5] * len(batch)}

        inputs = list(range(10))  # 10 samples

        with patch('judgezoo._utils.free_vram') as mock_free:
            result = with_max_batchsize(simulate_model_inference, inputs, initial_batch_size=8)

        # Should complete successfully
        assert "scores" in result
        assert len(result["scores"]) == 10
        assert all(score == 0.5 for score in result["scores"])

        # Should have called free_vram due to OOM
        assert mock_free.call_count > 0

        # OOM should have been triggered and handled
        assert oom_triggered

    def test_batch_size_progression(self):
        """Test that batch sizes progress correctly during OOM recovery."""
        batch_sizes_seen = []

        def track_batch_sizes(batch):
            batch_sizes_seen.append(len(batch))
            if len(batch) > 4:  # Trigger OOM for large batches
                raise torch.cuda.OutOfMemoryError("Simulated OOM")
            return batch

        inputs = list(range(16))  # 16 samples

        with patch('judgezoo._utils.free_vram'):
            output = with_max_batchsize(track_batch_sizes, inputs, initial_batch_size=16)

        # Should start with large batch and progressively reduce
        assert max(batch_sizes_seen) == 16  # Initial attempt
        assert min(batch_sizes_seen) <= 4   # Should reduce to working size

        # Total processed should equal input length
        assert len(output) == 16