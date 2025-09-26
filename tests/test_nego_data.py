import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_loader import NegoCompletionDataset
from data_provider.data_factory import nego_completion_collate_fn
from models.timer import Model as TimerModel


class MockTimerConfig:
    """Mock configuration for Timer model testing."""
    def __init__(self):
        self.input_token_len = 96
        self.output_token_len = 96
        self.d_model = 256
        self.n_heads = 4
        self.d_ff = 1024
        self.e_layers = 2
        self.dropout = 0.1
        self.activation = 'relu'
        self.use_norm = False


class TestNegoCompletionData:
    """Test suite for negotiation completion data logic and padding functionality."""

    def setup_method(self):
        """Setup test data and paths."""
        self.data_root = "/Users/emrekuru/Developer/Thesis/Negoformer/Negoformer_Training/data/oracle/test"
        self.min_input_len = 1000

    def test_variable_input_output_split(self):
        """Test that input/output splitting works correctly with real negotiation data."""
        # Create dataset with real data
        dataset = NegoCompletionDataset(
            root_path=self.data_root,
            flag='train',
            size=[None, None, None],  # Not used in NegoCompletionDataset
            min_input_len=self.min_input_len
        )

        assert len(dataset) > 0, "Dataset should contain completion pairs"

        # Test a specific sample
        sample_idx = 0
        seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[sample_idx]

        # Get the original session data for comparison
        session_idx, split_point = dataset.completion_pairs[sample_idx]
        original_session = dataset.data_list[session_idx]

        # Verify input is history up to split point
        expected_input = torch.tensor(original_session[:split_point], dtype=torch.float32)
        assert torch.allclose(seq_x, expected_input), f"Input sequence should match session[:split_point]"
        assert seq_x.shape[0] == split_point, f"Input length should be {split_point}, got {seq_x.shape[0]}"

        # Verify output is remaining session after split point
        expected_output = torch.tensor(original_session[split_point:], dtype=torch.float32)
        assert torch.allclose(seq_y, expected_output), f"Output sequence should match session[split_point:]"
        assert seq_y.shape[0] == len(original_session) - split_point, f"Output length should be {len(original_session) - split_point}, got {seq_y.shape[0]}"

        # Verify input + output = complete session
        reconstructed = torch.cat([seq_x, seq_y], dim=0)
        original_tensor = torch.tensor(original_session, dtype=torch.float32)
        assert torch.allclose(reconstructed, original_tensor), "Input + Output should equal original session"

        # Verify feature dimensions (should be 11 features after removing metadata columns)
        assert seq_x.shape[1] == 11, f"Input should have 11 features, got {seq_x.shape[1]}"
        assert seq_y.shape[1] == 11, f"Output should have 11 features, got {seq_y.shape[1]}"

        print(f"✓ Variable input/output split test passed")
        print(f"  Input shape: {seq_x.shape}, Output shape: {seq_y.shape}")
        print(f"  Split point: {split_point}, Session length: {len(original_session)}")

    def test_collate_function_padding(self):
        """Test the custom collate function properly pads variable-length sequences."""
        # Create mock batch with different sequence lengths
        batch_size = 3
        feature_dim = 11

        # Create sequences of different lengths
        seq_lengths_x = [1200, 1800, 2100]
        seq_lengths_y = [800, 1500, 900]

        batch = []
        for i in range(batch_size):
            seq_x = torch.randn(seq_lengths_x[i], feature_dim)
            seq_y = torch.randn(seq_lengths_y[i], feature_dim)
            seq_x_mark = torch.zeros(seq_lengths_x[i], 1)
            seq_y_mark = torch.zeros(seq_lengths_y[i], 1)
            batch.append((seq_x, seq_y, seq_x_mark, seq_y_mark))

        # Apply collate function
        batch_x, batch_y, batch_x_mark, batch_y_mark = nego_completion_collate_fn(batch)

        # Verify all tensors have the same dimensions within batch
        max_x_len = max(seq_lengths_x)
        max_y_len = max(seq_lengths_y)

        assert batch_x.shape == (batch_size, max_x_len, feature_dim), f"Expected batch_x shape {(batch_size, max_x_len, feature_dim)}, got {batch_x.shape}"
        assert batch_y.shape == (batch_size, max_y_len, feature_dim), f"Expected batch_y shape {(batch_size, max_y_len, feature_dim)}, got {batch_y.shape}"
        assert batch_x_mark.shape == (batch_size, max_x_len, 1), f"Expected batch_x_mark shape {(batch_size, max_x_len, 1)}, got {batch_x_mark.shape}"
        assert batch_y_mark.shape == (batch_size, max_y_len, 1), f"Expected batch_y_mark shape {(batch_size, max_y_len, 1)}, got {batch_y_mark.shape}"

        # Verify padding areas contain zeros
        for i in range(batch_size):
            # Check padding in x sequences
            if seq_lengths_x[i] < max_x_len:
                padding_x = batch_x[i, seq_lengths_x[i]:, :]
                assert torch.all(padding_x == 0), f"Padding in batch_x[{i}] should be zeros"

            # Check padding in y sequences
            if seq_lengths_y[i] < max_y_len:
                padding_y = batch_y[i, seq_lengths_y[i]:, :]
                assert torch.all(padding_y == 0), f"Padding in batch_y[{i}] should be zeros"

        print(f"✓ Collate function padding test passed")
        print(f"  Input batch shape: {batch_x.shape}")
        print(f"  Output batch shape: {batch_y.shape}")
        print(f"  Original lengths - X: {seq_lengths_x}, Y: {seq_lengths_y}")

    def test_10k_length_capping(self):
        """Test that sequences longer than 10k are properly truncated."""
        # Create mock batch with sequences longer than 10k
        feature_dim = 11
        long_seq_len = 12000  # Longer than 10k
        short_seq_len = 8000

        batch = []
        # Add one long sequence and one normal sequence
        seq_x_long = torch.randn(long_seq_len, feature_dim)
        seq_y_long = torch.randn(long_seq_len, feature_dim)
        seq_x_mark_long = torch.zeros(long_seq_len, 1)
        seq_y_mark_long = torch.zeros(long_seq_len, 1)

        seq_x_short = torch.randn(short_seq_len, feature_dim)
        seq_y_short = torch.randn(short_seq_len, feature_dim)
        seq_x_mark_short = torch.zeros(short_seq_len, 1)
        seq_y_mark_short = torch.zeros(short_seq_len, 1)

        batch.append((seq_x_long, seq_y_long, seq_x_mark_long, seq_y_mark_long))
        batch.append((seq_x_short, seq_y_short, seq_x_mark_short, seq_y_mark_short))

        # Apply collate function
        batch_x, batch_y, batch_x_mark, batch_y_mark = nego_completion_collate_fn(batch)

        # Verify sequences are capped at 10k
        assert batch_x.shape[1] <= 10000, f"Batch X length should be <= 10000, got {batch_x.shape[1]}"
        assert batch_y.shape[1] <= 10000, f"Batch Y length should be <= 10000, got {batch_y.shape[1]}"

        # Since we have one 12k and one 8k sequence, the max should be 10k (capped)
        expected_max_len = 10000
        assert batch_x.shape[1] == expected_max_len, f"Expected max length {expected_max_len}, got {batch_x.shape[1]}"
        assert batch_y.shape[1] == expected_max_len, f"Expected max length {expected_max_len}, got {batch_y.shape[1]}"

        print(f"✓ 10k length capping test passed")
        print(f"  Original lengths: {long_seq_len}, {short_seq_len}")
        print(f"  Final batch shape: {batch_x.shape}")

    def test_integration_with_real_data(self):
        """Integration test with DataLoader using real negotiation data."""
        # Create dataset
        dataset = NegoCompletionDataset(
            root_path=self.data_root,
            flag='train',
            size=[None, None, None],
            min_input_len=self.min_input_len
        )

        # Create DataLoader with custom collate function
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=nego_completion_collate_fn
        )

        # Test iteration through batches
        batch_count = 0
        for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
            # Verify no tensor stacking errors occurred
            assert isinstance(batch_x, torch.Tensor), "batch_x should be a tensor"
            assert isinstance(batch_y, torch.Tensor), "batch_y should be a tensor"

            # Verify batch dimensions
            batch_size = batch_x.shape[0]
            assert batch_x.shape[0] == batch_y.shape[0], f"Batch sizes should match: {batch_x.shape[0]} vs {batch_y.shape[0]}"
            assert batch_x.shape[2] == 11, f"Should have 11 features, got {batch_x.shape[2]}"
            assert batch_y.shape[2] == 11, f"Should have 11 features, got {batch_y.shape[2]}"

            # Verify no NaN or infinite values
            assert torch.isfinite(batch_x).all(), "batch_x should not contain NaN or infinite values"
            assert torch.isfinite(batch_y).all(), "batch_y should not contain NaN or infinite values"

            batch_count += 1
            if batch_count >= 3:  # Test first few batches
                break

        assert batch_count > 0, "Should have processed at least one batch"

        print(f"✓ Integration test with real data passed")
        print(f"  Processed {batch_count} batches successfully")
        print(f"  Final batch shapes - X: {batch_x.shape}, Y: {batch_y.shape}")

    def test_dataset_constraints(self):
        """Test dataset respects minimum input length constraints."""
        dataset = NegoCompletionDataset(
            root_path=self.data_root,
            flag='train',
            size=[None, None, None],
            min_input_len=self.min_input_len
        )

        # Verify all completion pairs respect minimum input length
        for session_idx, split_point in dataset.completion_pairs:
            assert split_point >= self.min_input_len, f"Split point {split_point} should be >= {self.min_input_len}"

            # Verify there's at least one output step
            session_length = len(dataset.data_list[session_idx])
            assert split_point < session_length, f"Split point {split_point} should be < session length {session_length}"

        print(f"✓ Dataset constraints test passed")
        print(f"  All {len(dataset.completion_pairs)} completion pairs respect min_input_len={self.min_input_len}")

    def test_loss_computation(self):
        """Test loss computation with Timer model using variable-length padded data."""
        # Create Timer model
        config = MockTimerConfig()
        model = TimerModel(config)
        model.eval()  # Set to eval mode for consistent behavior

        # Create mock batch with variable sequence lengths (like collate function produces)
        batch_size = 3
        feature_dim = 11

        # Different input sequence lengths
        seq_lengths_x = [1200, 1800, 2100]
        seq_lengths_y = [800, 1500, 900]

        # Create batch using our collate function
        batch = []
        for i in range(batch_size):
            seq_x = torch.randn(seq_lengths_x[i], feature_dim)
            seq_y = torch.randn(seq_lengths_y[i], feature_dim)
            seq_x_mark = torch.zeros(seq_lengths_x[i], 1)
            seq_y_mark = torch.zeros(seq_lengths_y[i], 1)
            batch.append((seq_x, seq_y, seq_x_mark, seq_y_mark))

        # Apply collate function to create padded batch
        batch_x, batch_y, batch_x_mark, batch_y_mark = nego_completion_collate_fn(batch)

        # Forward pass through model
        with torch.no_grad():
            outputs = model(batch_x, batch_x_mark, batch_y_mark)

        # Verify model output shape
        assert outputs.shape[0] == batch_size, f"Output batch size should be {batch_size}, got {outputs.shape[0]}"
        assert outputs.shape[2] == feature_dim, f"Output features should be {feature_dim}, got {outputs.shape[2]}"

        # Test loss computation (MSE loss like in training)
        criterion = torch.nn.MSELoss()

        # The Timer model processes input tokens and produces output tokens
        # For loss computation, we need to align the output with the target
        # Take only the output portion that matches target length
        min_seq_len = min(outputs.shape[1], batch_y.shape[1])
        outputs_for_loss = outputs[:, :min_seq_len, :]  # Truncate to target length
        batch_y_for_loss = batch_y[:, :min_seq_len, :]   # Truncate to match

        # Compute loss
        loss = criterion(outputs_for_loss, batch_y_for_loss)

        # Verify loss properties
        assert torch.isfinite(loss), "Loss should be finite (not NaN or Inf)"
        assert loss.item() >= 0, f"MSE loss should be non-negative, got {loss.item()}"
        assert loss.item() < 1000, f"Loss seems too large: {loss.item()}, possible issue with model or data"

        # Test gradient computation (ensure backprop works)
        model.train()
        outputs_grad = model(batch_x, batch_x_mark, batch_y_mark)
        outputs_grad_aligned = outputs_grad[:, :min_seq_len, :]  # Align for loss computation
        loss_with_grad = criterion(outputs_grad_aligned, batch_y_for_loss)
        loss_with_grad.backward()

        # Check that gradients are computed
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break

        assert has_gradients, "Model should have non-zero gradients after backward pass"

        # Verify no gradient explosion
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        assert total_grad_norm < 1000, f"Gradient norm too large: {total_grad_norm}, possible gradient explosion"

        print(f"✓ Loss computation test passed")
        print(f"  Model output shape: {outputs.shape}")
        print(f"  Loss value: {loss.item():.6f}")
        print(f"  Total gradient norm: {total_grad_norm:.6f}")
        print(f"  Input batch shapes - X: {batch_x.shape}, Y: {batch_y.shape}")


if __name__ == "__main__":
    # Run tests directly
    test_instance = TestNegoCompletionData()
    test_instance.setup_method()

    print("Running Negotiation Completion Data Tests...")
    print("=" * 50)

    try:
        test_instance.test_variable_input_output_split()
        test_instance.test_collate_function_padding()
        test_instance.test_10k_length_capping()
        test_instance.test_integration_with_real_data()
        test_instance.test_dataset_constraints()
        test_instance.test_loss_computation()

        print("=" * 50)
        print("🎉 All tests passed! The tensor dimension fix is working correctly.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise