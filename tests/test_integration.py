"""
Integration tests for speculative decoding.
Tests the full pipeline end-to-end.
"""

import pytest
import torch


class TestSpeculativeConfig:
    """Test suite for SpeculativeConfig validation."""
    
    def test_valid_config(self):
        from prolepsis.config import SpeculativeConfig
        
        config = SpeculativeConfig(
            draft_model_name="test/draft",
            target_model_name="test/target",
            gamma=4,
        )
        
        assert config.gamma == 4
        assert config.temperature == 1.0
    
    def test_invalid_gamma(self):
        from prolepsis.config import SpeculativeConfig
        
        with pytest.raises(ValueError, match="(?i)gamma must be >= 1"):
            SpeculativeConfig(
                draft_model_name="test/draft",
                target_model_name="test/target",
                gamma=0,
            )
    
    def test_invalid_temperature(self):
        from prolepsis.config import SpeculativeConfig
        
        with pytest.raises(ValueError, match="(?i)temperature cannot be negative"):
            SpeculativeConfig(
                draft_model_name="test/draft",
                target_model_name="test/target",
                temperature=-1.0,
            )
    
    def test_high_gamma_warning(self):
        from prolepsis.config import SpeculativeConfig
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SpeculativeConfig(
                draft_model_name="test/draft",
                target_model_name="test/target",
                gamma=20,
            )
            assert len(w) >= 1
            assert "gamma=20 is unusually high" in str(w[0].message)
    
    def test_get_torch_dtype(self):
        from prolepsis.config import SpeculativeConfig
        
        config = SpeculativeConfig(
            draft_model_name="test/draft",
            target_model_name="test/target",
            dtype="bfloat16",
        )
        
        assert config.get_torch_dtype() == torch.bfloat16


class TestModelWrappers:
    """Test model wrapper interfaces (without actual models)."""
    
    def test_draft_wrapper_interface(self):
        """Test that DraftModelWrapper has expected methods."""
        from prolepsis.models.wrapper import DraftModelWrapper
        
        # Check interface exists (not instantiating - requires real model)
        assert hasattr(DraftModelWrapper, 'prefill')
        assert hasattr(DraftModelWrapper, 'decode_one')
        assert hasattr(DraftModelWrapper, 'truncate_cache')
        assert hasattr(DraftModelWrapper, 'reset')
    
    def test_target_wrapper_interface(self):
        """Test that TargetModelWrapper has expected methods."""
        from prolepsis.models.wrapper import TargetModelWrapper
        
        assert hasattr(TargetModelWrapper, 'prefill')
        assert hasattr(TargetModelWrapper, 'verify_forward')
        assert hasattr(TargetModelWrapper, 'truncate_cache')
        assert hasattr(TargetModelWrapper, 'reset')


# Skip integration tests that require actual models
@pytest.mark.skip(reason="Requires actual models - run manually")
class TestEndToEnd:
    """End-to-end integration tests with real models."""
    
    def test_generate_produces_output(self):
        from prolepsis import SpeculativeDecoder, SpeculativeConfig
        
        config = SpeculativeConfig(
            draft_model_name="Qwen/Qwen3-1.7B",
            target_model_name="Qwen/Qwen3-8B",
            gamma=4,
        )
        
        decoder = SpeculativeDecoder(config)
        test_prompt = "The deployment of speculative decoding requires"
        output = decoder.generate(test_prompt, max_new_tokens=10)
        
        assert len(output) > len(test_prompt)
        assert decoder.get_tokens_per_pass() > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
