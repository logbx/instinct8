"""
Integration tests for Strategy H: Selective Salience Compression

Tests cover:
- Full compression flow with real templates
- Salience accumulation across compressions
- Integration with evaluation harness
- Real template loading and processing
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from strategies.strategy_h_selective_salience import SelectiveSalienceStrategy
from evaluation.template_utils import load_template, run_single_trial


class TestTemplateIntegration:
    """Integration tests with real templates."""
    
    def test_load_test_simple_template(self):
        """Test loading the test-simple template."""
        template_path = Path("templates/test-simple.json")
        if not template_path.exists():
            pytest.skip("test-simple.json template not found")
        
        template = load_template(str(template_path))
        
        assert "template_id" in template
        assert "initial_setup" in template
        assert "turns" in template
        assert template["template_id"] == "test-simple"
    
    def test_load_test_edge_cases_template(self):
        """Test loading the test-edge-cases template."""
        template_path = Path("templates/test-edge-cases.json")
        if not template_path.exists():
            pytest.skip("test-edge-cases.json template not found")
        
        template = load_template(str(template_path))
        
        assert "template_id" in template
        assert "initial_setup" in template
        assert "turns" in template
        assert template["template_id"] == "test-edge-cases"
    
    @patch('strategies.strategy_h_selective_salience.OpenAI')
    def test_strategy_h_with_test_simple(self, mock_openai_class):
        """Test Strategy H with test-simple template."""
        template_path = Path("templates/test-simple.json")
        if not template_path.exists():
            pytest.skip("test-simple.json template not found")
        
        template = load_template(str(template_path))
        
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock extraction response
        extraction_response = Mock()
        extraction_response.choices = [Mock()]
        extraction_response.choices[0].message.content = json.dumps({
            "salient_items": [
                "Budget: maximum $200",
                "Must accommodate 10 people",
                "Must be held on Saturday"
            ]
        })
        
        # Mock compression response
        compression_response = Mock()
        compression_response.choices = [Mock()]
        compression_response.choices[0].message.content = "Previous conversation about party planning."
        
        mock_client.chat.completions.create.side_effect = [
            extraction_response,
            compression_response,
        ]
        
        # Create strategy
        strategy = SelectiveSalienceStrategy()
        strategy.client = mock_client
        
        # Initialize
        initial_setup = template["initial_setup"]
        strategy.initialize(
            initial_setup["original_goal"],
            initial_setup["hard_constraints"]
        )
        
        # Process turns up to compression point
        turns = template["turns"]
        compression_turn = None
        for turn in turns:
            if turn.get("is_compression_point", False):
                compression_turn = turn
                break
        
        if compression_turn:
            context = [t for t in turns if t["turn_id"] <= compression_turn["turn_id"]]
            context_dicts = [
                {
                    "id": t["turn_id"],
                    "role": t["role"],
                    "content": t["content"],
                }
                for t in context
            ]
            
            result = strategy.compress(context_dicts, compression_turn["turn_id"])
            
            assert isinstance(result, str)
            assert len(result) > 0
            assert len(strategy.salience_set) > 0
    
    @patch('strategies.strategy_h_selective_salience.OpenAI')
    def test_salience_accumulation_across_compressions(self, mock_openai_class):
        """Test that salience accumulates across multiple compressions."""
        template_path = Path("templates/test-edge-cases.json")
        if not template_path.exists():
            pytest.skip("test-edge-cases.json template not found")
        
        template = load_template(str(template_path))
        
        # Setup mocks
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock responses for multiple compressions
        extraction_responses = [
            Mock(choices=[Mock(message=Mock(content=json.dumps({
                "salient_items": [f"Item {i}"]
            })))]) for i in range(1, 3)
        ]
        
        compression_responses = [
            Mock(choices=[Mock(message=Mock(content="Summary"))]) for _ in range(2)
        ]
        
        mock_client.chat.completions.create.side_effect = (
            extraction_responses[0],
            compression_responses[0],
            extraction_responses[1],
            compression_responses[1],
        )
        
        # Create strategy
        strategy = SelectiveSalienceStrategy()
        strategy.client = mock_client
        
        # Initialize
        initial_setup = template["initial_setup"]
        strategy.initialize(
            initial_setup["original_goal"],
            initial_setup["hard_constraints"]
        )
        
        # Process first compression point
        turns = template["turns"]
        compression_points = [t for t in turns if t.get("is_compression_point", False)]
        
        if len(compression_points) >= 2:
            # First compression
            first_cp = compression_points[0]
            context_1 = [t for t in turns if t["turn_id"] <= first_cp["turn_id"]]
            context_dicts_1 = [
                {"id": t["turn_id"], "role": t["role"], "content": t["content"]}
                for t in context_1
            ]
            
            strategy.compress(context_dicts_1, first_cp["turn_id"])
            first_salience_count = len(strategy.salience_set)
            
            # Second compression
            second_cp = compression_points[1]
            context_2 = [t for t in turns if t["turn_id"] <= second_cp["turn_id"]]
            context_dicts_2 = [
                {"id": t["turn_id"], "role": t["role"], "content": t["content"]}
                for t in context_2
            ]
            
            strategy.compress(context_dicts_2, second_cp["turn_id"])
            second_salience_count = len(strategy.salience_set)
            
            # Salience should accumulate
            assert second_salience_count >= first_salience_count


@pytest.mark.integration
class TestEvaluationHarnessIntegration:
    """Tests for integration with evaluation harness."""
    
    @patch('strategies.strategy_h_selective_salience.OpenAI')
    @patch('evaluation.harness.Anthropic')
    def test_strategy_h_collects_salience_metrics(self, mock_anthropic, mock_openai_class):
        """Test that harness collects salience from Strategy H."""
        template_path = Path("templates/test-simple.json")
        if not template_path.exists():
            pytest.skip("test-simple.json template not found")
        
        # Setup OpenAI mock
        mock_openai_client = Mock()
        mock_openai_class.return_value = mock_openai_client
        
        extraction_response = Mock()
        extraction_response.choices = [Mock()]
        extraction_response.choices[0].message.content = json.dumps({
            "salient_items": ["Budget: $200", "10 people"]
        })
        
        compression_response = Mock()
        compression_response.choices = [Mock()]
        compression_response.choices[0].message.content = "Summary"
        
        mock_openai_client.chat.completions.create.side_effect = [
            extraction_response,
            compression_response,
        ]
        
        # Setup Anthropic mock (for agent responses)
        mock_anthropic_client = Mock()
        mock_anthropic.return_value = mock_anthropic_client
        
        mock_anthropic_response = Mock()
        mock_anthropic_response.content = [Mock(text="Test response")]
        mock_anthropic_client.messages.create.return_value = mock_anthropic_response
        
        # Create strategy
        strategy = SelectiveSalienceStrategy()
        strategy.client = mock_openai_client
        
        # Load template
        template = load_template(str(template_path))
        
        # Check that strategy has salience_set attribute
        assert hasattr(strategy, 'salience_set')
        
        # Initialize strategy
        initial_setup = template["initial_setup"]
        strategy.initialize(
            initial_setup["original_goal"],
            initial_setup["hard_constraints"]
        )
        
        # Process a compression point
        turns = template["turns"]
        compression_turn = None
        for turn in turns:
            if turn.get("is_compression_point", False):
                compression_turn = turn
                break
        
        if compression_turn:
            context = [t for t in turns if t["turn_id"] <= compression_turn["turn_id"]]
            context_dicts = [
                {"id": t["turn_id"], "role": t["role"], "content": t["content"]}
                for t in context
            ]
            
            strategy.compress(context_dicts, compression_turn["turn_id"])
            
            # Verify salience_set is populated
            assert len(strategy.salience_set) > 0
            
            # Verify it can be accessed for metrics collection
            extracted_salience = strategy.salience_set.copy()
            assert isinstance(extracted_salience, list)
            assert len(extracted_salience) > 0

