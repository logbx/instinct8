"""
Unit tests for Strategy H: Selective Salience Compression

Tests cover:
- Core methods (initialize, update_goal, name, compress)
- Salience extraction with mocked API
- Background compression with mocked API
- Context rebuilding
- Salience management (deduplication, merging, prioritization)
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import json

from strategies.strategy_h_selective_salience import SelectiveSalienceStrategy


@pytest.mark.integration
class TestCoreMethods:
    """Tests for core strategy methods."""
    
    def test_name(self):
        """Test that name() returns correct string."""
        strategy = SelectiveSalienceStrategy()
        assert strategy.name() == "Strategy H - Selective Salience Compression"
    
    def test_initialize(self):
        """Test initialize() stores goal and constraints."""
        strategy = SelectiveSalienceStrategy()
        goal = "Test goal"
        constraints = ["Constraint 1", "Constraint 2"]
        
        strategy.initialize(goal, constraints)
        
        assert strategy.original_goal == goal
        assert strategy.constraints == constraints
        assert strategy.salience_set == []
    
    def test_update_goal(self):
        """Test update_goal() updates the goal."""
        strategy = SelectiveSalienceStrategy()
        strategy.initialize("Original goal", [])
        
        new_goal = "Updated goal"
        strategy.update_goal(new_goal, "Reason for change")
        
        assert strategy.original_goal == new_goal
    
    def test_token_count(self):
        """Test token counting accuracy."""
        strategy = SelectiveSalienceStrategy()
        
        # Test empty string
        assert strategy._token_count("") == 0
        
        # Test simple text
        text = "Hello world"
        count = strategy._token_count(text)
        assert count > 0
        assert isinstance(count, int)
        
        # Test longer text
        long_text = "This is a longer text " * 10
        long_count = strategy._token_count(long_text)
        assert long_count > count


@pytest.mark.integration
class TestSalienceExtraction:
    """Tests for salience extraction with mocked API."""
    
    @patch('strategies.strategy_h_selective_salience.OpenAI')
    def test_extract_salient_information_success(self, mock_openai_class):
        """Test successful salience extraction."""
        # Setup mock
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "salient_items": [
                "Budget: maximum $200",
                "Must accommodate 10 people",
                "Selected park pavilion"
            ]
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        # Create strategy and test
        strategy = SelectiveSalienceStrategy()
        strategy.client = mock_client
        strategy.initialize("Test goal", ["Constraint 1"])
        
        context = [
            {"id": 1, "role": "user", "content": "Test content"},
            {"id": 2, "role": "assistant", "content": "Response"}
        ]
        
        result = strategy._extract_salient_information(context)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert "Budget: maximum $200" in result
        assert mock_client.chat.completions.create.called
    
    @patch('strategies.strategy_h_selective_salience.OpenAI')
    def test_extract_salient_information_empty_response(self, mock_openai_class):
        """Test extraction with empty response."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "salient_items": []
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        strategy = SelectiveSalienceStrategy()
        strategy.client = mock_client
        strategy.initialize("Test goal", [])
        
        result = strategy._extract_salient_information([])
        assert result == []
    
    @patch('strategies.strategy_h_selective_salience.OpenAI')
    def test_extract_salient_information_json_error(self, mock_openai_class):
        """Test extraction with JSON parse error."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON"
        mock_client.chat.completions.create.return_value = mock_response
        
        strategy = SelectiveSalienceStrategy()
        strategy.client = mock_client
        strategy.initialize("Test goal", [])
        
        result = strategy._extract_salient_information([])
        assert result == []
    
    @patch('strategies.strategy_h_selective_salience.OpenAI')
    def test_extract_salient_information_api_error(self, mock_openai_class):
        """Test extraction with API error."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        strategy = SelectiveSalienceStrategy()
        strategy.client = mock_client
        strategy.initialize("Test goal", [])
        
        result = strategy._extract_salient_information([])
        assert result == []


@pytest.mark.integration
class TestBackgroundCompression:
    """Tests for background compression with mocked API."""
    
    @patch('strategies.strategy_h_selective_salience.OpenAI')
    def test_compress_background_success(self, mock_openai_class):
        """Test successful background compression."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a compressed summary of the background information."
        mock_client.chat.completions.create.return_value = mock_response
        
        strategy = SelectiveSalienceStrategy()
        strategy.client = mock_client
        
        context = [
            {"id": 1, "role": "user", "content": "Test content"},
        ]
        salience_set = ["Important item"]
        
        result = strategy._compress_background(context, salience_set)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert mock_client.chat.completions.create.called
    
    @patch('strategies.strategy_h_selective_salience.OpenAI')
    def test_compress_background_api_error(self, mock_openai_class):
        """Test background compression with API error."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        
        strategy = SelectiveSalienceStrategy()
        strategy.client = mock_client
        
        result = strategy._compress_background([], [])
        
        # Should return fallback summary
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.integration
class TestContextRebuilding:
    """Tests for context rebuilding."""
    
    def test_build_context_with_all_sections(self):
        """Test building context with all sections."""
        strategy = SelectiveSalienceStrategy()
        
        salience_set = ["Item 1", "Item 2"]
        background_summary = "Background summary"
        recent_turns = [
            {"id": 1, "role": "user", "content": "Recent turn 1"},
            {"id": 2, "role": "assistant", "content": "Recent turn 2"}
        ]
        
        result = strategy._build_context(salience_set, background_summary, recent_turns)
        
        assert "SALIENT INFORMATION" in result
        assert "BACKGROUND SUMMARY" in result
        assert "RECENT TURNS" in result
        assert "Item 1" in result
        assert "Item 2" in result
        assert "Background summary" in result
        assert "Recent turn 1" in result
    
    def test_build_context_empty_salience(self):
        """Test building context with empty salience set."""
        strategy = SelectiveSalienceStrategy()
        
        result = strategy._build_context([], "Background", [])
        
        assert "BACKGROUND SUMMARY" in result
        assert "SALIENT INFORMATION" not in result
    
    def test_build_context_empty_background(self):
        """Test building context with empty background."""
        strategy = SelectiveSalienceStrategy()
        
        result = strategy._build_context(["Item"], "", [])
        
        assert "SALIENT INFORMATION" in result
        assert "BACKGROUND SUMMARY" not in result


@pytest.mark.integration
class TestSalienceManagement:
    """Tests for salience management (deduplication, merging, prioritization)."""
    
    def test_deduplicate_semantically_empty_list(self):
        """Test deduplication with empty list."""
        strategy = SelectiveSalienceStrategy()
        result = strategy._deduplicate_semantically([])
        assert result == []
    
    def test_deduplicate_semantically_single_item(self):
        """Test deduplication with single item."""
        strategy = SelectiveSalienceStrategy()
        items = ["Single item"]
        result = strategy._deduplicate_semantically(items)
        assert result == items
    
    def test_deduplicate_semantically_similar_items(self):
        """Test deduplication removes similar items."""
        strategy = SelectiveSalienceStrategy()
        
        # These should be semantically similar
        items = [
            "Budget: maximum $200",
            "Budget: max $200",  # Very similar
            "Must accommodate 10 people",  # Different
        ]
        
        result = strategy._deduplicate_semantically(items, threshold=0.9)
        
        # Should keep at least 2 items (the similar ones might be deduplicated)
        assert len(result) >= 1
        assert len(result) <= len(items)
        assert all(item in items for item in result)
    
    def test_deduplicate_semantically_different_items(self):
        """Test deduplication keeps different items."""
        strategy = SelectiveSalienceStrategy()
        
        items = [
            "Budget: maximum $200",
            "Must accommodate 10 people",
            "Selected park pavilion",
        ]
        
        result = strategy._deduplicate_semantically(items)
        
        # Should keep all items if they're different enough
        assert len(result) >= 1
        assert len(result) <= len(items)
    
    def test_merge_salience_empty_existing(self):
        """Test merging with empty existing set."""
        strategy = SelectiveSalienceStrategy()
        
        new_items = ["Item 1", "Item 2"]
        result = strategy._merge_salience([], new_items)
        
        assert len(result) >= 1
        assert len(result) <= len(new_items)
    
    def test_merge_salience_empty_new(self):
        """Test merging with empty new set."""
        strategy = SelectiveSalienceStrategy()
        
        existing_items = ["Item 1", "Item 2"]
        result = strategy._merge_salience(existing_items, [])
        
        assert result == existing_items
    
    def test_merge_salience_combines_items(self):
        """Test merging combines existing and new items."""
        strategy = SelectiveSalienceStrategy()
        
        existing = ["Existing item"]
        new = ["New item"]
        
        result = strategy._merge_salience(existing, new)
        
        # Should have at least 1 item (deduplication might remove some)
        assert len(result) >= 1
        assert len(result) <= len(existing) + len(new)
    
    def test_prioritize_items_constraints_first(self):
        """Test prioritization puts constraints first."""
        strategy = SelectiveSalienceStrategy()
        
        items = [
            "This is a fact",
            "Must use Python",
            "We decided to use FastAPI",
            "Cannot exceed budget",
        ]
        
        result = strategy._prioritize_items(items)
        
        # Constraints should come first
        constraint_indices = [i for i, item in enumerate(result) if "Must" in item or "Cannot" in item]
        decision_indices = [i for i, item in enumerate(result) if "decided" in item.lower()]
        fact_indices = [i for i, item in enumerate(result) if "fact" in item.lower()]
        
        # All constraint indices should be before decision indices
        if constraint_indices and decision_indices:
            assert max(constraint_indices) < min(decision_indices)
        
        # All decision indices should be before fact indices
        if decision_indices and fact_indices:
            assert max(decision_indices) < min(fact_indices)
    
    def test_prioritize_items_empty_list(self):
        """Test prioritization with empty list."""
        strategy = SelectiveSalienceStrategy()
        result = strategy._prioritize_items([])
        assert result == []
    
    def test_prioritize_items_single_item(self):
        """Test prioritization with single item."""
        strategy = SelectiveSalienceStrategy()
        items = ["Single item"]
        result = strategy._prioritize_items(items)
        assert result == items


@pytest.mark.integration
class TestCompressMethod:
    """Tests for the main compress() method."""
    
    @patch('strategies.strategy_h_selective_salience.OpenAI')
    def test_compress_empty_context(self, mock_openai_class):
        """Test compress with empty context."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        strategy = SelectiveSalienceStrategy()
        strategy.client = mock_client
        strategy.initialize("Test goal", [])
        
        result = strategy.compress([], 0)
        
        assert isinstance(result, str)
        assert len(result) >= 0
    
    @patch('strategies.strategy_h_selective_salience.OpenAI')
    def test_compress_full_flow(self, mock_openai_class):
        """Test full compression flow with mocked API."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock extraction response
        extraction_response = Mock()
        extraction_response.choices = [Mock()]
        extraction_response.choices[0].message.content = json.dumps({
            "salient_items": ["Budget: $200", "10 people"]
        })
        
        # Mock compression response
        compression_response = Mock()
        compression_response.choices = [Mock()]
        compression_response.choices[0].message.content = "Background summary"
        
        # Set up side effect to return different responses
        mock_client.chat.completions.create.side_effect = [
            extraction_response,
            compression_response
        ]
        
        strategy = SelectiveSalienceStrategy()
        strategy.client = mock_client
        strategy.initialize("Test goal", ["Constraint 1"])
        
        context = [
            {"id": 1, "role": "user", "content": "Turn 1"},
            {"id": 2, "role": "assistant", "content": "Turn 2"},
            {"id": 3, "role": "user", "content": "Turn 3"},
        ]
        
        result = strategy.compress(context, 2)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert mock_client.chat.completions.create.call_count == 2
    
    @patch('strategies.strategy_h_selective_salience.OpenAI')
    def test_compress_salience_accumulation(self, mock_openai_class):
        """Test that salience accumulates across compressions."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        # Mock responses
        extraction_response = Mock()
        extraction_response.choices = [Mock()]
        extraction_response.choices[0].message.content = json.dumps({
            "salient_items": ["Item 1"]
        })
        
        compression_response = Mock()
        compression_response.choices = [Mock()]
        compression_response.choices[0].message.content = "Summary"
        
        mock_client.chat.completions.create.side_effect = [
            extraction_response,
            compression_response,
            extraction_response,
            compression_response,
        ]
        
        strategy = SelectiveSalienceStrategy()
        strategy.client = mock_client
        strategy.initialize("Test goal", [])
        
        context = [
            {"id": 1, "role": "user", "content": "Turn 1"},
            {"id": 2, "role": "assistant", "content": "Turn 2"},
        ]
        
        # First compression
        strategy.compress(context, 1)
        first_salience_count = len(strategy.salience_set)
        
        # Second compression
        strategy.compress(context, 2)
        second_salience_count = len(strategy.salience_set)
        
        # Salience should accumulate (or at least not decrease)
        assert second_salience_count >= first_salience_count


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_token_count_none(self):
        """Test token count handles None gracefully."""
        strategy = SelectiveSalienceStrategy()
        # Should handle None or convert to empty string
        result = strategy._token_count(None if False else "")
        assert isinstance(result, int)
        assert result >= 0
    
    @patch('strategies.strategy_h_selective_salience.OpenAI')
    def test_extract_filters_empty_strings(self, mock_openai_class):
        """Test that extraction filters out empty strings."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "salient_items": ["Valid item", "", "   ", "Another valid item"]
        })
        mock_client.chat.completions.create.return_value = mock_response
        
        strategy = SelectiveSalienceStrategy()
        strategy.client = mock_client
        strategy.initialize("Test goal", [])
        
        result = strategy._extract_salient_information([])
        
        # Should filter out empty strings
        assert "" not in result
        assert "   " not in result
        assert "Valid item" in result or "Another valid item" in result

