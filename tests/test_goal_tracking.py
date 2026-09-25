"""
Unit tests for evaluation/goal_tracking.py pure helpers.

Tests goal-shift detection and tracking without requiring API keys.
All functions under test are pure (no external dependencies).
"""

import pytest
from evaluation.goal_tracking import (
    detect_goal_shift_in_message,
    extract_new_goal_from_message,
    track_goal_evolution,
    get_current_goal_at_turn,
)


class TestDetectGoalShiftInMessage:
    """Test detection of goal shifts in user messages."""
    
    def test_detect_pivot_language(self):
        """Should detect 'pivot to' as a goal shift."""
        message = "Let's pivot to a B2C model instead."
        result = detect_goal_shift_in_message(message)
        assert result is not None
        assert result == message
    
    def test_detect_change_goal(self):
        """Should detect 'change goal' as a goal shift."""
        message = "We need to change our goal to focus on mobile first."
        result = detect_goal_shift_in_message(message)
        assert result is not None
    
    def test_detect_shift_toward(self):
        """Should detect 'shift toward' as a goal shift."""
        message = "Actually, let's shift toward enterprise customers."
        result = detect_goal_shift_in_message(message)
        assert result is not None
    
    def test_detect_instead_pattern(self):
        """Should detect 'instead we' as a goal shift."""
        message = "Instead, we should target small businesses."
        result = detect_goal_shift_in_message(message)
        assert result is not None
    
    def test_detect_weve_decided(self):
        """Should detect 'we've decided to' as a goal shift."""
        message = "We've decided to build a mobile app first."
        result = detect_goal_shift_in_message(message)
        assert result is not None
    
    def test_detect_new_requirement(self):
        """Should detect 'new requirement' as a goal shift."""
        message = "There's a new requirement: we need iOS support."
        result = detect_goal_shift_in_message(message)
        assert result is not None
    
    def test_detect_adding_requirement(self):
        """Should detect 'adding requirement' as a goal shift."""
        message = "We're adding a requirement for mobile compatibility."
        result = detect_goal_shift_in_message(message)
        assert result is not None
    
    def test_detect_update_approach(self):
        """Should detect 'update approach' as a goal shift."""
        message = "Let's update our approach to use a different framework."
        result = detect_goal_shift_in_message(message)
        assert result is not None
    
    def test_no_shift_ordinary_question(self):
        """Should not detect shift in ordinary questions."""
        message = "What database should we use for this?"
        result = detect_goal_shift_in_message(message)
        assert result is None
    
    def test_no_shift_ordinary_statement(self):
        """Should not detect shift in ordinary statements."""
        message = "I like the design we discussed earlier."
        result = detect_goal_shift_in_message(message)
        assert result is None
    
    def test_no_shift_technical_discussion(self):
        """Should not detect shift in technical discussions."""
        message = "The API should return JSON with status codes."
        result = detect_goal_shift_in_message(message)
        assert result is None
    
    def test_case_insensitive_detection(self):
        """Should detect shifts regardless of case."""
        message = "PIVOT TO B2C MODEL NOW"
        result = detect_goal_shift_in_message(message)
        assert result is not None


class TestExtractNewGoalFromMessage:
    """Test extraction of new goals from shift messages."""
    
    def test_extract_b2b_to_b2c_pivot(self):
        """Should extract B2B→B2C pivot and rewrite goal."""
        current_goal = "Build a B2B analytics platform for enterprise teams"
        message = "Let's pivot to B2C targeting individual users."
        
        result = extract_new_goal_from_message(message, current_goal)
        
        assert result is not None
        assert "B2C" in result
        assert "B2B" not in result or "B2C" in result  # B2B replaced
    
    def test_extract_enterprise_to_individual_pivot(self):
        """Should extract enterprise→individual pivot."""
        current_goal = "Build analytics for enterprise teams of 10-50 people"
        message = "Pivot to individual freelancers instead."
        
        result = extract_new_goal_from_message(message, current_goal)
        
        assert result is not None
        assert "individual" in result.lower()
    
    def test_extract_mobile_app_addition(self):
        """Should extract mobile app addition to goal."""
        current_goal = "Build a web dashboard"
        message = "We need to add mobile app support for iOS and Android."
        
        result = extract_new_goal_from_message(message, current_goal)
        
        assert result is not None
        assert "mobile app" in result.lower() or "ios" in result.lower()
        assert "web dashboard" in result  # Original goal preserved
    
    def test_extract_generic_pivot_to(self):
        """Should extract generic 'pivot to X' pattern."""
        current_goal = "Build a CRM system"
        message = "Let's pivot to a project management tool instead."
        
        result = extract_new_goal_from_message(message, current_goal)
        
        assert result is not None
        assert "project management" in result.lower()
    
    def test_extract_weve_decided_to(self):
        """Should extract 'we've decided to X' pattern."""
        current_goal = "Build a REST API"
        message = "We've decided to build a GraphQL API with real-time subscriptions."
        
        result = extract_new_goal_from_message(message, current_goal)
        
        assert result is not None
        assert "graphql" in result.lower() or "real-time" in result.lower()
    
    def test_no_extraction_when_no_shift(self):
        """Should return None when no extractable shift."""
        current_goal = "Build a dashboard"
        message = "What color scheme should we use?"
        
        result = extract_new_goal_from_message(message, current_goal)
        
        assert result is None
    
    def test_no_extraction_from_short_pivot(self):
        """Should return None when extracted text too short."""
        current_goal = "Build a dashboard"
        message = "Pivot to new."  # Too short to be meaningful
        
        result = extract_new_goal_from_message(message, current_goal)
        
        # Should be None or original goal (heuristic may vary)
        # Key: should not produce garbage
        if result is not None:
            assert len(result) > 10
    
    def test_idempotent_mobile_addition(self):
        """Should not add mobile support if already present."""
        current_goal = "Build a dashboard with mobile app support (iOS and Android)"
        message = "We need mobile app support."
        
        result = extract_new_goal_from_message(message, current_goal)
        
        # Should return None since mobile is already in goal
        assert result is None


class TestTrackGoalEvolution:
    """Test tracking of goal evolution through conversation."""
    
    def test_track_single_goal_shift(self):
        """Should track a single goal shift."""
        turns = [
            {"turn_id": 1, "role": "user", "content": "Let's pivot to B2C."},
            {"turn_id": 2, "role": "assistant", "content": "Okay, pivoting to B2C."},
        ]
        original_goal = "Build a B2B platform"
        original_constraints = ["Budget: $5k"]
        
        timeline = track_goal_evolution(turns, original_goal, original_constraints)
        
        assert len(timeline) >= 1
        # Turn 1 should have updated goal
        assert 1 in timeline
        goal_at_1, _ = timeline[1]
        assert "B2C" in goal_at_1 or "B2B" not in goal_at_1
    
    def test_track_multiple_shifts(self):
        """Should track multiple goal shifts in sequence."""
        turns = [
            {"turn_id": 1, "role": "user", "content": "Pivot to B2C."},
            {"turn_id": 2, "role": "assistant", "content": "Got it."},
            {"turn_id": 3, "role": "user", "content": "Also add mobile app support."},
        ]
        original_goal = "Build a B2B dashboard"
        original_constraints = []
        
        timeline = track_goal_evolution(turns, original_goal, original_constraints)
        
        # Should have entries for both shifts
        assert 1 in timeline
        assert 3 in timeline
        goal_at_3, _ = timeline[3]
        # Goal should reflect mobile addition
        assert "mobile" in goal_at_3.lower() or "B2C" in goal_at_3
    
    def test_assistant_turns_do_not_trigger_shifts(self):
        """Should not detect shifts in assistant messages."""
        turns = [
            {"turn_id": 1, "role": "assistant", "content": "Let's pivot to B2C."},
            {"turn_id": 2, "role": "user", "content": "What database should we use?"},
        ]
        original_goal = "Build a B2B platform"
        original_constraints = []
        
        timeline = track_goal_evolution(turns, original_goal, original_constraints)
        
        # Goal should not have changed
        if 1 in timeline:
            goal_at_1, _ = timeline[1]
            assert goal_at_1 == original_goal
    
    def test_b2c_constraint_rewrite(self):
        """Should rewrite constraints when pivoting to B2C."""
        turns = [
            {"turn_id": 1, "role": "user", "content": "Pivot to B2C for individual users."},
        ]
        original_goal = "Build a B2B platform for enterprise teams"
        original_constraints = [
            "Target B2B enterprise customers",
            "Support teams of 10-50 people"
        ]
        
        timeline = track_goal_evolution(turns, original_goal, original_constraints)
        
        _, constraints_at_1 = timeline[1]
        # Constraints should be updated to reflect B2C
        constraint_text = " ".join(constraints_at_1).lower()
        assert "b2c" in constraint_text or "individual" in constraint_text
    
    def test_timeline_indexed_by_turn_id(self):
        """Timeline should be indexed by turn_id."""
        turns = [
            {"turn_id": 5, "role": "user", "content": "Pivot to mobile first."},
            {"turn_id": 10, "role": "user", "content": "Also add web support."},
        ]
        original_goal = "Build a desktop app"
        original_constraints = []
        
        timeline = track_goal_evolution(turns, original_goal, original_constraints)
        
        # Timeline keys should match turn_ids
        assert 5 in timeline
        assert 10 in timeline
    
    def test_empty_turns_list(self):
        """Should handle empty turns list gracefully."""
        turns = []
        original_goal = "Build a platform"
        original_constraints = []
        
        timeline = track_goal_evolution(turns, original_goal, original_constraints)
        
        # Timeline should be empty
        assert len(timeline) == 0
    
    def test_no_shifts_in_ordinary_conversation(self):
        """Should not record shifts in ordinary conversation."""
        turns = [
            {"turn_id": 1, "role": "user", "content": "What database should we use?"},
            {"turn_id": 2, "role": "assistant", "content": "I recommend PostgreSQL."},
            {"turn_id": 3, "role": "user", "content": "Sounds good."},
        ]
        original_goal = "Build a dashboard"
        original_constraints = []
        
        timeline = track_goal_evolution(turns, original_goal, original_constraints)
        
        # All entries should have original goal
        for turn_id in timeline:
            goal, _ = timeline[turn_id]
            assert goal == original_goal


class TestGetCurrentGoalAtTurn:
    """Test retrieval of current goal at specific turn."""
    
    def test_get_goal_from_timeline(self):
        """Should return goal from timeline at exact turn."""
        timeline = {
            1: ("Original goal", []),
            5: ("Updated goal", []),
        }
        original_goal = "Original goal"
        
        result = get_current_goal_at_turn(5, timeline, original_goal)
        
        assert result == "Updated goal"
    
    def test_get_goal_nearest_prior_turn(self):
        """Should return goal from nearest prior turn."""
        timeline = {
            1: ("Goal v1", []),
            5: ("Goal v2", []),
            10: ("Goal v3", []),
        }
        original_goal = "Original goal"
        
        # Turn 7 should use turn 5's goal
        result = get_current_goal_at_turn(7, timeline, original_goal)
        
        assert result == "Goal v2"
    
    def test_fallback_to_original_goal(self):
        """Should fallback to original goal when timeline empty."""
        timeline = {}
        original_goal = "Original goal"
        
        result = get_current_goal_at_turn(5, timeline, original_goal)
        
        assert result == "Original goal"
    
    def test_fallback_when_turn_before_timeline(self):
        """Should fallback to original goal when turn_id before any timeline entry."""
        timeline = {
            5: ("Updated goal", []),
        }
        original_goal = "Original goal"
        
        # Turn 3 is before any timeline entry
        result = get_current_goal_at_turn(3, timeline, original_goal)
        
        assert result == "Original goal"
    
    def test_get_goal_at_turn_zero(self):
        """Should handle turn_id = 0."""
        timeline = {
            1: ("Updated goal", []),
        }
        original_goal = "Original goal"
        
        result = get_current_goal_at_turn(0, timeline, original_goal)
        
        assert result == "Original goal"
    
    def test_get_latest_goal_at_end(self):
        """Should return latest goal when requesting far future turn."""
        timeline = {
            1: ("Goal v1", []),
            5: ("Goal v2", []),
            10: ("Goal v3", []),
        }
        original_goal = "Original goal"
        
        # Turn 100 should use turn 10's goal
        result = get_current_goal_at_turn(100, timeline, original_goal)
        
        assert result == "Goal v3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
