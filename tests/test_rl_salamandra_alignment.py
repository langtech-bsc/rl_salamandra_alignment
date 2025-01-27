#!/usr/bin/env python

"""Tests for `rl_salamandra_alignment` package."""


# Replace `your_module` with the actual module name where `unfold_dict` is defined
from rl_salamandra_alignment.utils.general import unfold_dict, dict_sort
import unittest
import json


class TestRl_salamandra_alignment(unittest.TestCase):
    """Tests for `rl_salamandra_alignment` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""


class TestUnfoldDict(unittest.TestCase):
    """Tests for `unfold_dict` function."""

    def setUp(self):
        """Set up test fixtures."""
        self.simple_input = {
            "key1": "value1",
            "key2": [1, 2],
            "key3": "value3",
        }

        self.nested_input = {
            "key1": "value1",
            "key2": [1, 2],
            "key3": {
                "subkey1": "subvalue1",
                "subkey2": [10, 20],
            },
        }

        self.empty_input = {}

        self.single_list_input = {
            "key1": [1, 2, 3]
        }

        self.complex_input = {
            "key1": "value1",
            "key2": [1, 2],
            "key3": {
                "subkey1": ["a", "b"],
                "subkey2": [10, 20],
            },
            "key4": ["x", "y"],
        }

    def test_simple_input(self):
        """Test with a simple dictionary containing one list."""
        result = unfold_dict(self.simple_input)
        expected = dict_sort([
            {"key1": "value1", "key2": 1, "key3": "value3"},
            {"key1": "value1", "key2": 2, "key3": "value3"},
        ])
        self.assertEqual(result, expected)

    def test_nested_input(self):
        """Test with a nested dictionary containing lists."""
        result = unfold_dict(self.nested_input)
        expected = dict_sort([
            {"key1": "value1", "key2": 1, "key3": {
                "subkey1": "subvalue1", "subkey2": 10}},
            {"key1": "value1", "key2": 1, "key3": {
                "subkey1": "subvalue1", "subkey2": 20}},
            {"key1": "value1", "key2": 2, "key3": {
                "subkey1": "subvalue1", "subkey2": 10}},
            {"key1": "value1", "key2": 2, "key3": {
                "subkey1": "subvalue1", "subkey2": 20}},
            ]
        )
        self.assertEqual(result, expected)

    def test_empty_input(self):
        """Test with an empty dictionary."""
        result = unfold_dict(self.empty_input)
        expected = [{}]
        self.assertEqual(result, expected)

    def test_single_list_input(self):
        """Test with a dictionary that has only one list."""
        result = unfold_dict(self.single_list_input)
        expected = dict_sort([
            {"key1": 1},
            {"key1": 2},
            {"key1": 3},
        ])
        self.assertEqual(result, expected)

    def test_complex_input(self):
        """Test with a complex nested dictionary containing multiple lists."""
        result = unfold_dict(self.complex_input)
        expected_length = 16  # 2 (key2) * 2 (subkey1) * 2 (subkey2) * 2 (key4)
        self.assertEqual(len(result), expected_length)

    def test_no_list_or_dict(self):
        """Test with a dictionary containing no lists or nested dictionaries."""
        input_data = {"key1": "value1", "key2": "value2"}
        result = unfold_dict(input_data)
        expected = dict_sort([{"key1": "value1", "key2": "value2"}])
        self.assertEqual(result, expected)

    def test_key_with_empty_list(self):
        """Test with a dictionary where one key has an empty list."""
        input_data = {"key1": "value1", "key2": []}
        result = unfold_dict(input_data)
        expected = []  # No combinations possible when a list is empty
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
