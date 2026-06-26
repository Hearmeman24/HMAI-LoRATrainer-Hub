"""Tests for output file renaming convention — rename_output + sanitize_trigger_word.

Covers: all 7 in-scope models, high/low variant suffix for wan2.2,
sanitize edge cases, and the exact filename contract preserved from DESIGN §B7.
"""

import pytest

from config import rename_output, sanitize_trigger_word


class TestRenameOutput:
    # wan2.2 with noise variants
    def test_wan22_high(self):
        result = rename_output("daniella01", "wan2.2", "high", 80)
        assert result == "daniella01_wan2.2high_epoch80.safetensors"

    def test_wan22_low(self):
        result = rename_output("daniella01", "wan2.2", "low", 80)
        assert result == "daniella01_wan2.2low_epoch80.safetensors"

    # All 7 in-scope models — no noise variant
    def test_sdxl(self):
        result = rename_output("daniella01", "sdxl", None, 100)
        assert result == "daniella01_sdxl_epoch100.safetensors"

    def test_qwen_image(self):
        result = rename_output("daniella01", "qwen_image", None, 80)
        assert result == "daniella01_qwen-image_epoch80.safetensors"

    def test_qwen_image_2512(self):
        result = rename_output("daniella01", "qwen_image_2512", None, 80)
        assert result == "daniella01_qwen-image-2512_epoch80.safetensors"

    def test_z_image(self):
        result = rename_output("daniella01", "z_image", None, 80)
        assert result == "daniella01_z-image_epoch80.safetensors"

    def test_ideogram4(self):
        result = rename_output("daniella01", "ideogram4", None, 60)
        assert result == "daniella01_ideogram-4_epoch60.safetensors"

    def test_flux_klein_9b(self):
        result = rename_output("daniella01", "flux_klein_9b", None, 50)
        assert result == "daniella01_flux-klein-9b_epoch50.safetensors"

    # Deleted models should NOT appear (absence of contract)
    # These are not tested here — they aren't in the registry.

    # Epoch values
    def test_epoch_zero(self):
        result = rename_output("tw", "sdxl", None, 0)
        assert "_epoch0.safetensors" in result

    def test_large_epoch(self):
        result = rename_output("tw", "sdxl", None, 1000)
        assert "_epoch1000.safetensors" in result

    # Trigger word sanitization in rename_output
    def test_trigger_word_spaces_to_underscores(self):
        result = rename_output("my trigger word", "sdxl", None, 100)
        assert "my_trigger_word" in result

    def test_trigger_word_special_chars_stripped(self):
        result = rename_output("test@word!", "sdxl", None, 100)
        assert "@" not in result
        assert "!" not in result

    def test_empty_noise_variant_no_suffix(self):
        result = rename_output("tw", "sdxl", "", 10)
        assert "epoch10.safetensors" in result
        # no variant suffix between branch and epoch
        assert "_sdxl_epoch10.safetensors" in result


class TestSanitizeTriggerWord:
    def test_spaces_to_underscores(self):
        assert sanitize_trigger_word("my trigger word") == "my_trigger_word"

    def test_special_chars_removed(self):
        result = sanitize_trigger_word("test@word#123!")
        assert "@" not in result
        assert "#" not in result
        assert "!" not in result

    def test_normal_word_unchanged(self):
        assert sanitize_trigger_word("daniella01") == "daniella01"

    def test_leading_trailing_spaces(self):
        assert sanitize_trigger_word("  test  ") == "test"

    def test_digits_only_preserved(self):
        result = sanitize_trigger_word("123")
        assert result == "123"

    def test_dots_preserved(self):
        result = sanitize_trigger_word("v1.5")
        assert "." in result

    def test_hyphens_preserved(self):
        result = sanitize_trigger_word("my-trigger")
        assert result == "my-trigger"

    def test_quotes_stripped(self):
        result = sanitize_trigger_word('sks "art" style')
        assert '"' not in result

    def test_colons_stripped(self):
        result = sanitize_trigger_word("colon:trigger")
        assert ":" not in result
