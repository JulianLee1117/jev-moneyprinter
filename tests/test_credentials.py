import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_alpha.credentials import ALLOWED_KEYS, openrouter_key, read_key


class CredentialTests(unittest.TestCase):
    def test_allowlisted_keys_are_read_independently_without_env_mutation(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path = Path(folder) / ".env"
            path.write_text("\n".join(f"{key}='synthetic-{key}-$VALUE'" for key in ALLOWED_KEYS))
            for key in ALLOWED_KEYS:
                self.assertEqual(read_key(key, path), f"synthetic-{key}-$VALUE")
                self.assertNotIn(key, os.environ)
            self.assertEqual(openrouter_key(path), read_key("OPENROUTER_API_KEY", path))

    def test_unknown_key_rejected_before_file_or_environment_access(self):
        with patch("jev_alpha.credentials.os.environ.get") as environ, patch("jev_alpha.credentials.Path") as file:
            for name in ("PATH", "AWS_SECRET_ACCESS_KEY", "APCA_API_SECRET_KEY\n", None):
                with self.assertRaises(ValueError):
                    read_key(name)
            environ.assert_not_called()
            file.assert_not_called()

    def test_each_environment_key_wins_and_blank_falls_back(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text("SCRY_API_KEY=synthetic-file-value")
            with patch.dict(os.environ, {"SCRY_API_KEY": " synthetic-environment "}, clear=True):
                self.assertEqual(read_key("SCRY_API_KEY", path), "synthetic-environment")
            with patch.dict(os.environ, {"SCRY_API_KEY": "  "}, clear=True):
                self.assertEqual(read_key("SCRY_API_KEY", path), "synthetic-file-value")

    def test_named_secret_errors_never_include_secret(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path = Path(folder) / ".env"
            for key in ("SCRY_API_KEY", "APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
                for entry in (f"{key}=synthetic-private\n{key}=second", f'{key}="synthetic-private', f"{key}=synthetic-private\tvalue"):
                    path.write_text(entry)
                    with self.assertRaises(ValueError) as error:
                        read_key(key, path)
                    self.assertNotIn("synthetic-private", str(error.exception))

    def test_environment_wins_without_file_access(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "synthetic-test-value"}):
            self.assertEqual(openrouter_key("does-not-exist"), "synthetic-test-value")

    def test_quotes_comments_and_export_without_interpolation(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path = Path(folder) / ".env"
            path.write_text("OTHER=ignored\nexport OPENROUTER_API_KEY='synthetic-$VALUE' # local\n")
            self.assertEqual(openrouter_key(path), "synthetic-$VALUE")
            self.assertNotIn("OPENROUTER_API_KEY", os.environ)

    def test_duplicate_and_malformed_errors_do_not_reveal_value(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path = Path(folder) / ".env"
            for content in ("OPENROUTER_API_KEY=synthetic-private\nOPENROUTER_API_KEY=second", 'OPENROUTER_API_KEY="synthetic-private'):
                path.write_text(content)
                with self.assertRaises(ValueError) as error:
                    openrouter_key(path)
                self.assertNotIn("synthetic-private", str(error.exception))


if __name__ == "__main__":
    unittest.main()
