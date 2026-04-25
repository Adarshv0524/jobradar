import unittest


class TestResumeParser(unittest.TestCase):
    def test_text_resume_decodes_and_normalizes(self):
        import os
        import sys

        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from resume_parser import parse_resume_bytes

        raw = b"Line 1\r\n\r\n\r\nLine 2\x00\x00\rLine 3\n\n\n\n"
        res = parse_resume_bytes(raw, content_type="text/plain", filename="resume.txt")
        self.assertEqual(res.method, "utf8")
        self.assertIn("Line 1", res.text)
        self.assertIn("Line 2", res.text)
        self.assertIn("Line 3", res.text)
        # Collapses excessive blanks (should not preserve 4+ empty lines)
        self.assertTrue("\n\n\n\n" not in res.text)


if __name__ == "__main__":
    unittest.main()
