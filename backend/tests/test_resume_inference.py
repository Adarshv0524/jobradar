import unittest


class TestResumeInference(unittest.TestCase):
    def test_senior_secondary_does_not_trigger_senior(self):
        import os
        import sys

        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if backend_dir not in sys.path:
            sys.path.insert(0, backend_dir)

        from resume_inference import infer_resume_experience

        text = """
        Education
        Senior Secondary (CBSE) - 92%
        B.Tech Computer Science - Expected Graduation 2026

        Experience
        Data Engineer Intern — 4 months
        Built ETL pipelines with Python and SQL.
        """
        res = infer_resume_experience(text)
        self.assertNotEqual(res.level, "senior")
        self.assertIn(res.level, ("intern", "junior"))


if __name__ == "__main__":
    unittest.main()

