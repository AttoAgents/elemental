"""
Test save evaluation of mathematical expressions for the calculator tool.
"""

import unittest

from elemental_agents.utils.safe_eval import safe_eval


class TestSafeEval(unittest.TestCase):
    """
    Test save evaluation of mathematical expressions.
    """

    def test_safe_eval(self) -> None:
        """
        Test the safe evaluation of mathematical expressions. Test expressions
        including operators and functions.
        """

        self.assertEqual(safe_eval("1 + 1"), 2)
        self.assertEqual(safe_eval("2 * 3"), 6)
        self.assertEqual(safe_eval("2 ** 3"), 8)
        self.assertEqual(safe_eval("sin(0)"), 0.0)
        self.assertEqual(safe_eval("cos(0)"), 1.0)
        self.assertEqual(safe_eval("tan(0)"), 0.0)
        self.assertEqual(safe_eval("sqrt(4)"), 2.0)
        self.assertEqual(safe_eval("1 + 2 * 3"), 7)
        self.assertEqual(safe_eval("2 ^ 3"), 8.0)
        self.assertEqual(safe_eval("sqrt(16)"), 4.0)

        self.assertAlmostEqual(safe_eval("log(10)"), 2.302585092994046, places=7)
        self.assertAlmostEqual(safe_eval("exp(1)"), 2.718281828459045, places=7)
        self.assertAlmostEqual(safe_eval("sin(0) + cos(0)"), 1.0, places=7)
        self.assertAlmostEqual(safe_eval("tan(0) + sqrt(4)"), 2.0, places=7)
        self.assertAlmostEqual(
            safe_eval("log(10) + exp(1)"), 5.020866921453091, places=7
        )
        self.assertAlmostEqual(
            safe_eval(
                "1 + 2 * 3 + sin(0) + cos(0) + tan(0) + sqrt(4) + log(10) + exp(1)"
            ),
            15.020866921453091,
            places=7,
        )
        self.assertAlmostEqual(safe_eval("sin(0)"), 0.0, places=7)
        self.assertAlmostEqual(safe_eval("cos(3.1415)"), -1.0, places=7)


if __name__ == "__main__":
    unittest.main()
