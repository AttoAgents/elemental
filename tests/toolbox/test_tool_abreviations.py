"""
Test abbreviations resolver tool.
"""

import json
import os
import unittest

from elemental_agents.tools.abbreviations import (
    AbbreviationsResolver,
    AbbreviationsResolverInitParameters,
    AbbreviationsResolverParams,
    AbbreviationsResolverResult,
)
from elemental_agents.utils.exceptions import ToolException


class TestToolAbbreviations(unittest.TestCase):
    """
    Test abbreviations resolver tool for resolving abbreviations to their full
    form from reading a json file.
    """

    def test_abbreviations_resolver(self) -> None:
        """
        Test abbreviations resolver.
        """

        data = {"CCSD": "Coupled Cluster with Single and Double excitations"}

        # Save the data to a file
        with open("abbreviations-testing.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

        init_params = AbbreviationsResolverInitParameters(
            source_file="abbreviations-testing.json"
        )
        abbreviations_resolver = AbbreviationsResolver(init_params)

        abbreviations_resolver_json = '{"abbreviation": "CCSD"}'
        params = json.loads(abbreviations_resolver_json)
        result = abbreviations_resolver.run(AbbreviationsResolverParams(**params))

        expected = AbbreviationsResolverResult(
            result="Coupled Cluster with Single and Double excitations"
        )

        self.assertEqual(result.result, expected.result)

        # remove the file
        os.remove("abbreviations-testing.json")

    def test_abbreviations_resolver_error(self) -> None:
        """
        Test abbreviations resolver error.
        """

        init_params = AbbreviationsResolverInitParameters(
            source_file="abbreviations-testing-2.json"
        )
        abbreviations_resolver = AbbreviationsResolver(init_params=init_params)

        abbreviations_resolver_json = '{"abbreviation": "etc"}'
        params = json.loads(abbreviations_resolver_json)

        with self.assertRaises(ToolException):
            abbreviations_resolver.run(AbbreviationsResolverParams(**params))


if __name__ == "__main__":
    unittest.main()
