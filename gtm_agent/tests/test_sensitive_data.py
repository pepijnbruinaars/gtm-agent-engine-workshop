import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent import gtm_agent as agent


SENSITIVE_FIELDS = {"tax_id", "date_of_birth", "card_on_file", "credit_check_ref"}
PROSPECT_ID = "LEAD-TEST"
PROSPECT_RECORD = {
    "name": "Alex Example",
    "email": "alex@example.com",
    "billing_qualification": {
        "tax_id": "TEST-TAX-ID",
        "date_of_birth": "2000-01-01",
        "card_on_file": "TEST-CARD-ON-FILE",
        "credit_check_ref": "TEST-CREDIT-REF",
    },
    "disqualified": False,
    "annual_revenue": 50000000,
    "enrichment_source": "Example Enrichment",
    "engagement_history": [{"channel": "Referral"}],
    "account_details": [{"industry": "Software"}],
    "tech_stack": ["Snowflake", "AWS"],
}
OFFERING = {
    "required_tech_stack": ["Snowflake", "AWS"],
    "min_annual_revenue": 1000000,
    "description": "Example offering",
}


def keys_in(value):
    if isinstance(value, dict):
        return set(value).union(*(keys_in(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(keys_in(item) for item in value))
    return set()


class SensitiveDataTests(unittest.TestCase):
    def setUp(self):
        data_service._PROFILES.pop(PROSPECT_ID, None)

    def tearDown(self):
        data_service._PROFILES.pop(PROSPECT_ID, None)

    def test_get_prospect_returns_only_contact_fields(self):
        with patch.object(data_service, "get_prospect_record", return_value=PROSPECT_RECORD):
            result = agent.get_prospect.invoke({"prospect_id": PROSPECT_ID})

        contact = result["prospect"]
        self.assertEqual(contact, {
            "prospect_id": PROSPECT_ID,
            "name": "Alex Example",
            "email": "alex@example.com",
        })
        self.assertTrue(SENSITIVE_FIELDS.isdisjoint(keys_in(result)))
        self.assertNotIn("billing_qualification", contact)

        sent = agent.send_prospect_email.func(
            contact,
            "Hello",
            "Following up",
            SimpleNamespace(config={"metadata": {}}),
            from_rep={"name": "Taylor Rep", "email": "taylor@example.com"},
        )
        self.assertEqual(sent["status"], "sent")
        self.assertEqual(sent["to"], "alex@example.com")

    def test_build_profile_excludes_billing_data_from_result_and_cache(self):
        with (
            patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(data_service, "get_prospect_record", return_value=PROSPECT_RECORD),
            patch.object(data_service, "fetch_engagement_history", return_value=PROSPECT_RECORD["engagement_history"]),
            patch.object(data_service, "fetch_account_details", return_value=PROSPECT_RECORD["account_details"]),
            patch.object(data_service, "fetch_tech_stack", return_value=PROSPECT_RECORD["tech_stack"]),
        ):
            result = agent.build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})

        profile = result["prospect_profile"]
        self.assertTrue(SENSITIVE_FIELDS.isdisjoint(keys_in(result)))
        self.assertNotIn("billing_qualification", profile)
        self.assertEqual(data_service._PROFILES[PROSPECT_ID], profile)
        self.assertEqual(profile["annual_revenue"], PROSPECT_RECORD["annual_revenue"])
        self.assertEqual(profile["account_details"], PROSPECT_RECORD["account_details"])
        self.assertEqual(profile["tech_stack"], PROSPECT_RECORD["tech_stack"])

    def test_build_profile_sanitizes_existing_cached_profile(self):
        cached = {"prospect_id": PROSPECT_ID, **PROSPECT_RECORD}
        data_service._PROFILES[PROSPECT_ID] = cached
        with patch.object(data_service, "get_profile_from_db", return_value={"prospect_profile": cached}):
            result = agent.build_prospect_profile.invoke({"prospect_id": PROSPECT_ID})

        self.assertTrue(SENSITIVE_FIELDS.isdisjoint(keys_in(result)))
        self.assertNotIn("billing_qualification", data_service._PROFILES[PROSPECT_ID])

    def test_score_sends_only_scoring_fields_to_nested_model(self):
        captured = {}

        class Result:
            def model_dump(self):
                return {"score": 90}

        def capture(messages):
            captured["messages"] = messages
            return Result()

        with patch.object(data_service, "fetch_tech_stack", return_value=["Snowflake"]), \
                patch.object(agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = capture
            result = agent.score_prospect.invoke({
                "prospect_profile": {
                    "prospect_id": PROSPECT_ID,
                    **PROSPECT_RECORD,
                    "account_details": PROSPECT_RECORD["account_details"],
                    "tech_stack": PROSPECT_RECORD["tech_stack"],
                },
                "offering": OFFERING,
            })

        user = captured["messages"][1]["content"]
        serialized_profile = json.loads(user.split("Prospect profile:\n", 1)[1])
        self.assertEqual(result["score"], 90)
        self.assertEqual(set(serialized_profile), {
            "prospect_id", "annual_revenue", "account_details", "tech_stack",
        })
        self.assertTrue(SENSITIVE_FIELDS.isdisjoint(keys_in(serialized_profile)))
        self.assertNotIn("billing_qualification", user)
        self.assertEqual(serialized_profile["tech_stack"], ["Snowflake"])


if __name__ == "__main__":
    unittest.main()
