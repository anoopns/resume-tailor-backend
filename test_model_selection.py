import os
import unittest
from unittest.mock import Mock, call, patch

from fastapi.testclient import TestClient
from google.genai import errors

from prompts import (
    COVER_LETTER_SYSTEM_INSTRUCTION,
    COVER_LETTER_USER_PROMPT,
    RESUME_SYSTEM_INSTRUCTION,
    RESUME_USER_PROMPT,
)

with (
    patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "auto"}),
    patch("dotenv.load_dotenv"),
    patch("google.genai.Client"),
):
    import main


def busy_error():
    return errors.ServerError(503, {"error": {"message": "High demand", "status": "UNAVAILABLE"}})


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.sdk = Mock()
        self.sleep = Mock()
        self.response = Mock(text="# Tailored document")
        for patcher in (
            patch.object(main, "client", self.sdk),
            patch.object(main.time, "sleep", self.sleep),
            patch.object(main, "GEMINI_AUTO_MODELS", ("primary", "fallback")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def generate(self, model="auto"):
        return main.generate_with_retry(model, "resume content", "system instruction")

    def selected_models(self):
        return [item.kwargs["model"] for item in self.sdk.models.generate_content.call_args_list]

    def test_auto_primary_success_preserves_prompt_and_config(self):
        self.sdk.models.generate_content.return_value = self.response
        self.assertIs(self.generate(), self.response)
        self.assertEqual(self.selected_models(), ["primary"])
        kwargs = self.sdk.models.generate_content.call_args.kwargs
        self.assertEqual(kwargs["contents"], "resume content")
        self.assertEqual(kwargs["config"].system_instruction, "system instruction")
        self.assertEqual(kwargs["config"].temperature, 0.2)
        self.sleep.assert_not_called()

    def test_busy_primary_recovers_without_fallback(self):
        self.sdk.models.generate_content.side_effect = [busy_error(), self.response]
        self.assertIs(self.generate(), self.response)
        self.assertEqual(self.selected_models(), ["primary", "primary"])
        self.sleep.assert_called_once_with(1)

    def test_auto_falls_back_after_three_busy_attempts(self):
        self.sdk.models.generate_content.side_effect = [busy_error()] * 3 + [self.response]
        self.assertIs(self.generate(), self.response)
        self.assertEqual(self.selected_models(), ["primary"] * 3 + ["fallback"])
        self.assertEqual(self.sleep.call_args_list, [call(1), call(2)])

    def test_all_busy_is_bounded_and_raises_last_error(self):
        error = busy_error()
        self.sdk.models.generate_content.side_effect = error
        with self.assertRaises(errors.ServerError) as caught:
            self.generate()
        self.assertIs(caught.exception, error)
        self.assertEqual(self.selected_models(), ["primary"] * 3 + ["fallback"] * 3)
        self.assertEqual(self.sleep.call_args_list, [call(1), call(2)] * 2)

    def test_explicit_model_never_falls_back(self):
        self.sdk.models.generate_content.side_effect = busy_error()
        with self.assertRaises(errors.ServerError):
            self.generate("chosen-model")
        self.assertEqual(self.selected_models(), ["chosen-model"] * 3)

    def test_non_busy_api_errors_are_not_retried_or_routed(self):
        for code in (400, 401, 403, 404, 429, 500):
            with self.subTest(code=code):
                self.sdk.reset_mock()
                error_class = errors.ClientError if code < 500 else errors.ServerError
                error = error_class(code, {"error": {"message": "Request failed"}})
                self.sdk.models.generate_content.side_effect = error
                with self.assertRaises(error_class) as caught:
                    self.generate()
                self.assertIs(caught.exception, error)
                self.assertEqual(self.selected_models(), ["primary"])
        self.sleep.assert_not_called()

    def test_unexpected_error_is_not_retried(self):
        self.sdk.models.generate_content.side_effect = ValueError("Invalid request")
        with self.assertRaises(ValueError):
            self.generate()
        self.assertEqual(self.selected_models(), ["primary"])
        self.sleep.assert_not_called()

    def test_both_endpoints_use_auto_and_preserve_response_shape(self):
        with (
            TestClient(main.app) as client,
            patch.object(main, "GEMINI_MODEL", "auto"),
            patch.object(main, "get_or_parse_resume", return_value=("Resume text", False)),
            patch.object(main, "get_cached_tailored_result", return_value=None),
            patch.object(main, "save_tailored_result") as save,
        ):
            for output in ("resume", "cover-letter"):
                with self.subTest(output=output):
                    self.sdk.reset_mock()
                    self.sdk.models.generate_content.side_effect = [busy_error()] * 3 + [self.response]
                    response = client.post(
                        f"/api/tailor-{output}",
                        files={"resume_file": ("resume.pdf", b"test PDF", "application/pdf")},
                        data={"jd_source": "Software engineer"},
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), {"markdown": "# Tailored document", "cached": False})
                    self.assertEqual(self.selected_models(), ["primary"] * 3 + ["fallback"])
                    self.assertEqual(save.call_args.args[-1], output)

    def test_both_endpoints_return_503_when_all_models_busy(self):
        with (
            TestClient(main.app) as client,
            patch.object(main, "GEMINI_MODEL", "auto"),
            patch.object(main, "get_or_parse_resume", return_value=("Resume text", False)),
            patch.object(main, "get_cached_tailored_result", return_value=None),
            patch.object(main, "save_tailored_result") as save,
        ):
            self.sdk.models.generate_content.side_effect = busy_error()
            for output in ("resume", "cover-letter"):
                with self.subTest(output=output):
                    self.sdk.reset_mock()
                    response = client.post(
                        f"/api/tailor-{output}",
                        files={"resume_file": ("resume.pdf", b"test PDF", "application/pdf")},
                        data={"jd_source": "Software engineer"},
                    )
                    self.assertEqual(response.status_code, 503)
                    self.assertIn("UNAVAILABLE", response.json()["detail"])
                    self.assertEqual(self.selected_models(), ["primary"] * 3 + ["fallback"] * 3)
            save.assert_not_called()

    def test_both_endpoints_render_external_prompts_with_literal_braces(self):
        resume_text = "Resume with {braces} and Unicode: \u00e9"
        jd_text = "Job with {requirements}"
        extra_context = "Context with {details}"
        target_context = (
            f"TARGET JOB DESCRIPTION:\n---\n{jd_text}\n---\n\n"
            f"ADDITIONAL CONTEXT:\n---\n{extra_context}\n---"
        )
        self.sdk.models.generate_content.return_value = self.response
        with (
            TestClient(main.app) as client,
            patch.object(main, "get_or_parse_resume", return_value=(resume_text, False)),
            patch.object(main, "get_cached_tailored_result", return_value=None),
            patch.object(main, "save_tailored_result"),
        ):
            for output, system_instruction, template in (
                ("resume", RESUME_SYSTEM_INSTRUCTION, RESUME_USER_PROMPT),
                ("cover-letter", COVER_LETTER_SYSTEM_INSTRUCTION, COVER_LETTER_USER_PROMPT),
            ):
                with self.subTest(output=output):
                    response = client.post(
                        f"/api/tailor-{output}",
                        files={"resume_file": ("resume.pdf", b"test PDF", "application/pdf")},
                        data={"jd_source": jd_text, "additional_context": extra_context},
                    )
                    self.assertEqual(response.status_code, 200)
                    kwargs = self.sdk.models.generate_content.call_args.kwargs
                    self.assertEqual(kwargs["config"].system_instruction, system_instruction)
                    self.assertEqual(
                        kwargs["contents"],
                        template.format(master_resume_text=resume_text, target_context=target_context),
                    )
                    self.assertIn(resume_text, kwargs["contents"])
                    self.assertIn(target_context, kwargs["contents"])

    def test_resume_prompt_forbids_star_labels(self):
        rules = RESUME_SYSTEM_INSTRUCTION + RESUME_USER_PROMPT
        self.assertIn("NEVER label or split bullets into STAR parts", rules)
        self.assertNotIn("STAR-format bullet points", rules)

    def test_prompt_change_invalidates_tailored_cache(self):
        original = main.get_tailored_output_cache_file("pdf", "jd", "ctx", "resume")
        self.assertNotEqual(original, main.get_tailored_output_cache_file("pdf", "jd", "ctx", "cover-letter"))
        with patch.dict(main.PROMPT_HASHES, {"resume": "f" * 64}):
            self.assertNotEqual(original, main.get_tailored_output_cache_file("pdf", "jd", "ctx", "resume"))

    def test_cache_hit_skips_generation(self):
        with (
            TestClient(main.app) as client,
            patch.object(main, "get_or_parse_resume", return_value=("Resume text", True)),
            patch.object(main, "get_cached_tailored_result", return_value="Cached document"),
        ):
            for output in ("resume", "cover-letter"):
                response = client.post(
                    f"/api/tailor-{output}",
                    files={"resume_file": ("resume.pdf", b"test PDF", "application/pdf")},
                    data={"jd_source": "Software engineer"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"markdown": "Cached document", "cached": True})
            self.sdk.models.generate_content.assert_not_called()


if __name__ == "__main__":
    unittest.main()
