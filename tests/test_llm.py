import json

from backend.llm import complete


def test_llm_complete_parses_chat_completion(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "Fact-based answer"}}]}).encode()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("backend.llm.urlopen", lambda *_args, **_kwargs: Response())
    assert complete("What is recovered?", {"metrics": {"recovered": 3}}) == "Fact-based answer"
