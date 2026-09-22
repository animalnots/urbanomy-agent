from urbanomy_agent.llm import init_llm


def test_reasoning_effort_sets_extra_body(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("URBANOMY_LLM_MODEL", "m")
    monkeypatch.setenv("URBANOMY_REASONING_EFFORT", "low")
    assert init_llm(temperature=0).extra_body == {"reasoning": {"effort": "low"}}


def test_absent_reasoning_effort_leaves_extra_body_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("URBANOMY_LLM_MODEL", "m")
    monkeypatch.delenv("URBANOMY_REASONING_EFFORT", raising=False)
    assert not init_llm(temperature=0).extra_body
