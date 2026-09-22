from langchain_core.messages import AIMessage

from urbanomy.land_value.ga_mc_optimizer import Evaluation, StrategicAlignmentScorer


class _LLM:
    def __init__(self, response):
        self._response = response

    def invoke(self, _prompt):
        return self._response


def _score(response):
    scorer = StrategicAlignmentScorer(llm=_LLM(response), prompt="strategy")
    return scorer.score_candidate(
        params_repaired={"l": 1.0}, land_value_gain=1.0, investor_npv=1.0
    )["score"]


def test_scores_langchain_ai_message():
    # ChatOpenAI.invoke returns an AIMessage whose .content holds the score JSON; an
    # AIMessage is itself a pydantic model, so it must not be dumped as if it were an Evaluation.
    assert _score(AIMessage(content='{"score": 0.7}')) == 0.7


def test_scores_structured_evaluation():
    assert _score(Evaluation(score=0.4)) == 0.4


def test_scores_mapping_and_text():
    assert _score({"score": 0.25}) == 0.25
    assert _score('{"score": 1.0}') == 1.0
