import litellm
import pytest


@pytest.fixture(autouse=True)
def _discard_routers(monkeypatch):
    """Discard the Routers a test builds.

    A Router stays in LiteLLM's global callback lists until discarded, and
    LiteLLM takes no new callbacks once those lists hold 30.
    """
    routers = []
    init = litellm.Router.__init__

    def record(self, *args, **kwargs):
        init(self, *args, **kwargs)
        routers.append(self)

    monkeypatch.setattr(litellm.Router, "__init__", record)
    yield
    for router in routers:
        router.discard()
