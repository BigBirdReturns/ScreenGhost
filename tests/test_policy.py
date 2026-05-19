from core.policy import SafetyPolicy


def test_policy_allows_and_blocks():
    policy = SafetyPolicy(allowed_apps={"Home"}, blocked_actions={"delete_account"})
    assert policy.allows_app("Home")
    assert not policy.allows_app("Settings")
    assert policy.allows_action("toggle")
    assert not policy.allows_action("delete_account")
