import pytest

from pydantic import TypeAdapter, ValidationError

from gempyor.sync._sync import SyncProtocols

@pytest.mark.parametrize(
    "protocols", [
    {
        "demorsync" : {
            'type': 'rsync', 'source' : '.', 'target' : 'host:~/some/path'
        },
        "demos3sync": {
            'type': 's3sync', 'source' : '.', 'target' : 'some/path'
        },
        "demogit" : {
            'type' : 'git'
        }
    },
    {
        "justone" : {
            'type' : 'git'
        }
    },
    {}
])
def test_successfully_construct_from_valid_protocols(protocols: dict):
    """
    Ensures SyncProtocols can instantiate valid objects
    """

    SyncProtocols(protocols=protocols)

@pytest.mark.parametrize(
    "protocols", [
    {
        "demorsync" : {
            'type': 'unsupported', 'source' : '.', 'target' : 'host:~/some/path'
        }
    },
    {
        "missingtar" : {
            'type' : 'rsync', "source" : "."
        }
    },
    {
        "badgit" : {
            'type' : 'git', "source" : "."
        }
    },
])
def test_fail_construct_from_invalid_protocols(protocols: dict):
    """
    Ensures SyncProtocols doesn't instantiate invalid objects
    """
    with pytest.raises(ValidationError):
        SyncProtocols(protocols=protocols)


# @pytest.mark.parametrize(
#     "data",
#     [
#         {'TriggerType': 'Scheduled'},
#         {'TriggerType': 'OnDemand', 'TriggerProperties': {'foo': 'bar'}},
#         {'TriggerType': 'Event', 'TriggerProperties': {'foo': 'bar'}}
#     ]
# )
# def test_trigger_config_invalid(data: dict):
#     """
#     Ensures TriggerConfig raises error when instantiating invalid objects
#     """

#     ta = TypeAdapter(TriggerConfig)
#     with pytest.raises(ValidationError):
#         _ = ta.validate_python(data)