import asyncio
import json
import threading
from typing import Annotated

import pytest
from ai_functions import AIFunctions


@pytest.fixture(scope="session")
def external_loop():
    loop = asyncio.new_event_loop()

    def run():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    yield loop


# Example test data
def example_func1(x: Annotated[int, "x val"]):
    """Adds 1 to x."""
    return x + 1


def example_func2(x: Annotated[int, "x val"], y: Annotated[str, "y val"]):
    """Adds x and int(y)."""
    return x + int(y)


@pytest.mark.asyncio
async def async_example_func1(x: Annotated[int, "x val"]):
    """Adds 1 to x."""
    return x + 1


def bad_func(x: int):
    return x + 1


@pytest.fixture
def ai_functions():
    # Initialize the AIFunctions instance with example functions
    funcs = [example_func1, example_func2]
    return AIFunctions(funcs=funcs)


def test_bad_func(ai_functions: AIFunctions):
    with pytest.raises(ValueError):
        ai_functions.add(bad_func)


def test_execute_function(ai_functions):
    # Test regular execution of a synchronous function
    result = ai_functions.execute("example_func1", json.dumps(dict(x=5)))
    assert result == "6"


def test_execute_function_no_convert(ai_functions):
    # Test regular execution of a synchronous function
    ai_functions.convert_output = False
    result = ai_functions.execute("example_func1", json.dumps(dict(x=5)))
    assert result == 6


def test_execute_function_auto_convert(ai_functions):
    # Test regular execution of a synchronous function
    result = ai_functions.execute("example_func1", json.dumps(dict(x="5")))
    assert result == "6"


def test_execute_function_loop(ai_functions, external_loop):
    ai_functions.add(async_example_func1)
    # Test execution of an asynchronous function using a loop
    result = ai_functions.execute("async_example_func1", {"x": 5}, loop=external_loop)
    assert result == "6"


def test_execute_function_no_loop(ai_functions, external_loop):
    ai_functions.add(async_example_func1)
    # Test execution of an asynchronous function using a loop
    with pytest.raises(ValueError):
        ai_functions.execute("async_example_func1", {"x": 5})


async def test_execute_async_function(ai_functions):
    # Test execution of an asynchronous function using a loop
    result = await ai_functions.async_execute("async_example_func1", 5)
    assert result == "6"


def test_execute_unknown_function(ai_functions):
    # Test executing an unknown function, which should raise KeyError
    with pytest.raises(KeyError):
        ai_functions.execute("unknown_function", 5)


def test_openai_dict(ai_functions):
    # Test generating an openai compatible function dict
    openai_dict = ai_functions.openai_dict()
    assert openai_dict == [
        {
            "name": "example_func1",
            "description": "Adds 1 to x.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "x val"}
                },
            },
        },
        {
            "name": "example_func2",
            "description": "Adds x and int(y).",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "x val"},
                    "y": {"type": "string", "description": "y val"},
                },
            },
        },
    ]


def test_openai_execute(ai_functions):
    # Test execution of a function using the openai style function_call
    call = {"name": "example_func2", "arguments": {"x": 2, "y": "4"}}
    result = ai_functions.openai_execute(call)
    assert result == "6"


def test_description(ai_functions):
    # Test getting the description of a function
    description = ai_functions.description("example_func1")
    assert description == "Adds 1 to x."
