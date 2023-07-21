# Python AI Functions


Simple library that can convert from python functions to a JSON schema description of those functions, suitable for use with AI libraries.


For example:

```
from ai_functions import get_openai_functions, execute_function


def search_web(query: Annotated[str, "google formatted keywords to search for"]):
    """Search the web"""

print(openai_function_dict([search_web]))
```


Also, if you get a `response.function_call` from openai, you can use `execute_function`

```
openai_function_execute([search_web], function_call)
```

Or, if you have the name and arguments split out already:

```
function_execute([search_web], name, arguments)
```

Finally, if you want a container to handle this:

```
from ai_functions import AIFunctions


container = AIFunctions([search_web, add_calendar_entry])

functions = container.openai_dict()
subset_functions = container.openai_dict(["search_web"])
container.execute("search_web", {"query":"top web hosting sites"})
container.opeanai_execute({"name": "search_web", "arguments": "{\"query\":\"top web hosting sites\"}")

```


It handles converting arguments to JSON if they are specified as a string.

It auto-casts arguments to the right types, if they aren't right.

It returns errors that AI engines understands, instead of errors with poor descriptions.

If a loop is provided to the AIFunctions constructor, or to any execute calls, it will be used to schedule a coroutine.

Async versions of execute are available, prefix all calls with `async_`
