# Python AI Functions


Simple library that can convert from python functions to a JSON schema description of those functions, suitable for use with AI libraries.


## Installation

`pip install ai-functions`

## Usage

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
## What stuff does this handle?

 - Converts your annotated schema into an appropriate prompt 
 - Handles converting arguments to JSON if they are specified as a string.
 - Auto-casts arguments to the right types, if they aren't right.
 - Raises errors that AI engines understand if returned as a function response, instead of errors with poor descriptions.

## Async execute
 - If a loop is provided to the AIFunctions constructor, or to any execute calls, it will be used to schedule a coroutine.
 - Async versions of execute are available, prefix all calls with `async_`


### Some fine print

If you want to have a paramter that is still seen as "valid", but isn't part of the schema, you can 
annotate it with None as the description.  But this is really an "enforcement" thing, and might not
belong in this library.

Dealing with context is a beast in chat apps, so more work here might be helpful.

For example, I use meta-functions that unlock others, to prevent context-bloat.  

Might put that in another lib soon, or put it here.


