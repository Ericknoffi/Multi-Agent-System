from Middlewares.pii import redact_pii

def pii_middleware(node_func):

    async def wrapper(state):

        if "user_query" in state:
            state["user_query"] = redact_pii(
                state["user_query"]
            )

        result = await node_func(state)

        if "final_response" in result:
            result["final_response"] = redact_pii(
                result["final_response"]
            )

        return result

    return wrapper