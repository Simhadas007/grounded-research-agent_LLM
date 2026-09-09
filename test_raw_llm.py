from agent import llm

response = llm.invoke(
    'Return only this JSON: {"ok": true}'
)

print("TYPE:")
print(type(response))

print("\nFULL RESPONSE:")
print(repr(response))

print("\nCONTENT:")
print(repr(getattr(response, "content", None)))

print("\nADDITIONAL KWARGS:")
print(repr(getattr(response, "additional_kwargs", None)))

print("\nRESPONSE METADATA:")
print(repr(getattr(response, "response_metadata", None)))