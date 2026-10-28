# Web Search Fixture

Pre-recorded web search response for SC-42 mock test.

File: `f35_search_response.json` — recorded from a live `web_search_handoff` call
with query='F-35 specifications'. Record by running SC-42/algorithmic once and
capturing the tool output from the test result JSON.

Until the live recording is done, SC-42 runs as a pure integration test (no mock).
The fixture file is a placeholder — once recorded, SC-42 can be split into a
deterministic unit test using this response.
