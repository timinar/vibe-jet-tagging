#!/usr/bin/env python3
"""
Add debug logging to LocalLLMClassifier for event loop debugging.

This script shows the exact modifications to add to local_llm_classifier.py
for detailed debugging output.

DO NOT run this script - it's a reference for manual edits.
"""

print("""
To enable debug logging, add these modifications to local_llm_classifier.py:

1. At the top of the file, add:
   import sys

2. In _get_or_create_event_loop(), add logging:

   def _get_or_create_event_loop(self) -> asyncio.AbstractEventLoop:
       if self._event_loop is None or self._event_loop.is_closed():
           print(f"[DEBUG] Creating NEW event loop (clf id={id(self)})", file=sys.stderr)
           self._event_loop = asyncio.new_event_loop()
           asyncio.set_event_loop(self._event_loop)
       else:
           print(f"[DEBUG] Reusing event loop (clf id={id(self)})", file=sys.stderr)

       print(f"[DEBUG] Loop: {self._event_loop}, closed={self._event_loop.is_closed()}", file=sys.stderr)
       return self._event_loop

3. In async_client property, add logging:

   @property
   def async_client(self) -> AsyncOpenAI:
       if self._async_client is None:
           print(f"[DEBUG] Creating async client (clf id={id(self)})", file=sys.stderr)
           self._async_client = AsyncOpenAI(
               base_url=self.base_url,
               api_key=self.api_key,
           )
       return self._async_client

4. In __del__, add logging:

   def __del__(self):
       print(f"[DEBUG] __del__ called (clf id={id(self)})", file=sys.stderr)
       if self._event_loop:
           print(f"[DEBUG]   Loop closed: {self._event_loop.is_closed()}", file=sys.stderr)
       if self._async_client:
           print(f"[DEBUG]   Async client exists: {self._async_client}", file=sys.stderr)

5. In _run_with_persistent_loop(), add logging:

   def _run_with_persistent_loop(self, X, verbose, max_concurrent):
       print(f"[DEBUG] START _run_with_persistent_loop (clf id={id(self)})", file=sys.stderr)
       loop = self._get_or_create_event_loop()

       result = loop.run_until_complete(
           self._predict_async(X, verbose=verbose, max_concurrent=max_concurrent)
       )

       print(f"[DEBUG] END _run_with_persistent_loop (loop closed={loop.is_closed()})", file=sys.stderr)
       return result

Then run:
   uv run python scripts/debug_event_loop.py 2>&1 | tee debug_output.log

Look for:
- When loops are created vs reused
- When __del__ is called relative to new classifier creation
- Whether the loop is closed when __del__ tries to use it
""")
