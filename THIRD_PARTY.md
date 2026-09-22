# Third-party notices

| Component | Use | License |
|---|---|---|
| [SemIf](https://github.com/TheoLeeCJ/SemIf) | Decision prompt format (`direct-options-v1`) and option-logit readout method, adapted in `src/llav/prompt.py` and `src/llav/engine.py` | MIT, Copyright (c) 2026 TheoLeeCJ |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Runtime (`llama-server`), installed separately, not bundled | MIT |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Default model, downloaded separately as a GGUF quantization by [bartowski](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF) | Apache-2.0 |
| [TypeSafe System One API](https://docs.typesafe.ai/api) | Public request/response shape that llav's API follows; no TypeSafe code, model or data is used | — |

SemIf's MIT license text:

```
MIT License

Copyright (c) 2026 TheoLeeCJ

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
