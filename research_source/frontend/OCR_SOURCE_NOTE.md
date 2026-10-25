The historical Ropes frontend uses frozen Qwen2.5-VL-3B text transcription.
The external OCR model is not included. Recorded OCR facts are available in
the frozen fact records. The fixed prompt was:

```python
PROMPT='Transcribe ONLY the uppercase color word printed on the white rectangular label. Read the printed letters. Do not infer the word from the rope color. Return only the printed word. If the printed label cannot be read, return UNREADABLE.'
```
