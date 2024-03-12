In applications converting speech to text, transcriptions often suffer from issues like punctuation errors and missing sentence boundaries, particularly in languages with fewer resources like Urdu. While previous attempts using rule-based and machine learning methods have had limited success, there has been no reported exploration using Large Language Models (LLMs). This project aims to address Urdu transcription challenges using LLMs, focusing on smaller models with 13 billion parameters or fewer due to resource constraints.


Various techniques have been employed to improve sentence boundary detection in Urdu text. For example, one-shot learning and fine-tuning approaches have been utilized to generate accurately punctuated Urdu sentences. LLMs such as Llama2, Vicuna, and Falcon were tested in one-shot learning experiments, with Vicuna 13b model yielding the most promising results.


However, fine-tuning the Llama2-7b model using the QLoRA method did not yield satisfactory outcomes. The resulting punctuated Urdu sentences did not consistently align with correct sentence structures and occasionally included non-Urdu words.

_______________________________________________________________________________________________________________________________________

### One-shot learning example:

Input Example (unpunctuated sentence):
مجھے سخت زمین والا وہ گندا گھر اچھا نہیں لگتا یہاں کتنی نرم گھاس ہے سونے میں کتنا مزہ آئے گا

Original Sentence (with punctuations):
مجھے سخت زمین والا وہ گندا گھر اچھا نہیں لگتا۔ یہاں کتنی نرم گھاس ہے، سونے میں کتنا مزہ آئے گا۔

LLM generated sentence (with predicted punctuations):
مجھے سخت زمین والا وہ گندا گھر اچھا نہیں لگتا، یہاں کتنی نرم گھاس ہے، سونے میں کتنا مزہ آئے گا۔
