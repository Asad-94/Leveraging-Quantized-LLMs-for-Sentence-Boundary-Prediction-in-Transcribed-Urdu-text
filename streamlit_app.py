import re
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def load_model():
    model_name_or_path = "TheBloke/vicuna-13B-v1.5-GPTQ"
    config = AutoConfig.from_pretrained(model_name_or_path)
    config.quantization_config["disable_exllama"] = True
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 config=config,
                                                 device_map="auto",
                                                 trust_remote_code=False,
                                                 revision="main")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    return model, tokenizer

def generate_punctuated_sentence(test_sentence, model, tokenizer):
    prompt_template = f'''### Instruction: Add appropriate punctuations and sentence boundaries to the following Urdu text in input. Don't include any kind of html tags and don't truncate the output. Provide the actual input that has been asked. "،" "۔" These are the punctuations that need to be added
Input: "دنیا اس کے لئے مسخر کر دی گئی ہے لیکن ظلم و ستم کرنے اپنی حد سے تجاوز کرنے اور فضول خرچی کرنے سے منع بھی کیا گیا ہے"
Response: "دنیا اس کے لئے مسخر کر دی گئی ہے، لیکن ظلم و ستم کرنے، اپنی حد سے تجاوز کرنے اور فضول خرچی کرنے سے منع بھی کیا گیا ہے۔"

### Instruction: Add appropriate punctuations and sentence boundaries to the following Urdu text in input. Don't include any kind of html tags and don't truncate the output. Provide the actual input that has been asked. "،" "۔" These are the punctuations that need to be added
\n### Input: \"{test_sentence}\"
\n### Response:\n
'''

    input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
    output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=256)
    return tokenizer.decode(output[0])

def main():
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

    st.title("Urdu Punctuation Prediction")

    if not st.session_state.model_loaded:
        st.write("Loading model...")
        st.session_state.model, st.session_state.tokenizer = load_model()
        st.session_state.model_loaded = True
        st.write("Model loaded successfully!")

    # Input
    user_input = st.text_area("Please enter the unpunctuated Urdu sentence (maximum 30 words):", height=100)

    # Button to generate punctuated sentence
    if st.button("Generate Punctuated Sentence"):
        if user_input:
            st.markdown(f"### Input:")
            st.markdown(f"\"{user_input}\"")
            st.markdown("### Response:")
            punctuated_sentence = generate_punctuated_sentence(user_input, st.session_state.model, st.session_state.tokenizer)
            matching = re.search(r'### Response:[\s\S]*?(.+?)(?:(?:</s>)|\n\n|\n|### Input:|$)', punctuated_sentence)

            # Check if a match is found
            if matching:
                response = matching.group(1)
            else:
                response = punctuated_sentence
            st.write(response)
        else:
            st.warning("Please enter an Urdu sentence.")

if __name__ == "__main__":
    main()
