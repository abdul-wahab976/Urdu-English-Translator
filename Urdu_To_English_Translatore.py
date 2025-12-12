import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_NAME = "Helsinki-NLP/opus-mt-ur-en"

# Custom dictionary for proper names & technical words
CUSTOM_DICT = {
    "عبدالوہاب": "Abdul Wahab",
    "ڈیٹا سائنس": "Data Science"
}


class TranslatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Urdu → English Translator")
        self.root.geometry("760x520")
        self.root.resizable(False, False)

        s = ttk.Style()
        s.configure("TButton", padding=6, relief="flat", font=("Segoe UI", 10))
        s.configure("TLabel", font=("Segoe UI", 10))

        top_frame = ttk.Frame(root, padding=(10, 10))
        top_frame.pack(fill="x")
        self.status_label = ttk.Label(top_frame, text="⏳ Model not loaded yet.")
        self.status_label.pack(side="left")

        btn_frame = ttk.Frame(root, padding=(10, 0))
        btn_frame.pack(fill="x")
        self.load_btn = ttk.Button(btn_frame, text="Load Model", command=self.start_model_load)
        self.load_btn.pack(side="left", padx=(0, 6))
        self.translate_btn = ttk.Button(btn_frame, text="Translate →", command=self.on_translate, state="disabled")
        self.translate_btn.pack(side="left", padx=(0, 6))
        self.clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear_texts)
        self.clear_btn.pack(side="left", padx=(0, 6))
        self.copy_btn = ttk.Button(btn_frame, text="Copy English", command=self.copy_english, state="disabled")
        self.copy_btn.pack(side="left", padx=(0, 6))

        io_frame = ttk.Frame(root, padding=10)
        io_frame.pack(fill="both", expand=True)

        input_label = ttk.Label(io_frame, text="Urdu (input):")
        input_label.grid(row=0, column=0, sticky="w")
        self.urdu_text = scrolledtext.ScrolledText(io_frame, wrap=tk.WORD, width=40, height=18,
                                                   font=("Noto Nastaliq Urdu", 12))
        self.urdu_text.grid(row=1, column=0, padx=(0, 10), pady=(6, 0))

        output_label = ttk.Label(io_frame, text="English (output):")
        output_label.grid(row=0, column=1, sticky="w")
        self.english_text = scrolledtext.ScrolledText(io_frame, wrap=tk.WORD, width=40, height=18,
                                                      font=("Segoe UI", 11))
        self.english_text.grid(row=1, column=1, padx=(10, 0), pady=(6, 0))

        footer = ttk.Frame(root, padding=(10, 6))
        footer.pack(fill="x")
        self.info_label = ttk.Label(footer, text="Type any Urdu sentence and press Translate.")
        self.info_label.pack(side="left")

        self.tokenizer = None
        self.model = None
        self.model_lock = threading.Lock()

        self.root.after(100, self.start_model_load)

    def start_model_load(self):
        self.load_btn.config(state="disabled")
        t = threading.Thread(target=self.load_model, daemon=True)
        t.start()

    def load_model(self):
        try:
            self.update_status("⏳ Loading model... (first time may take a while)")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            with self.model_lock:
                self.tokenizer = tokenizer
                self.model = model
            self.update_status(" Model loaded. Ready to translate.")
            self.root.after(0, lambda: self.translate_btn.config(state="normal"))
            self.root.after(0, lambda: self.copy_btn.config(state="normal"))
        except Exception as e:
            self.update_status(" Failed to load model.")
            messagebox.showerror("Model Load Error", f"Could not load model:\n{e}")
            self.root.after(0, lambda: self.load_btn.config(state="normal"))

    def update_status(self, text):
        self.status_label.config(text=text)

    def on_translate(self):
        text = self.urdu_text.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("No input", "Please type an Urdu sentence to translate.")
            return
        self.translate_btn.config(state="disabled")
        self.update_status(" Translating...")
        t = threading.Thread(target=self.translate_worker, args=(text,), daemon=True)
        t.start()

    def translate_worker(self, text):
        try:
            with self.model_lock:
                if self.model is None or self.tokenizer is None:
                    raise RuntimeError("Model not loaded.")
                tokenizer = self.tokenizer
                model = self.model

            inputs = tokenizer.encode(text, return_tensors="pt")
            output_tokens = model.generate(
                inputs,
                max_length=200,
                num_beams=5,
                early_stopping=True
            )
            english = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

            # Apply custom dictionary for proper names / phrases
            for urdu_word, eng_word in CUSTOM_DICT.items():
                if urdu_word in text:
                    english = english.replace(english,
                                              english.replace(english, eng_word) if urdu_word in english else english)
                    english = english.replace(english, english.replace(english, eng_word))

            # Better simple replacement (handles multiple words)
            for urdu_word, eng_word in CUSTOM_DICT.items():
                english = english.replace(urdu_word, eng_word)

            self.root.after(0, lambda: self.english_text.delete("1.0", tk.END))
            self.root.after(0, lambda: self.english_text.insert(tk.END, english))
            self.root.after(0, lambda: self.update_status(" Translation complete."))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Translation Error", str(e)))
            self.root.after(0, lambda: self.update_status(" Error during translation."))
        finally:
            self.root.after(0, lambda: self.translate_btn.config(state="normal"))

    def clear_texts(self):
        self.urdu_text.delete("1.0", tk.END)
        self.english_text.delete("1.0", tk.END)

    def copy_english(self):
        english = self.english_text.get("1.0", tk.END).strip()
        if english:
            self.root.clipboard_clear()
            self.root.clipboard_append(english)
            messagebox.showinfo("Copied", "English translation copied to clipboard.")
        else:
            messagebox.showinfo("Empty", "No English text to copy.")


if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    root.mainloop()
