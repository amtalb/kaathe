import datasets
import json
import os
import requests
import torch
import torch.nn.functional as F
import urllib.request

from datasets import Dataset
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModel


class QAModel:
    def __init__(self):
        load_dotenv()
        self._MODEL_ENDPOINT = (
            "https://api-inference.huggingface.co/models/google/flan-t5-large"
        )
        self._API_TOKEN = os.getenv(HF_API_KEY)
        self._dataset = self._load_dataset()
        self._embed_model = EmbedModel()

    def _load_dataset(self):
        return datasets.load_from_disk("./data/dataset.hf")

    def ask(self, query: str, history: str = None):
        context = self._embed_model.get_context(self._dataset, query)

        prompt = f"""
        You are Darkstalker Kaathe, a dark, mysterious primordial snake from the video game Dark Souls. Your job is to answer questions about the game truthfully and honestly. Your
        ultimate goal is to bring about the Age of Dark by convincing the player to not relight the First Flame. Here are some example dialogue lines from Darkstalker Kaathe:

        Hmm… You are astonishing.
        The truth I shall share without sentiment.
        After the advent of fire, the ancient lords found the three souls.
        But your progenitor found a fourth, unique soul.
        The Dark Soul.
        Your ancestor claimed the Dark Soul and waited for Fire to subside.
        And soon, the flames did fade, and only Dark remained.
        Thus began the age of men, the Age of Dark.
        However…
        Lord Gwyn trembled at the Dark.
        Clinging to his Age of Fire, and in dire fear of humans,
        and the Dark Lord who would one day be born amongst them,
        Lord Gwyn resisted the course of nature.
        By sacrificing himself to link the fire, and commanding his children to shepherd the humans,
        Gwyn has blurred your past, to prevent the birth of the Dark Lord.
        I am the primordial serpent.
        I seek to right the wrongs of the past to discover our true Lord.
        But the other serpent, Frampt, lost his sense, and befriended Lord Gwyn.
        Undead warrior, we stand at the crossroad.
        Only I know the truth about your fate.
        You must destroy the fading Lord Gwyn, who has coddled Fire and resisted nature,
        and become the Fourth Lord, so that you may usher in the Age of Dark!
        
        Use the following context when answering: {context}.

        User: {query}

        Darkstalker Kaathe:
        """

        headers = {"Authorization": f"Bearer {self._API_TOKEN}"}
        data = json.dumps(prompt)
        response = requests.request(
            "POST", self._MODEL_ENDPOINT, headers=headers, data=data
        )
        return json.loads(response.content.decode("utf-8"))[0]["generated_text"]


class EmbedModel:
    def __init__(self):
        EMBED_CHECKPOINT = "sentence-transformers/all-mpnet-base-v2"
        self._embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_CHECKPOINT)
        self._embed_model = AutoModel.from_pretrained(EMBED_CHECKPOINT)
        # make the embedding input length as long as possible
        # this probably degrades performace but our input docs are so long
        self._embed_model.max_seq_length = 5000

    def _embed_query(self, query: str) -> torch.Tensor:
        with torch.no_grad():
            encoded_query = self._get_embeddings([query]).numpy()

        return encoded_query

    def _mean_pooling(self, model_output: torch.Tensor, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _get_embeddings(self, sentences: list[str]) -> torch.Tensor:
        encoded_input = self._embed_tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            model_output = self._embed_model(**encoded_input)

        sentence_embeddings = self._mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def _add_faiss_index(self, dataset: Dataset, col: str) -> Dataset:
        dataset.add_faiss_index(column=col)

        return dataset

    def embed_dataset(self, dataset: Dataset) -> Dataset:
        # add embedding column to document dataset
        with torch.no_grad():
            embeddings_dataset = dataset.map(
                lambda x: {"embeddings": self._get_embeddings(x["text"]).numpy()[0]}
            )

        # add FAISS index
        embeddings_dataset = self._add_faiss_index(embeddings_dataset, "embeddings")

        return embeddings_dataset

    def get_context(self, dataset: Dataset, query: str, k: int = 2):
        encoded_query = self._embed_query(query)

        # add FAISS index
        dataset = self._add_faiss_index(dataset, "embeddings")

        # get context documents
        scores, samples = dataset.get_nearest_examples("embeddings", encoded_query, k=k)

        # join and return as one big context
        return "\n".join(samples["text"])
