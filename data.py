from model import EmbedModel

from bs4 import BeautifulSoup
import faiss
import os
import concurrent.futures
import requests
import torch

from datasets import Dataset
from sentence_transformers import SentenceTransformer


class KaatheData:
    def __init__(self):
        self.BASE_URL = "https://darksouls.wiki.fextralife.com"
        self.WEAPONS_URL = self.BASE_URL + "/Weapons"
        self.SHIELDS_URL = self.BASE_URL + "/Shields"
        self.MAGIC_URL = self.BASE_URL + "/Magic"
        self.UPGRADES_URL = self.BASE_URL + "/Upgrades"
        self.RINGS_URL = self.BASE_URL + "/Rings"
        self.ARMOR_URL = self.BASE_URL + "/Armor"
        self.ITEMS_URL = self.BASE_URL + "/Items"

        self._embed_model = EmbedModel()

    def _get_page_main_content(self, url: str) -> str:
        page = requests.get(url)
        soup = BeautifulSoup(page.text, "html.parser")
        main_content = soup.find(id="wiki-content-block")
        return main_content

    def _get_page_urls(self, url: str, remove_dmg_table: bool = False) -> list[str]:
        main_content = self._get_page_main_content(url)
        raw_links = [
            a.get("href")
            for a in main_content.find_all("a", href=True)
            if a.find("img")
        ]

        if remove_dmg_table:
            raw_links = raw_links[: raw_links.index("/Physical+Damage")]

        urls = []
        for link in raw_links:
            if "fextralife" not in link:
                urls.append(self.BASE_URL + link)
            else:
                urls.append(link)

        return urls

    def _get_page_contents(self, url: str) -> str:
        main_content = self._get_page_main_content(url)

        ret_str = ""

        for text in main_content.find_all(["p", "li", "h3"]):
            ret_str += text.get_text()

        for t in main_content.find(class_="wiki_table").find_all(["td", "th"]):
            if img := t.find("img"):
                ret_str += (
                    img["src"]
                    .removeprefix("/file/Dark-Souls/")
                    .removesuffix("_dark_souls.jpg")
                )
                ret_str += " "
            else:
                ret_str += t.get_text()
                ret_str += " "

        return ret_str.replace("\xa0", " ")

    def _save_to_disk(self, data, file_path: str = "./data") -> None:
        if type(data) == Dataset:
            data.drop_index("embeddings")
            data.save_to_disk(file_path + "/dataset.hf")

        else:
            for doc in data:
                file_path += f"/{hash(doc)}.txt"
                if not os.path.exists(file_path):
                    with open(file_path, "w+b") as f:
                        f.write(doc)

    def _convert_to_dataset(self, docs: list[str]) -> Dataset:
        dataset = Dataset.from_dict({"text": docs})
        return dataset

    def _add_embeddings_to_dataset(self, dataset: Dataset) -> Dataset:
        return self._embed_model.embed_dataset(dataset)

    def _download_documents(self) -> list[str]:
        # create a thread pool with 5 threads
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)

        equipment_docs = []
        for url in [
            self.WEAPONS_URL,
            self.SHIELDS_URL,
            self.ARMOR_URL,
            self.UPGRADES_URL,
        ]:
            pages = pool.submit(self._get_page_urls(url, remove_dmg_table=True))
            pages.append(url)
            for page in pages:
                equipment_docs.append(pool.submit(self._get_page_contents(page)))

        for url in [self.MAGIC_URL, self.RINGS_URL, self.ITEMS_URL]:
            pages = pool.submit(self._get_page_urls(url))
            pages.append(url)
            for page in pages:
                equipment_docs.append(pool.submit(self._get_page_contents(page)))

        # wait for all tasks to complete
        pool.shutdown(wait=True)

        return equipment_docs

    def get_data(self):
        docs = self._download_documents()
        # self._save_to_disk(docs)
        dataset = self._convert_to_dataset(docs)
        embed_dataset = self._add_embeddings_to_dataset(dataset)
        self._save_to_disk(embed_dataset)
