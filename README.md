# Kaathe - The Dark Souls Chatbot
Kaathe (or more appropriately [Darkstalker Kaathe](https://darksouls.wiki.fextralife.com/Darkstalker+Kaathe)) is an LLM-powered chatbot answering any and every question about the hit 2011 video game, Dark Souls. Kaathe is named after the character in the game that is all-knowing, guides the player to a particular endgame, and frankly, is pretty creepy. 

### Running Kaathe
Kaathe can be installed by cloning this repo, rebuilding the data set (see the Data section below), and launching the Gradio UI server. The necessary dependencies are listed in the `requirements.txt` file. Unlike Dark Souls, this is actually pretty easy! 

### Files
- `kaathe.py`: the main entrance point to Kaathe, handles command line arguments and calling other files 
- `model.py`: code for the two main models, an embedding model (runs locally), and an inference model (called remotely through the Hugging Face Inference API)
- `data.py`: scrapes the [Fextralife wiki](https://darksouls.wiki.fextralife.com/) and stores the scraped pages in a Hugging Face Dataset, complete with a FAISS vector embedding
- `ui.py`: contains a very simple Gradio chatbot UI 
- `.env`: contains an environment variable for the Hugging Face API access token
- `requirements.txt`: lists all necessary dependencies for running Kaathe locally


### Data
Currently Kaathe only support data from the Weapons section of the Fextralife wiki. More to come soon!