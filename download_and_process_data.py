from data_preparation.processing import process_data
from data_preparation.download_all import download_data


if __name__ == "__main__":

    # downloads, cleans and saves the data in: data/plain-text
    download_data()

    # trains a tokenizer with the train data, saves it in: objects/bpe_tokenizer
    #  and then uses it to convert all the data to input IDs and save it in
    #  data/processed-data
    #  it also processes the PoS data by creating sentences with PoS labels and
    #  saving a sklearn label encoder in: objects/le.pkl
    process_data()
