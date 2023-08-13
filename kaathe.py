import os
import sys

from data import KaatheData

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--refresh-data" or sys.argv[1] == "-d":
            kd = KaatheData()
            kd.get_data()
        else:
            print(f"Incorrect flag: {sys.argv[1]}")
            print()
            print("Valid flags:")
            print(
                "\t--refresh-data, -d: Scrapes the fextralife wiki and builds an embedding data set for each page."
            )
    else:
        os.system("gradio ui.py")
