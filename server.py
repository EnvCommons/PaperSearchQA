from openreward.environments import Server

from papersearchqa import PaperSearchQA

if __name__ == "__main__":
    Server([PaperSearchQA]).run()
