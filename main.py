from taipy.gui import Gui, Markdown, notify
from pages.examples import page_examples

pages = {"/":"<|navbar|>",
         "Examples":page_examples}

if __name__=="__main__":
    gui = Gui(pages=pages)
    gui.run(port=5006)
