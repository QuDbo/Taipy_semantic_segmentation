from taipy.gui import Gui
from pages.examples import page_examples

pages = {"/":"""
Cette page est générée avec Taipy. Voir taipy.io pour plus d'informations
<|navbar|>""",
"Examples":page_examples}

if __name__=="__main__":
    gui = Gui(pages=pages)
    gui.run(port=5006)
