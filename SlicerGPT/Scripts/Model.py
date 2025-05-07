from Scripts.Utils import extract_mrml_scene_as_text

from transformers import AutoModelForCausalLM, AutoTokenizer
import slicer
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

FAISS_DIR = "./SlicerFAISS"

class Model:
    def __init__(self, manager, model_name="Qwen/Qwen3-0.6B"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.manager = manager
        # self.history = [{"role": "system", "content": f"You are a powerful and helpful AI, a '3D Slicer' software expert, and a great computer scientist with a huge knowlege on medical images. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer, try to guide the user with all the tools available on 3D Slicer."}]
        self.history = []
        self.has_history = True
        self.enable_thinking = False


    def generate_response(self, user_input):
        

        docs = self.manager.search(user_input) # Récupère les 3 documents les plus pertinents
        mrml_scene = extract_mrml_scene_as_text()
        context = "Here is the whole context I found : " + " ".join([doc.page_content for doc in docs]) + f"\n\nHere is the MRML scene in case of you need to look at the scene:\n{mrml_scene}\n\nUse the following pieces of context and your knowledge to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer, try to guide the user with all the tools available on 3D Slicer.\nAnd here is the user request, please be the most accurate and answer like if this was the user who directly asked this question to you : "
        messages = [{"role": "user", "content": context + user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

# if __name__ == "__main__":

    # manager = VectorStoreManager(FAISS_DIR)
    # chatbot = Model(manager=manager)
    # prompts = [
    #     'What is 3D Slicer?',
    #     'How to create a custom extension for 3D Slicer using Python?',
    #     'How to extract a volume using the Segment Editor module?',
    #     'What is the difference between vtkMRMLModelNode and vtkMRMLSegmentationNode?',
    #     'How to export a segmentation as an STL file using Python?',
    #     'How to load a large DICOM volume without slowing down Slicer?',
    #     'How to use the CLI module to automate a task in C++?',
    #     'What is the structure of a .mrml file in 3D Slicer?',
    #     'How to enable GPU acceleration for volume rendering?',
    #     'How to save a Python script as a module in Slicer?',
    #     'Can 3D Slicer run in headless mode (without GUI)?',
    #     'How to interface 3D Slicer with a DICOM PACS server?',
    #     'How to apply a smoothing filter to a 3D model in Slicer?',
    #     'How to automatically save modifications to a node?',
    #     'What is the best method to merge multiple segmentations?',
    #     'How to use the Elastix registration tool in Slicer?',
    # ]
    # import time
    
    # for pr in prompts:
    #     start = time.perf_counter()
    #     response = chatbot.generate_response(pr)
    #     end = time.perf_counter()
    #     print(pr)
    #     print(response)
    #     print(f"Generate in {end - start:.4f} seconds.")
    #     print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")