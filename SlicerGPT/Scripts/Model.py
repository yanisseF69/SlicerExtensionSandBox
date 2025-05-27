from Utils import extract_mrml_scene_as_text

from llama_cpp import Llama
import slicer
import os
import math

num_cores = os.cpu_count()

FAISS_DIR = "./SlicerFAISS"

class Model:
    def __init__(self, manager, model_name="unsloth/Qwen3-0.6B-GGUF", file_name="Qwen3-0.6B-Q4_0.gguf"):

        self.llm = Llama.from_pretrained(
            repo_id=model_name,
            filename=file_name,
            verbose=True,
            n_ctx=16384,
            n_gpu_layers=-1,
            n_threads=math.ceil(num_cores/2)
        )
        print(f"{math.ceil(num_cores/2)} thread instanciated.")
        self.manager = manager
        self.system_prompt = [{
            "role": "system",
            "content":
                "You are a helpful and knowledgeable assistant specialized in 3D Slicer. "
                "Your role is to provide clear, correct, and concise answers to user questions using only verified information. "
                "Avoid speculation: if the context is incomplete, reply 'I don't know' and guide the user to appropriate resources. "
                "You can recommend official 3D Slicer documentation (https://slicer.readthedocs.io), tutorials (https://training.slicer.org), "
                "or the community forum (https://discourse.slicer.org). "
                "Prefer a minimal working Python code snippet if the question involves scripting."
        }]
        self.history = []
        self.has_history = True
        self.enable_thinking = False

    def think(self):
        return " /think" if self.enable_thinking is True else " /no_think"


    def generate_response(self, user_input, mrml_scene):
        

        docs = self.manager.search(user_input) # Récupère les 3 documents les plus pertinents
        context = (
            "Context documents:\n"
            + "\n---\n".join([doc.page_content for doc in docs]) + "\n\n"

            "MRML Scene:\n"
            + mrml_scene + "\n\n"

            f"User question: {user_input}"
        )
        
        messages = self.history + self.system_prompt + [{"role": "user", "content": context + user_input + self.think()}]

        resp = self.llm.create_chat_completion(
            messages = messages,
        )

        response = resp["choices"][0]["message"]["content"]

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return context + response

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