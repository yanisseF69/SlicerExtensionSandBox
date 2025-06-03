from llama_cpp import Llama
import os
import math

num_cores = os.cpu_count()

FAISS_DIR = "./SlicerFAISS"

class Model:
    def __init__(self, manager, model_name="unsloth/Qwen3-0.6B-GGUF", file_name="Qwen3-0.6B-Q8_0.gguf"):

        self.llm = Llama.from_pretrained(
            repo_id=model_name,
            filename=file_name,
            verbose=False,
            n_ctx=40960,
            n_gpu_layers=-1,
            n_threads=1
        )
        # print(f"{math.ceil(num_cores/2)} thread instanciated.") 
        self.manager = manager
        self.history = [{
            "role": "system", 
            "content": (
                "You are a 3D Slicer assistant. Provide accurate, concise help with Python scripting and GUI operations. "
                "Use official documentation when possible. Give working code examples when the user ask it."
            )
        }]

        # self.history = []
        self.has_history = True
        self.enable_thinking = False

    def think(self):
        return " /think" if self.enable_thinking is True else " /no_think"


    def generate_response(self, user_input, mrml_scene):
        

        docs = self.manager.search(user_input, k=3) # Récupère les 3 documents les plus pertinents
        context = (
            "Context documents:\n"
            + "\n---\n".join([doc.page_content for doc in docs]) + "\n\n"

            "MRML Scene:\n"
            + mrml_scene + "\n\n"

            "Now, based on this context, the recent conversation, and your internal knowledge of 3D Slicer, "
            "answer the user's question as a real 3D Slicer expert would. "
            "Be technically accurate, easy to understand, and do not make up facts. "
            "Prefer a minimal working Python code snippet if the question involves scripting.\n\n"

            f"User question: {user_input}"
        )


        print("context ok")
        
        
        print("context ok")
        
        messages = self.history + [{"role": "user", "content": context + user_input + self.think()}]

        resp = self.llm.create_chat_completion(
            messages=messages,
        )

        response = resp["choices"][0]["message"]["content"]

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