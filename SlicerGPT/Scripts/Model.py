from Scripts.Utils import extract_mrml_scene_as_text

from llama_cpp import Llama
import slicer

FAISS_DIR = "./SlicerFAISS"

class Model:
    def __init__(self, manager, model_name="Qwen/Qwen3-0.6B-GGUF", file_name="Qwen3-0.6B-Q8_0.gguf"):

        self.llm = Llama.from_pretrained(
            repo_id=model_name,
            filename=file_name,
            verbose=True,
            n_ctx=40960,
            n_gpu_layers=30,
            n_threads=4
        )
        self.manager = manager
        self.history = [{"role": "system", "content": f"You are a powerful and helpful AI, a '3D Slicer' software expert, and a great computer scientist with a huge knowlege on medical images. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer, try to guide the user with all the tools available on 3D Slicer."}]
        self.history = []
        self.has_history = True
        self.enable_thinking = False

    def think(self):
        return " /think" if self.enable_thinking is True else " /no_think"


    def generate_response(self, user_input):
        

        docs = self.manager.search(user_input) # Récupère les 3 documents les plus pertinents
        mrml_scene = extract_mrml_scene_as_text()
        context = (
            "You are a helpful and knowledgeable assistant, an expert in the 3D Slicer software. "
            "Your goal is to answer user questions as precisely and reliably as possible, using only verified information. "
            "Below are context documents retrieved from the Slicer knowledge base, followed by the MRML scene for reference. "
            "Do not invent answers. If the context is insufficient, say 'I don't know' and suggest relevant tools or documentation in 3D Slicer that could help."
            "You can recommend the user to read the 3D Slicer documentation, forums in https://discourse.slicer.org, or tutorials in https://training.slicer.org/\n\n"
            
            "Context documents:\n"
            + "\n---\n".join([doc.page_content for doc in docs]) + "\n\n"
            
            "MRML Scene:\n"
            + mrml_scene + "\n\n"

            "Now, based on this context, the last messages sent and your internal knowledge of 3D Slicer, answer the following question as if you were a real expert talking to the user. "
            "Be concise, accurate, and do not make up facts.\n\n"
            
            f"User question: {user_input}"
        )

        print(self.think())
        
        messages = self.history + [{"role": "user", "content": context + user_input + self.think()}]

        resp = self.llm.create_chat_completion(
            messages = messages,
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