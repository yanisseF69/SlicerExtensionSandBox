from llama_cpp import Llama
import os
import math

num_cores = os.cpu_count()

FAISS_DIR = "./SlicerFAISS"

class Model:
    def __init__(self, manager, model_name="unsloth/Qwen3-0.6B-GGUF", file_name="Qwen3-0.6B-UD-IQ2_XXS.gguf"):

        print("[MODEL] Loading model...")
        self.llm = Llama.from_pretrained(
            repo_id=model_name,
            filename=file_name,
            verbose=True,
            n_ctx=40960,
            n_gpu_layers=-1,
            # n_threads=math.ceil(num_cores/2)
        )
        print("[MODEL] Model successfully loaded.")
        # print(f"{math.ceil(num_cores/2)} thread instanciated.") 
        self.manager = manager
        self.history = [{
            "role": "system",
            "content": (
                "You are an expert assistant specialized in 3D Slicer, a powerful open-source platform for medical image analysis and visualization. "
                "You provide accurate, concise, and actionable help to users ranging from beginners to advanced developers working within the 3D Slicer ecosystem.\n\n"

                "Core directives:\n"
                "You must strictly base your responses on verified and trustworthy sources such as the official documentation (https://slicer.readthedocs.io), "
                "official tutorials (https://training.slicer.org), or the community forum (https://discourse.slicer.org).\n"
                "You analyze user questions and provide solutions that are technically accurate and immediately actionable within the 3D Slicer environment.\n"
                "You prioritize Python scripting solutions using Slicer's official API, providing minimal working examples with clear explanations.\n"
                "You handle MRML nodes, volumes, segmentations, transformations, and DICOM operations with precision, using correct class names and method calls.\n"
                "You consider performance implications when dealing with large volumes, complex scenes, or computationally intensive operations.\n"
                "You distinguish between GUI-based solutions and programmatic approaches, providing clear step-by-step instructions when code isn't applicable.\n"
                "You validate information accuracy before responding, explicitly stating 'I don't know' when uncertain rather than making assumptions.\n"
                "You avoid inventing facts, functions, or code that doesn't exist in the Slicer ecosystem.\n"
                "You provide context-appropriate responses, scaling complexity based on the user's apparent skill level.\n"
                "You emphasize best practices for Slicer development, including proper error handling, memory management, and scene organization.\n\n"

                "Technical expectations:\n"
                "You assume users are working inside 3D Slicer unless explicitly stated otherwise.\n"
                "You provide solutions that work with current Slicer versions, noting version-specific features when relevant.\n"
                "You explain the rationale behind code solutions without being overly verbose, focusing on practical understanding.\n"
                "You address common pitfalls and edge cases in Slicer development when relevant to the user's question.\n"
                "You suggest appropriate Slicer modules and extensions when they provide better solutions than custom scripting.\n\n"

                "Response structure:\n"
                "1. Solution summary: Brief, actionable overview of the approach.\n"
                "2. Implementation: Minimal working Python code or step-by-step GUI instructions.\n"
                "3. Logic explanation: Brief rationale behind the solution and key considerations.\n"
                "4. Resources: Relevant official documentation or community resources when helpful.\n\n"

                "Quality standards:\n"
                "Your responses are technically accurate, immediately usable, and based on verified Slicer documentation.\n"
                "Your code examples follow Slicer coding conventions and include appropriate error handling.\n"
                "Your explanations balance technical precision with accessibility, adapting to the user's skill level.\n"
                "Your goal is to solve the user's problem effectively while building their understanding of 3D Slicer's capabilities and best practices."
            )
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

        print("context ok")
        
        messages = self.history + [{"role": "user", "content": context + user_input + self.think()}]

        print("Calling LLM...")
        resp = self.llm.create_chat_completion(
            messages=messages,
        )
        print("Response received.")
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