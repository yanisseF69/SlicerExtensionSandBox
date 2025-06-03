
def extract_mrml_scene_as_text():
    """
    Extracts the current MRML scene in 3D Slicer, converts it to XML format,
    and returns the content as a string. Useful for injecting into an LLM via RAG.
    """
    import slicer
    import tempfile
    import os

    temp_file = tempfile.NamedTemporaryFile(suffix=".mrml", delete=False)
    temp_file_path = temp_file.name
    temp_file.close()

    try:
        slicer.mrmlScene.Commit(temp_file_path)

        with open(temp_file_path, 'r', encoding='utf-8') as file:
            mrml_text = file.read()

        return mrml_text
    finally:
        os.remove(temp_file_path)

def markdown_to_html(content: str) -> str:
    import re

    # Links : [text](url)
    content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', content)

    # Gras : **text**
    content = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', content)

    # Italique : *text*
    content = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', content)

    # Italic : <think>text</think> → <i>text</i>
    content = re.sub(r'<think>(.+?)</think>', r'<i>\1</i>', content, flags=re.DOTALL)

    # Titles : ### Title 3
    content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)

    # Titles : ## Title 2
    content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)

    # Bullet lists: - item
    content = re.sub(r'(?m)^- (.+)', r'<li>\1</li>', content)
    # Wrap <li> in <ul>
    if "<li>" in content:
        content = re.sub(r'((<li>.*?</li>\s*)+)', r'<ul>\1</ul>', content, flags=re.DOTALL)

    # <br>
    content = content.replace('\n', '<br>\n')

    return content

def list_nodes(filter_type="names", class_name=None, name=None, id=None):
    """
    List MRML nodes directly using the Slicer Python API.

    Parameters:
    - filter_type: specifies the type of information to retrieve ("names", "ids", or "properties").
    - class_name: filter by class name (optional).
    - name: filter by node name (optional).
    - id: filter by node ID (optional).

    Returns a dictionary with the node information.
    """
    import slicer
    
    try:
        nodes = slicer.mrmlScene.GetNodes()  # Get all nodes in the MRML scene
        result = {"nodes": []}

        for node in nodes:
            if filter_type == "names":
                # Collect node names
                result["nodes"].append(node.GetName())
            elif filter_type == "ids":
                # Collect node IDs
                result["nodes"].append(node.GetID())
            elif filter_type == "properties":
                # Collect node properties
                node_properties = {}
                node_properties["name"] = node.GetName()
                node_properties["id"] = node.GetID()
                # Add more properties as needed
                result["nodes"].append({node.GetName(): node_properties})

            # Apply filtering based on class_name, name, or id
            if class_name and node.GetClassName() != class_name:
                continue
            if name and node.GetName() != name:
                continue
            if id and node.GetID() != id:
                continue

        return result

    except Exception as e:
        return {"error": f"Node listing failed: {str(e)}"}

def execute_python_code(self, code: str) -> dict:
    """
    Execute Python code directly in 3D Slicer.

    Parameters:
    code (str): The Python code to execute.

    The code parameter is a string containing the Python code to be executed in 3D Slicer's Python environment.
    The code should be executable by Python's `exec()` function. To get return values, the code should assign the result to a variable named `__execResult`.

    Examples:
    - Create a sphere model: 
    execute_python_code_locally("sphere = slicer.vtkMRMLModelNode(); slicer.mrmlScene.AddNode(sphere); sphere.SetName('MySphere'); __execResult = sphere.GetID()")
    - Get the number of nodes in the current scene: 
    execute_python_code_locally("__execResult = len(slicer.mrmlScene.GetNodes())")
    - Calculate 1+1: 
    execute_python_code_locally("__execResult = 1 + 1")

    Returns:
        dict: A dictionary containing the execution result.

        If the code execution is successful, the dictionary will contain the following key-value pairs:
        - "success": True
        - "message": The result of the code execution. If the code assigns the result to `__execResult`, the value of `__execResult` is returned, otherwise it returns empty.

        If the code execution fails, the dictionary will contain the following key-value pairs:
        - "success": False
        - "message": A string containing an error message indicating the cause of the failure.
    """
    try:
        # Prepare the context to execute the code
        local_globals = globals().copy()
        local_globals['__execResult'] = None

        # Execute the Python code in the local environment
        exec(code, local_globals)
        
        # Check if the code has set __execResult
        result = local_globals.get('__execResult', None)
        
        if result is not None:
            return {"success": True, "message": result}
        else:
            return {"success": True, "message": "Code executed without setting __execResult."}

    except Exception as e:
        return {"success": False, "message": f"Execution failed: {str(e)}"}