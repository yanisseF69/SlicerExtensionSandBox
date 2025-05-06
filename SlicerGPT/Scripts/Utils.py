
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
